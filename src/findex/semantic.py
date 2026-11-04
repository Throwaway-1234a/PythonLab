import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from findex.ranking import SearchResult

if TYPE_CHECKING:
    from findex.index import Index

log = logging.getLogger("findex.semantic")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_text(text: str, max_words: int = 250) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[:max_words])]


def build_embeddings(
    index: "Index",
    output_dir: Path,
    model_name: str = MODEL_NAME,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "chunk_meta.json"

    if not index.raw_docs and index.doc_meta:
        for doc_id, meta in index.doc_meta.items():
            p = Path(meta.path)
            if p.exists():
                index.raw_docs[doc_id] = p.read_text(encoding="utf-8", errors="replace")
            else:
                fallback = Path("data/corpus") / p.name
                if fallback.exists():
                    index.raw_docs[doc_id] = fallback.read_text(
                        encoding="utf-8", errors="replace"
                    )

    all_chunks: list[str] = []
    chunk_map: list[dict[str, Any]] = []

    for doc_id, text in index.raw_docs.items():
        doc_chunks = chunk_text(text)
        for c in doc_chunks:
            all_chunks.append(c)
            chunk_map.append({"doc_id": doc_id, "text": c[:200]})

    if not all_chunks:
        raise ValueError(
            "No text found to embed. Ensure data/corpus contains documents."
        )

    model = TextEmbedding(model_name)
    batch_size = 64
    embeddings_list: list[np.ndarray] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            f"Embedding {len(all_chunks)} chunks...", total=len(all_chunks)
        )
        for emb in model.embed(all_chunks, batch_size=batch_size):
            embeddings_list.append(emb)
            progress.advance(task, advance=1)

    embeddings = np.vstack(embeddings_list).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = embeddings / norms

    np.save(npy_path, normalized_embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f)

    return npy_path, meta_path


class SemanticSearcher:
    def __init__(
        self,
        npy_path: Path,
        meta_path: Path,
        index: "Index",
        model_name: str = MODEL_NAME,
    ) -> None:
        self.embeddings = np.load(npy_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunk_map: list[dict[str, Any]] = json.load(f)
        self.index = index
        self.model = TextEmbedding(model_name)

    def search_semantic(self, query: str, top_k: int = 10) -> list[SearchResult]:
        embed_res = next(iter(self.model.embed([query])))
        q_emb: NDArray[np.float32] = np.array(embed_res, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_emb))
        if q_norm > 0:
            q_emb /= q_norm

        sims: NDArray[np.float32] = cast(NDArray[np.float32], self.embeddings @ q_emb)

        doc_best_score: dict[int, float] = {}
        doc_best_chunk: dict[int, str] = {}

        for idx, score in enumerate(sims):
            doc_id = int(self.chunk_map[idx]["doc_id"])
            score_val = float(score)
            if score_val > doc_best_score.get(doc_id, -1.0):
                doc_best_score[doc_id] = score_val
                doc_best_chunk[doc_id] = str(self.chunk_map[idx]["text"])

        sorted_docs = sorted(doc_best_score.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        results: list[SearchResult] = []
        for doc_id, score in sorted_docs:
            title = (
                self.index.doc_meta[doc_id].title
                if doc_id in self.index.doc_meta
                else f"Doc {doc_id}"
            )
            snippet = doc_best_chunk.get(doc_id, "") + "..."
            results.append(
                SearchResult(doc_id=doc_id, score=score, title=title, snippet=snippet)
            )

        return results


def reciprocal_rank_fusion(
    keyword_results: list[SearchResult],
    semantic_results: list[SearchResult],
    k: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    rrf_scores: dict[int, float] = defaultdict(float)
    titles: dict[int, str] = {}
    snippets: dict[int, str] = {}

    for rank, item in enumerate(keyword_results, start=1):
        rrf_scores[item.doc_id] += 1.0 / (k + rank)
        titles[item.doc_id] = item.title
        snippets[item.doc_id] = item.snippet

    for rank, item in enumerate(semantic_results, start=1):
        rrf_scores[item.doc_id] += 1.0 / (k + rank)
        titles[item.doc_id] = item.title
        if item.doc_id not in snippets or not snippets[item.doc_id]:
            snippets[item.doc_id] = item.snippet

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        SearchResult(
            doc_id=doc_id,
            score=score,
            title=titles[doc_id],
            snippet=snippets.get(doc_id, ""),
        )
        for doc_id, score in sorted_docs
    ]
