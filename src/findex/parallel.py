import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from findex.index import DocMeta, Index, Posting
from findex.tokenize import tokenize


@dataclass(slots=True)
class PartialIndex:
    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    raw_docs: dict[int, str]


def build_partial(
    doc_paths: list[Path],
    base_doc_id: int = 0,
    record_positions: bool = True,
) -> PartialIndex:
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    raw_docs: dict[int, str] = {}

    for i, path in enumerate(doc_paths):
        doc_id = base_doc_id + i
        text = path.read_text(encoding="utf-8", errors="replace")
        title = path.stem.replace("-", " ").replace("_", " ").title()

        doc_meta[doc_id] = DocMeta(doc_id=doc_id, title=title, path=str(path))
        raw_docs[doc_id] = text

        term_positions: dict[str, list[int]] = defaultdict(list)
        total_tokens = 0
        for pos, tok in enumerate(tokenize(text)):
            term_positions[tok].append(pos)
            total_tokens += 1

        doc_lengths[doc_id] = total_tokens
        for term, pos_list in term_positions.items():
            if record_positions:
                postings[term].append(
                    Posting(doc_id=doc_id, tf=len(pos_list), positions=tuple(pos_list))
                )
            else:
                postings[term].append(Posting(doc_id=doc_id, tf=len(pos_list)))

    return PartialIndex(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        raw_docs=raw_docs,
    )


def _build_partial_unpack(args: tuple[list[Path], int, bool]) -> PartialIndex:
    return build_partial(args[0], base_doc_id=args[1], record_positions=args[2])


def merge(partials: Iterable[PartialIndex]) -> Index:
    merged_postings: dict[str, list[Posting]] = defaultdict(list)
    merged_doc_lengths: dict[int, int] = {}
    merged_doc_meta: dict[int, DocMeta] = {}
    merged_raw_docs: dict[int, str] = {}

    for part in partials:
        merged_doc_lengths.update(part.doc_lengths)
        merged_doc_meta.update(part.doc_meta)
        merged_raw_docs.update(part.raw_docs)
        for term, posts in part.postings.items():
            merged_postings[term].extend(posts)

    return Index(
        postings=merged_postings,
        doc_lengths=merged_doc_lengths,
        doc_meta=merged_doc_meta,
        raw_docs=merged_raw_docs,
    )


def _partition_paths(
    paths: Sequence[Path], n_chunks: int
) -> list[tuple[list[Path], int]]:
    if not paths or n_chunks <= 0:
        return []
    k, m = divmod(len(paths), n_chunks)
    chunks: list[tuple[list[Path], int]] = []
    current_doc_id = 0

    for i in range(n_chunks):
        chunk_len = k + (1 if i < m else 0)
        if chunk_len == 0:
            continue
        chunk_slice = list(paths[current_doc_id : current_doc_id + chunk_len])
        chunks.append((chunk_slice, current_doc_id))
        current_doc_id += chunk_len

    return chunks


def build_index_parallel(
    corpus_dir: Path,
    executor: Literal["serial", "threads", "processes"] = "processes",
    workers: int = 4,
    record_positions: bool = True,
) -> tuple[Index, float]:
    all_paths = sorted(corpus_dir.rglob("*.txt"))
    if not all_paths:
        return Index(), 0.0

    if executor == "serial" or workers <= 1:
        part = build_partial(
            all_paths, base_doc_id=0, record_positions=record_positions
        )
        t0 = time.perf_counter()
        idx = merge([part])
        merge_time = time.perf_counter() - t0
        return idx, merge_time

    chunks = _partition_paths(all_paths, workers)
    tasks = [
        (chunk_paths, base_id, record_positions) for chunk_paths, base_id in chunks
    ]

    if executor == "processes":
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            partials = list(pool.map(_build_partial_unpack, tasks))
    elif executor == "threads":
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            partials = list(pool.map(_build_partial_unpack, tasks))
    else:
        raise ValueError(f"Unknown executor: {executor}")

    t0 = time.perf_counter()
    idx = merge(partials)
    merge_time = time.perf_counter() - t0

    return idx, merge_time
