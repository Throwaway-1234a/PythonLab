import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from findex.index import DocMeta, Index
from findex.ranking import SearchResult
from findex.snippet import make_snippet

if TYPE_CHECKING:
    from findex.query import Node


class VectorizedIndex:
    def __init__(
        self,
        doc_ids_map: dict[str, np.ndarray],
        tfs_map: dict[str, np.ndarray],
        doc_lengths: np.ndarray,
        doc_meta: dict[int, DocMeta],
        raw_docs: dict[int, str],
    ) -> None:
        self.doc_ids_map = doc_ids_map
        self.tfs_map = tfs_map
        self.doc_lengths = doc_lengths  # int32[N] array
        self.doc_meta = doc_meta
        self.raw_docs = raw_docs
        self.num_docs = len(doc_meta)
        self.avg_doc_length = (
            float(np.mean(doc_lengths)) if len(doc_lengths) > 0 else 1.0
        )

    @classmethod
    def from_index(cls, index: Index) -> "VectorizedIndex":
        doc_ids_map: dict[str, np.ndarray] = {}
        tfs_map: dict[str, np.ndarray] = {}

        for term, postings in index.postings.items():
            doc_ids_map[term] = np.array([p.doc_id for p in postings], dtype=np.int32)
            tfs_map[term] = np.array([p.tf for p in postings], dtype=np.int32)

        n = index.num_docs
        lengths_arr = np.zeros(n, dtype=np.int32)
        for doc_id, length in index.doc_lengths.items():
            if doc_id < n:
                lengths_arr[doc_id] = length

        return cls(
            doc_ids_map=doc_ids_map,
            tfs_map=tfs_map,
            doc_lengths=lengths_arr,
            doc_meta=index.doc_meta,
            raw_docs=index.raw_docs,
        )

    def search_vectorized(
        self,
        query: "str | Node",
        scorer: Literal["bm25", "tfidf"] = "bm25",
        k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[SearchResult]:
        from findex.query import Node, parse

        ast: Node = parse(query) if isinstance(query, str) else query
        query_terms = [t for t in ast.terms() if t in self.doc_ids_map]
        if not query_terms or self.num_docs == 0:
            return []

        scores = np.zeros(self.num_docs, dtype=np.float32)

        for term in query_terms:
            doc_ids = self.doc_ids_map[term]
            tfs = self.tfs_map[term].astype(np.float32)
            df = len(doc_ids)

            if scorer == "bm25":
                idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
                dls = self.doc_lengths[doc_ids].astype(np.float32)
                norm = 1.0 - b + b * (dls / self.avg_doc_length)
                term_scores = idf * (tfs * (k1 + 1.0)) / (tfs + k1 * norm)
                scores[doc_ids] += term_scores
            else:  # tfidf
                tf_weight = 1.0 + np.log(tfs)
                idf = math.log((self.num_docs + 1.0) / (df + 1.0)) + 1.0
                dls = np.sqrt(self.doc_lengths[doc_ids].astype(np.float32))
                dls[dls == 0] = 1.0
                scores[doc_ids] += (tf_weight * idf) / dls

        candidate_ids = np.flatnonzero(scores > 0.0)
        if len(candidate_ids) == 0:
            return []

        candidate_scores = scores[candidate_ids]
        top_k = min(k, len(candidate_ids))

        part_idx = np.argpartition(-candidate_scores, top_k - 1)[:top_k]
        top_sorted_local = part_idx[np.argsort(-candidate_scores[part_idx])]
        top_doc_ids = candidate_ids[top_sorted_local]

        target_set = set(query_terms)
        results: list[SearchResult] = []
        for doc_id_int in top_doc_ids:
            doc_id = int(doc_id_int)
            title = self.doc_meta[doc_id].title
            raw_text = self.raw_docs.get(doc_id, "")
            snippet = make_snippet(raw_text, target_set)
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=float(scores[doc_id]),
                    title=title,
                    snippet=snippet,
                )
            )

        return results
