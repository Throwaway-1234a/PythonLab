import functools
import heapq
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from findex.corpus import iter_documents
from findex.snippet import make_snippet
from findex.tokenize import tokenize
from findex.utils import timed

if TYPE_CHECKING:
    from findex.query import Node
    from findex.ranking import Scorer, SearchResult


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    title: str
    path: str


Posting.__module__ = "findex.index"
DocMeta.__module__ = "findex.index"


class Index(Mapping[str, list[Posting]]):
    def __init__(
        self,
        postings: dict[str, list[Posting]] | None = None,
        doc_lengths: dict[int, int] | None = None,
        doc_meta: dict[int, DocMeta] | None = None,
        raw_docs: dict[int, str] | None = None,
    ) -> None:
        self.postings: dict[str, list[Posting]] = (
            postings if postings is not None else defaultdict(list)
        )
        self.doc_lengths: dict[int, int] = (
            doc_lengths if doc_lengths is not None else {}
        )
        self.doc_meta: dict[int, DocMeta] = doc_meta if doc_meta is not None else {}
        self.raw_docs: dict[int, str] = raw_docs if raw_docs is not None else {}

        self._evaluate_query_cached = functools.lru_cache(maxsize=256)(
            self._evaluate_query_uncached
        )

    def __len__(self) -> int:
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __getitem__(self, term: str) -> list[Posting]:
        term_norm = term.casefold()
        if term_norm not in self.postings:
            raise KeyError(term)
        return self.postings[term_norm]

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key.casefold() in self.postings
        return False

    def __repr__(self) -> str:
        return f"Index(terms={len(self):,}, docs={self.num_docs:,})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths.get(doc_id, 0)

    def df(self, term: str) -> int:
        return len(self.postings.get(term.casefold(), []))

    def get_postings(self, term: str) -> list[Posting]:
        return self.postings.get(term.casefold(), [])

    @classmethod
    @timed
    def build(cls, corpus_dir: Path, record_positions: bool = True) -> "Index":
        idx = cls()
        for doc in iter_documents(corpus_dir):
            doc_id = len(idx.doc_meta)
            title = doc.doc_id.replace("-", " ").replace("_", " ").title()
            idx.doc_meta[doc_id] = DocMeta(
                doc_id=doc_id, title=title, path=str(doc.path)
            )
            idx.raw_docs[doc_id] = doc.text

            term_positions: dict[str, list[int]] = defaultdict(list)
            total_tokens = 0
            for pos, tok in enumerate(tokenize(doc.text)):
                term_positions[tok].append(pos)
                total_tokens += 1

            idx.doc_lengths[doc_id] = total_tokens
            for term, positions in term_positions.items():
                if record_positions:
                    idx.postings[term].append(
                        Posting(
                            doc_id=doc_id, tf=len(positions), positions=tuple(positions)
                        )
                    )
                else:
                    idx.postings[term].append(Posting(doc_id=doc_id, tf=len(positions)))
        return idx

    @timed
    def save(self, path: Path) -> None:
        from findex.store import save as store_save

        store_save(self, path)

    @classmethod
    @timed
    def load(cls, path: Path) -> "Index":
        from findex.store import load as store_load

        return store_load(path)

    def _evaluate_query_uncached(self, query_str: str) -> frozenset[int]:
        from findex.query import parse

        ast = parse(query_str)
        return frozenset(ast.evaluate(self))

    @timed
    def search(
        self,
        query: "str | Node",
        scorer: "Scorer | None" = None,
        k: int = 10,
    ) -> list["SearchResult"]:
        from findex.query import parse
        from findex.ranking import BM25, SearchResult

        if scorer is None:
            scorer = BM25()

        if isinstance(query, str):
            ast = parse(query)
            matching_doc_ids = self._evaluate_query_cached(query)
        else:
            ast = query
            matching_doc_ids = frozenset(ast.evaluate(self))

        target_terms = ast.terms()
        candidates: list[SearchResult] = []

        for doc_id in matching_doc_ids:
            score = 0.0
            for term in target_terms:
                postings = self.postings.get(term, [])
                p = next((post for post in postings if post.doc_id == doc_id), None)
                if p is not None:
                    score += scorer.score(term, p, self)

            title = self.doc_meta[doc_id].title
            raw_text = self.raw_docs.get(doc_id, "")
            snippet = make_snippet(raw_text, target_terms)
            candidates.append(
                SearchResult(doc_id=doc_id, score=score, title=title, snippet=snippet)
            )

        return heapq.nlargest(k, candidates)


InvertedIndex = Index
InvertedIndex.__module__ = "findex.index"
Index.__module__ = "findex.index"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build and persist an inverted index.")
    parser.add_argument("corpus", type=Path, help="Directory containing text documents")
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("index.bin"),
        help="Output index destination path",
    )
    parser.add_argument(
        "--positions", action="store_true", help="Record token positions"
    )
    args = parser.parse_args()

    print(f"Building index from {args.corpus}...")
    idx = Index.build(args.corpus, record_positions=args.positions)
    idx.save(args.out)
    print(f"Successfully saved {idx.num_docs} documents to {args.out}")


if __name__ == "__main__":
    main()
