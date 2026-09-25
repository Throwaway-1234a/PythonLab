import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from findex.index import Posting

if TYPE_CHECKING:
    from findex.index import Index


@dataclass(order=True)
class SearchResult:
    doc_id: int = field(compare=False)
    score: float
    title: str = field(compare=False)
    snippet: str = field(default="", compare=False)


@runtime_checkable
class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: "Index") -> float: ...


class TfIdf:
    def score(self, term: str, posting: Posting, index: "Index") -> float:
        if posting.tf <= 0:
            return 0.0

        tf_weight = 1.0 + math.log(posting.tf)

        idf = math.log((index.num_docs + 1.0) / (index.df(term) + 1.0)) + 1.0

        dl = index.doc_length(posting.doc_id)
        norm = math.sqrt(dl) if dl > 0 else 1.0
        return (tf_weight * idf) / norm


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, term: str, posting: Posting, index: "Index") -> float:
        if posting.tf <= 0:
            return 0.0
        df = index.df(term)
        N = index.num_docs

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        dl = index.doc_length(posting.doc_id)
        avg_dl = index.avg_doc_length or 1.0
        norm = 1.0 - self.b + self.b * (dl / avg_dl)
        tf_component = (posting.tf * (self.k1 + 1.0)) / (posting.tf + self.k1 * norm)
        return idf * tf_component
