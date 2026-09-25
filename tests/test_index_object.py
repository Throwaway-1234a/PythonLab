import tempfile
from collections.abc import Mapping
from pathlib import Path

from findex.index import DocMeta, Index, Posting
from findex.ranking import BM25, TfIdf
from findex.utils import open_index


def test_index_mapping_dunders() -> None:
    idx = Index()
    idx.postings["whale"] = [Posting(0, 5)]
    idx.doc_meta[0] = DocMeta(0, "Moby Dick", "moby.txt")
    idx.doc_lengths[0] = 50

    assert isinstance(idx, Mapping)
    assert len(idx) == 1
    assert "whale" in idx
    assert "shark" not in idx
    assert idx["whale"] == [Posting(0, 5)]
    assert list(iter(idx)) == ["whale"]
    assert "Index(terms=1, docs=1)" in repr(idx)


def test_properties_and_helpers() -> None:
    idx = Index()
    idx.doc_meta = {0: DocMeta(0, "A", "a.txt"), 1: DocMeta(1, "B", "b.txt")}
    idx.doc_lengths = {0: 100, 1: 200}
    idx.postings["whale"] = [Posting(0, 2), Posting(1, 1)]

    assert idx.num_docs == 2
    assert idx.doc_length(0) == 100
    assert idx.avg_doc_length == 150.0
    assert idx.df("whale") == 2
    assert idx.df("missing") == 0


def test_open_index_context_manager() -> None:
    idx = Index()
    idx.doc_meta[0] = DocMeta(0, "Hamlet", "hamlet.txt")
    idx.doc_lengths[0] = 300
    idx.postings["ghost"] = [Posting(0, 3)]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.pkl"
        idx.save(path)

        with open_index(path) as opened:
            assert isinstance(opened, Index)
            assert "ghost" in opened
            assert opened.num_docs == 1


def test_scorers_adhere_to_protocol() -> None:
    idx = Index()
    idx.doc_meta = {0: DocMeta(0, "A", "a.txt")}
    idx.doc_lengths = {0: 100}
    idx.postings["term"] = [Posting(0, 2)]

    tfidf = TfIdf()
    bm25 = BM25()

    p = idx["term"][0]
    score_tfidf = tfidf.score("term", p, idx)
    score_bm25 = bm25.score("term", p, idx)

    assert score_tfidf > 0.0
    assert score_bm25 > 0.0
