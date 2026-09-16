import tempfile
from pathlib import Path

from findex.index import DocMeta, InvertedIndex, Posting
from findex.search import (
    difference_postings,
    execute_query,
    intersect_postings,
    union_postings,
)
from findex.store import load, save


def test_dataclasses_frozen_slotted_and_hashable() -> None:
    p1 = Posting(doc_id=1, tf=2, positions=(0, 5))
    p2 = Posting(doc_id=1, tf=2, positions=(0, 5))
    m1 = DocMeta(doc_id=1, title="Test", path="test.txt")

    s = {p1, p2, m1}
    assert len(s) == 2
    assert p1 in s
    assert m1 in s

    assert not hasattr(p1, "__dict__")
    assert not hasattr(m1, "__dict__")


def test_two_pointer_merges() -> None:
    p1 = [Posting(1, 1), Posting(3, 2), Posting(5, 1)]
    p2 = [Posting(2, 1), Posting(3, 4), Posting(6, 1)]

    # AND
    assert [p.doc_id for p in intersect_postings(p1, p2)] == [3]
    # OR
    assert [p.doc_id for p in union_postings(p1, p2)] == [1, 2, 3, 5, 6]
    # AND NOT
    assert [p.doc_id for p in difference_postings(p1, p2)] == [1, 5]


def test_merge_and_set_engines_match() -> None:
    idx = InvertedIndex()
    idx.doc_meta = {
        i: DocMeta(doc_id=i, title=f"Doc {i}", path=f"doc_{i}.txt") for i in range(5)
    }
    idx.postings["whale"] = [Posting(0, 1), Posting(2, 3), Posting(4, 1)]
    idx.postings["captain"] = [Posting(1, 1), Posting(2, 1), Posting(3, 2)]
    idx.postings["sea"] = [Posting(2, 1), Posting(3, 1), Posting(4, 2)]

    queries = [
        "whale captain",
        "whale OR sea",
        "whale NOT captain",
        "whale AND sea NOT captain",
    ]

    for q in queries:
        res_merge = execute_query(idx, q, engine="merge")
        res_set = execute_query(idx, q, engine="set")
        assert res_merge == res_set, f"Mismatch on query: {q}"


def test_serialization_roundtrip_pickle_and_json() -> None:
    idx = InvertedIndex()
    idx.doc_meta = {0: DocMeta(doc_id=0, title="Hamlet", path="hamlet.txt")}
    idx.doc_lengths = {0: 100}
    idx.postings["ghost"] = [Posting(doc_id=0, tf=3, positions=(10, 25, 80))]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        pkl_path = root / "index.pkl"
        save(idx, pkl_path, fmt="pickle")
        loaded_pkl = load(pkl_path, fmt="pickle")
        assert loaded_pkl.postings["ghost"] == idx.postings["ghost"]
        assert loaded_pkl.doc_meta[0] == idx.doc_meta[0]

        json_path = root / "index.json"
        save(idx, json_path, fmt="json")
        loaded_json = load(json_path, fmt="json")
        assert loaded_json.postings["ghost"] == idx.postings["ghost"]
        assert loaded_json.doc_meta[0] == idx.doc_meta[0]
