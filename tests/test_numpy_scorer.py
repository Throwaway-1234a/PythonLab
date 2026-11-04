import math
from pathlib import Path

from findex.index import Index
from findex.ranking import BM25, TfIdf
from findex.vectorized import VectorizedIndex


def test_numpy_bm25_matches_pure_python(corpus_dir: Path) -> None:
    idx = Index.build(corpus_dir, record_positions=False)
    vec_idx = VectorizedIndex.from_index(idx)

    queries = ["whale", "ocean", "whale ocean"]
    for q in queries:
        res_py = idx.search(q, scorer=BM25(), k=5)
        res_np = vec_idx.search_vectorized(q, scorer="bm25", k=5)

        assert len(res_py) == len(res_np)
        for r_py, r_np in zip(res_py, res_np, strict=True):
            assert r_py.doc_id == r_np.doc_id
            assert math.isclose(r_py.score, r_np.score, rel_tol=1e-4)


def test_numpy_tfidf_matches_pure_python(corpus_dir: Path) -> None:
    idx = Index.build(corpus_dir, record_positions=False)
    vec_idx = VectorizedIndex.from_index(idx)

    res_py = idx.search("ocean", scorer=TfIdf(), k=5)
    res_np = vec_idx.search_vectorized("ocean", scorer="tfidf", k=5)

    assert len(res_py) == len(res_np)
    for r_py, r_np in zip(res_py, res_np, strict=True):
        assert r_py.doc_id == r_np.doc_id
        assert math.isclose(r_py.score, r_np.score, rel_tol=1e-4)
