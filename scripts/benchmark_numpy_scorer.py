import time
from pathlib import Path

from findex.index import Index
from findex.ranking import BM25
from findex.vectorized import VectorizedIndex

INDEX_PATH = Path("index.bin")


def measure(fn, iters: int = 100) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return ((time.perf_counter() - t0) / iters) * 1000.0


def main() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(f"Index {INDEX_PATH} not found.")

    idx = Index.load(INDEX_PATH)
    vec_idx = VectorizedIndex.from_index(idx)
    bm25 = BM25()

    sorted_terms = sorted(idx.postings.keys(), key=lambda t: len(idx.postings[t]))
    rare_term = next(t for t in sorted_terms if len(idx.postings[t]) >= 3)
    common_term = next(
        t for t in reversed(sorted_terms) if len(idx.postings[t]) > idx.num_docs * 0.6
    )

    queries = [
        ("common term", common_term),
        ("rare term", rare_term),
        ("3-term query", "whale ship captain"),
    ]

    print(
        f"\n| {'Query':16} | {'Docs matched':12} | {'Lab 3 scorer (ms)':17} | {'NumPy scorer (ms)':16} | {'Speedup':8} |"
    )
    print(f"|:{'-' * 16}-|-{'-' * 12}:|-{'-' * 17}:|-{'-' * 16}:|-{'-' * 8}:|")

    for label, q in queries:
        matched = len(idx.search(q, scorer=bm25, k=1000))
        t_py = measure(lambda query=q: idx.search(query, scorer=bm25, k=10))
        t_np = measure(
            lambda query=q: vec_idx.search_vectorized(query, scorer="bm25", k=10)
        )
        speedup = t_py / max(t_np, 1e-6)

        print(
            f"| {label:16} | {matched:12d} | {t_py:15.3f} ms | {t_np:14.3f} ms | {speedup:7.2f}x |"
        )


if __name__ == "__main__":
    main()
