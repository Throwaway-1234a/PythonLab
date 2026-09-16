import time
from pathlib import Path

from findex.search import execute_query
from findex.store import load

INDEX_PATH = Path("index.bin")


def main() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"Index {INDEX_PATH} not found. Build it first with: python -m findex.index data/corpus --out index.bin"
        )

    index = load(INDEX_PATH)
    sorted_terms = sorted(
        index.postings.keys(), key=lambda t: len(index.postings[t]), reverse=True
    )
    common_pair = (sorted_terms[0], sorted_terms[1])
    rare_pair = (sorted_terms[-1], sorted_terms[-2])

    test_queries = [
        ("Two Most Common", f"{common_pair[0]} AND {common_pair[1]}"),
        ("Two Rarest", f"{rare_pair[0]} AND {rare_pair[1]}"),
    ]

    print("\nBenchmarking Query Engines (1,000 iterations per query):\n")
    print("| Term Category | Query | Engine | Latency |")
    print("|:---|:---|:---|:---:|")

    for category, q in test_queries:
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = execute_query(index, q, engine="merge")
        merge_ms = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(1000):
            _ = execute_query(index, q, engine="set")
        set_ms = time.perf_counter() - t0

        print(f"| {category} | `{q}` | merge (two-pointer) | {merge_ms:.4f} s |")
        print(f"| {category} | `{q}` | set (hash set)      | {set_ms:.4f} s |")


if __name__ == "__main__":
    main()
