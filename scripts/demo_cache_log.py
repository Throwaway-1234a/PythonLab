import logging
from pathlib import Path

from findex.index import Index

logging.basicConfig(level=logging.INFO, format="%(message)s")
INDEX_PATH = Path("index.bin")


def main() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(f"Index {INDEX_PATH} not found.")

    idx = Index.load(INDEX_PATH)
    q = "whale AND captain"

    print(f"\n--- First run for: '{q}' (Cold cache) ---")
    _ = idx.search(q, k=5)

    print(f"\n--- Second run for: '{q}' (Hot cache hit) ---")
    _ = idx.search(q, k=5)

    print(f"\nCache Stats: {idx._evaluate_query_cached.cache_info()}")


if __name__ == "__main__":
    main()
