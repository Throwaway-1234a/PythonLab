import cProfile
import io
import pstats
import time
from pathlib import Path

from findex.index import Index

INDEX_PATH = Path("index.bin")
CORPUS_DIR = Path("data/corpus")
QUERY = "whale ship captain"


def profile_indexing() -> None:
    print(f"\n--- Profiling 'findex index' on {CORPUS_DIR} ---")
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    _ = Index.build(CORPUS_DIR, record_positions=True)
    elapsed = time.perf_counter() - t0
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(s.getvalue())
    print(f"Total indexing duration: {elapsed:.3f} s")


def profile_search() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(f"Index {INDEX_PATH} not found.")

    idx = Index.load(INDEX_PATH)
    print(f"\n--- Profiling 'findex search' for '{QUERY}' (100 repetitions) ---")
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = idx.search(QUERY, k=10)
    elapsed = time.perf_counter() - t0
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(15)
    print(s.getvalue())
    print(f"Total search duration: {elapsed:.3f} s")


if __name__ == "__main__":
    profile_indexing()
    profile_search()
