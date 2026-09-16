import time
import tracemalloc
from pathlib import Path

from findex.index import InvertedIndex
from findex.store import save

CORPUS_DIR = Path("data/corpus")


def measure(positions: bool):
    tracemalloc.start()
    t0 = time.perf_counter()
    idx = InvertedIndex.build(CORPUS_DIR, record_positions=positions)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tmp_path = Path("data") / f"index_pos_{positions}.pkl"
    save(idx, tmp_path, fmt="pickle")
    size_mb = tmp_path.stat().st_size / (1024 * 1024)
    tmp_path.unlink(missing_ok=True)

    return elapsed, peak / (1024 * 1024), size_mb


def main() -> None:
    print("Measuring Positions Flag Cost on Corpus...\n")
    t_no, mem_no, sz_no = measure(positions=False)
    t_pos, mem_pos, sz_pos = measure(positions=True)

    print("| Configuration | Build Time | Peak Memory | File Size |")
    print("|:---|:---:|:---:|:---:|")
    print(
        f"| Without positions (doc_id, tf) | {t_no:.3f} s | {mem_no:.2f} MB | {sz_no:.2f} MB |"
    )
    print(
        f"| With positions (doc_id, tf, positions) | {t_pos:.3f} s | {mem_pos:.2f} MB | {sz_pos:.2f} MB |"
    )


if __name__ == "__main__":
    main()
