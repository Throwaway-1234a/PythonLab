import time
from pathlib import Path

from findex.index import InvertedIndex
from findex.store import load, save

CORPUS_DIR = Path("data/corpus")


def main() -> None:
    print("Building in-memory index from corpus...")
    idx = InvertedIndex.build(CORPUS_DIR)

    tmp_dir = Path("data")
    pkl_path = tmp_dir / "bench_index.pkl"
    json_path = tmp_dir / "bench_index.json"

    t0 = time.perf_counter()
    save(idx, pkl_path, fmt="pickle")
    pkl_save = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = load(pkl_path, fmt="pickle")
    pkl_load = time.perf_counter() - t0
    pkl_size = pkl_path.stat().st_size / (1024 * 1024)

    t0 = time.perf_counter()
    save(idx, json_path, fmt="json")
    json_save = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = load(json_path, fmt="json")
    json_load = time.perf_counter() - t0
    json_size = json_path.stat().st_size / (1024 * 1024)

    print("\n| Format | File Size | Save Time | Load Time |")
    print("|:---|:---:|:---:|:---:|")
    print(
        f"| pickle (Protocol 5) | {pkl_size:.2f} MB | {pkl_save:.3f} s | {pkl_load:.3f} s |"
    )
    print(
        f"| json                | {json_size:.2f} MB | {json_save:.3f} s | {json_load:.3f} s |"
    )

    pkl_path.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
