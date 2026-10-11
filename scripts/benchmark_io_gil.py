import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CORPUS_DIR = Path("data/corpus")


def read_file(p: Path) -> int:
    return len(p.read_bytes())


def main() -> None:
    paths = sorted(CORPUS_DIR.rglob("*.txt"))
    if not paths:
        raise SystemExit("Corpus empty.")

    t0 = time.perf_counter()
    _ = [read_file(p) for p in paths]
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        _ = list(pool.map(read_file, paths))
    t_threads = time.perf_counter() - t0

    print("I/O-Bound Benchmark (Reading raw files off disk):")
    print(f"  Serial (1 worker):     {t_serial:.3f} s")
    print(
        f"  Threads (4 workers):   {t_threads:.3f} s (Speedup: {t_serial / t_threads:.2f}x)"
    )


if __name__ == "__main__":
    main()
