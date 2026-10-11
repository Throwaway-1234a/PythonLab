import os
import resource
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

from findex.parallel import build_index_parallel

CORPUS_DIR = Path("data/corpus")
RUNS = 3


def get_peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_configuration(executor: str, workers: int) -> tuple[float, float, float, float]:
    wall_times: list[float] = []
    cpu_times: list[float] = []
    merge_times: list[float] = []

    for _ in range(RUNS):
        cpu_start = time.process_time()
        wall_start = time.perf_counter()

        _, m_time = build_index_parallel(CORPUS_DIR, executor=executor, workers=workers)  # type: ignore[arg-type]

        wall_times.append(time.perf_counter() - wall_start)
        cpu_times.append(time.process_time() - cpu_start)
        merge_times.append(m_time)

    return (
        statistics.median(wall_times),
        statistics.median(cpu_times),
        get_peak_rss_mb(),
        statistics.median(merge_times),
    )


def main() -> None:
    if not CORPUS_DIR.exists():
        raise SystemExit(f"Corpus directory '{CORPUS_DIR}' not found.")

    cpu_count = os.cpu_count() or 4
    worker_counts = sorted({1, 2, 4, 8, cpu_count})
    gil_active = getattr(sys, "_is_gil_enabled", lambda: True)()

    print(
        f"Platform: Python {sys.version.split()[0]} on {sys.platform} ({cpu_count} logical cores)"
    )
    print(f"GIL Active: {gil_active}\n")

    print("Running baseline (serial, 1 worker)...")
    base_wall, base_cpu, base_rss, base_merge = run_configuration("serial", 1)

    results: list[dict[str, object]] = []
    results.append(
        {
            "executor": "serial",
            "workers": 1,
            "wall": base_wall,
            "cpu": base_cpu,
            "rss": base_rss,
            "merge": base_merge,
            "speedup": 1.0,
        }
    )

    threads_data: dict[int, float] = {1: 1.0}
    processes_data: dict[int, float] = {1: 1.0}

    for w in worker_counts:
        if w == 1:
            continue
        print(f"Running threads ({w} workers)...")
        w_wall, w_cpu, w_rss, w_merge = run_configuration("threads", w)
        sp = base_wall / w_wall
        threads_data[w] = sp
        results.append(
            {
                "executor": "threads (3.13t)" if not gil_active else "threads",
                "workers": w,
                "wall": w_wall,
                "cpu": w_cpu,
                "rss": w_rss,
                "merge": w_merge,
                "speedup": sp,
            }
        )

    for w in worker_counts:
        if w == 1:
            continue
        print(f"Running processes ({w} workers)...")
        w_wall, w_cpu, w_rss, w_merge = run_configuration("processes", w)
        sp = base_wall / w_wall
        processes_data[w] = sp
        results.append(
            {
                "executor": "processes",
                "workers": w,
                "wall": w_wall,
                "cpu": w_cpu,
                "rss": w_rss,
                "merge": w_merge,
                "speedup": sp,
            }
        )

    print("\n" + "=" * 80)
    print(
        f"| {'Executor':16} | {'Workers':7} | {'Wall (s, med)':14} | {'CPU (s)':9} | {'Peak RSS':10} | {'Merge (s)':10} | {'Speedup':8} |"
    )
    print(
        f"|:{'-' * 16}-|-{'-' * 7}:|-{'-' * 14}:|-{'-' * 9}:|-{'-' * 10}:|-{'-' * 10}:|-{'-' * 8}:|"
    )
    for r in results:
        print(
            f"| {r['executor']!s:16} | {int(r['workers']):7d} | {float(r['wall']):13.3f}s | "
            f"{float(r['cpu']):8.2f}s | {float(r['rss']):7.1f} MB | {float(r['merge']):9.3f}s | "
            f"{float(r['speedup']):7.2f}x |"
        )
    print("=" * 80 + "\n")

    Path("docs").mkdir(parents=True, exist_ok=True)
    plot_path = Path("docs/speedup_plot.png")

    plt.figure(figsize=(8, 5))
    xs = sorted(set(worker_counts))
    plt.plot(xs, xs, "k--", label="Ideal Linear (y = x)")
    plt.plot(
        list(processes_data.keys()),
        list(processes_data.values()),
        "o-",
        color="#2da44e",
        label="processes (ProcessPool)",
    )
    plt.plot(
        list(threads_data.keys()),
        list(threads_data.values()),
        "s-",
        color="#cf222e",
        label="threads (GIL)" if gil_active else "threads (3.13t)",
    )

    plt.xlabel("Workers (Cores)")
    plt.ylabel("Speedup vs. Serial (1.0x)")
    plt.title("Speedup vs. Workers on Gutenberg Corpus")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved speedup plot to {plot_path}")


if __name__ == "__main__":
    main()
