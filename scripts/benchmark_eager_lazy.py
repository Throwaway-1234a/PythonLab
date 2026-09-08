import itertools
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize

CORPUS_DIR = Path("data/corpus")
SLICE_LIMIT = 500


def run_eager(root: Path, limit: int) -> tuple[int, int]:
    doc_stream = itertools.islice(iter_documents(root), limit)
    docs = [doc.text for doc in doc_stream]

    token_lists = [list(tokenize(text)) for text in docs]

    counter = Counter(token for sublist in token_lists for token in sublist)
    return len(docs), sum(counter.values())


def run_lazy(root: Path, limit: int) -> tuple[int, int]:
    doc_stream = itertools.islice(iter_documents(root), limit)
    counter: Counter[str] = Counter()
    doc_count = 0
    token_count = 0

    for doc in doc_stream:
        doc_count += 1
        for tok in tokenize(doc.text):
            counter[tok] += 1
            token_count += 1

    return doc_count, token_count


def benchmark(fn, label: str, limit: int) -> tuple[int, float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()

    doc_count, _ = fn(CORPUS_DIR, limit)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    return doc_count, peak_mb, elapsed


def main() -> None:
    if not CORPUS_DIR.exists():
        raise SystemExit(
            "Corpus directory missing. Run scripts/download_corpus.py first."
        )

    print(f"Benchmarking Eager vs. Lazy on {SLICE_LIMIT} documents...\n")
    docs_e, mem_e, time_e = benchmark(run_eager, "eager (lists)", SLICE_LIMIT)
    docs_l, mem_l, time_l = benchmark(run_lazy, "lazy (generators)", SLICE_LIMIT)

    print("| Version           | Documents | Peak memory | Elapsed  |")
    print("|:------------------|:---------:|:-----------:|:--------:|")
    print(f"| eager (lists)     | {docs_e:9d} | {mem_e:8.2f} MB | {time_e:6.3f} s |")
    print(f"| lazy (generators) | {docs_l:9d} | {mem_l:8.2f} MB | {time_l:6.3f} s |")


if __name__ == "__main__":
    main()
