import io
import pickle
import time
import tracemalloc
from array import array
from dataclasses import dataclass


@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class SlottedPosting:
    doc_id: int
    tf: int


N = 1_000_000


def test_plain():
    tracemalloc.start()
    t0 = time.perf_counter()
    postings = [PlainPosting(i, (i % 10) + 1) for i in range(N)]
    _ = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    buf = io.BytesIO()
    pickle.dump(postings, buf)
    file_size_mb = buf.tell() / (1024 * 1024)

    buf.seek(0)
    t0 = time.perf_counter()
    _ = pickle.load(buf)
    load_time = time.perf_counter() - t0

    return peak / (1024 * 1024), file_size_mb, load_time


def test_slotted():
    tracemalloc.start()
    t0 = time.perf_counter()
    postings = [SlottedPosting(i, (i % 10) + 1) for i in range(N)]
    _ = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    buf = io.BytesIO()
    pickle.dump(postings, buf)
    file_size_mb = buf.tell() / (1024 * 1024)

    buf.seek(0)
    t0 = time.perf_counter()
    _ = pickle.load(buf)
    load_time = time.perf_counter() - t0

    return peak / (1024 * 1024), file_size_mb, load_time


def test_primitive_arrays():
    tracemalloc.start()
    t0 = time.perf_counter()
    doc_ids = array("I", (i for i in range(N)))
    tfs = array("I", (((i % 10) + 1) for i in range(N)))
    _ = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    buf = io.BytesIO()
    pickle.dump((doc_ids, tfs), buf)
    file_size_mb = buf.tell() / (1024 * 1024)

    buf.seek(0)
    t0 = time.perf_counter()
    _ = pickle.load(buf)
    load_time = time.perf_counter() - t0

    return peak / (1024 * 1024), file_size_mb, load_time


def main() -> None:
    print(f"Measuring 3 Posting Storage Variants ({N:,} records)...\n")
    m1, s1, l1 = test_plain()
    m2, s2, l2 = test_slotted()
    m3, s3, l3 = test_primitive_arrays()

    print(
        "| Postings representation | Peak memory (build) | Index file size | Load time |"
    )
    print("|:---|:---:|:---:|:---:|")
    print(
        f"| list[Posting] with plain @dataclass | {m1:.2f} MB | {s1:.2f} MB | {l1:.3f} s |"
    )
    print(
        f"| list[Posting] with slots=True       | {m2:.2f} MB | {s2:.2f} MB | {l2:.3f} s |"
    )
    print(
        f"| array('I') pairs (doc_ids, tfs)     | {m3:.2f} MB | {s3:.2f} MB | {l3:.3f} s |"
    )


if __name__ == "__main__":
    main()
