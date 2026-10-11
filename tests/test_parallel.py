import pickle
from pathlib import Path

import pytest

from findex.parallel import build_index_parallel, build_partial


def test_executors_produce_identical_indexes(corpus_dir: Path) -> None:
    idx_serial, _ = build_index_parallel(corpus_dir, executor="serial", workers=1)
    idx_threads, _ = build_index_parallel(corpus_dir, executor="threads", workers=2)
    idx_processes, _ = build_index_parallel(corpus_dir, executor="processes", workers=2)

    assert len(idx_serial) == len(idx_threads) == len(idx_processes)
    assert idx_serial.doc_meta == idx_threads.doc_meta == idx_processes.doc_meta
    assert (
        idx_serial.doc_lengths == idx_threads.doc_lengths == idx_processes.doc_lengths
    )

    for term in idx_serial:
        assert idx_serial[term] == idx_threads[term] == idx_processes[term]

    dump_serial = pickle.dumps(
        (dict(idx_serial.postings), idx_serial.doc_lengths, idx_serial.doc_meta)
    )
    dump_threads = pickle.dumps(
        (dict(idx_threads.postings), idx_threads.doc_lengths, idx_threads.doc_meta)
    )
    dump_processes = pickle.dumps(
        (
            dict(idx_processes.postings),
            idx_processes.doc_lengths,
            idx_processes.doc_meta,
        )
    )

    assert dump_serial == dump_threads == dump_processes


def test_worker_exception_propagates() -> None:
    with pytest.raises(FileNotFoundError):
        build_partial([Path("non_existent_file_12345.txt")])
