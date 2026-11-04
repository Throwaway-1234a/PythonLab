from pathlib import Path

from pytest_benchmark.fixture import BenchmarkFixture

from findex.index import Index


def test_benchmark_build_index(benchmark: BenchmarkFixture, corpus_dir: Path) -> None:
    benchmark(Index.build, corpus_dir, record_positions=False)


def test_benchmark_search(benchmark: BenchmarkFixture, corpus_dir: Path) -> None:
    idx = Index.build(corpus_dir, record_positions=False)
    benchmark(idx.search, "whale", k=5)
