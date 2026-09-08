import argparse
import itertools
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def compute_stats(
    root: Path, limit: int | None = None
) -> tuple[int, int, int, list[tuple[str, int]]]:
    doc_stream = iter_documents(root)
    if limit is not None:
        doc_stream = itertools.islice(doc_stream, limit)

    term_counter: Counter[str] = Counter()
    doc_count = 0
    token_count = 0

    for doc in doc_stream:
        doc_count += 1
        for token in tokenize(doc.text):
            term_counter[token] += 1
            token_count += 1

    vocab_size = len(term_counter)
    top_50 = term_counter.most_common(50)
    return doc_count, token_count, vocab_size, top_50


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute corpus statistics in streaming constant memory."
    )
    parser.add_argument("corpus", type=Path, help="Path to corpus directory")
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Process only first N documents"
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Error: directory '{args.corpus}' does not exist.", file=sys.stderr)
        sys.exit(1)

    tracemalloc.start()
    t0 = time.perf_counter()

    doc_count, token_count, vocab_size, top_50 = compute_stats(
        args.corpus, limit=args.limit
    )

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n" + "=" * 50)
    print("FINDEX CORPUS STATISTICS")
    print("=" * 50)
    print(f"Documents processed: {doc_count:,}")
    print(f"Total tokens:        {token_count:,}")
    print(f"Vocabulary size:     {vocab_size:,} distinct terms")
    print(f"Elapsed time:        {elapsed:.3f} s")
    print(f"Peak memory:         {peak_bytes / (1024 * 1024):.2f} MB")
    print("-" * 50)
    print("Top 10 terms:")
    for rank, (term, count) in enumerate(top_50[:10], start=1):
        print(f"  {rank:2d}. {term:15s} {count:,}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
