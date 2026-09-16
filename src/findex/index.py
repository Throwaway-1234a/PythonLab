import argparse
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    title: str
    path: str


class InvertedIndex:
    def __init__(
        self,
        postings: dict[str, list[Posting]] | None = None,
        doc_lengths: dict[int, int] | None = None,
        doc_meta: dict[int, DocMeta] | None = None,
    ) -> None:
        self.postings: dict[str, list[Posting]] = (
            postings if postings is not None else defaultdict(list)
        )
        self.doc_lengths: dict[int, int] = (
            doc_lengths if doc_lengths is not None else {}
        )
        self.doc_meta: dict[int, DocMeta] = doc_meta if doc_meta is not None else {}

    def __len__(self) -> int:
        return len(self.postings)

    @classmethod
    def build(cls, corpus_dir: Path, record_positions: bool = False) -> "InvertedIndex":
        idx = cls()

        for doc in iter_documents(corpus_dir):
            doc_id = len(idx.doc_meta)
            title = doc.doc_id.replace("-", " ").replace("_", " ").title()
            idx.doc_meta[doc_id] = DocMeta(
                doc_id=doc_id, title=title, path=str(doc.path)
            )

            if record_positions:
                term_positions: dict[str, list[int]] = defaultdict(list)
                total_tokens = 0
                for pos, tok in enumerate(tokenize(doc.text)):
                    term_positions[tok].append(pos)
                    total_tokens += 1

                idx.doc_lengths[doc_id] = total_tokens
                for term, positions in term_positions.items():
                    idx.postings[term].append(
                        Posting(
                            doc_id=doc_id, tf=len(positions), positions=tuple(positions)
                        )
                    )
            else:
                counts = Counter(tokenize(doc.text))
                idx.doc_lengths[doc_id] = sum(counts.values())
                for term, count in counts.items():
                    idx.postings[term].append(Posting(doc_id=doc_id, tf=count))

        return idx


Posting.__module__ = "findex.index"
DocMeta.__module__ = "findex.index"
InvertedIndex.__module__ = "findex.index"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist an inverted index.")
    parser.add_argument("corpus", type=Path, help="Directory containing .txt documents")
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("index.bin"),
        help="Output index file path",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["pickle", "json"],
        default="pickle",
        help="Serialization format",
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Record token positions for phrase queries",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Error: corpus path '{args.corpus}' not found.", file=sys.stderr)
        sys.exit(1)

    tracemalloc.start()
    t0 = time.perf_counter()

    index = InvertedIndex.build(args.corpus, record_positions=args.positions)

    from findex.store import save

    save(index, args.out, fmt=args.format)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    file_size_mb = args.out.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 50)
    print("INDEX BUILD SUMMARY")
    print("=" * 50)
    print(f"Documents indexed:   {len(index.doc_meta):,}")
    print(f"Vocabulary size:     {len(index):,} terms")
    print(f"Positions recorded:  {args.positions}")
    print(f"Output path:         {args.out} ({file_size_mb:.2f} MB, {args.format})")
    print(f"Elapsed time:        {elapsed:.3f} s")
    print(f"Peak memory:         {peak_bytes / (1024 * 1024):.2f} MB")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
