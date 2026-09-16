import argparse
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Literal

from findex.index import InvertedIndex, Posting
from findex.store import load


def intersect_postings(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """AND query:"""
    res: list[Posting] = []
    i = j = 0
    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            res.append(p1[i])
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            i += 1
        else:
            j += 1
    return res


def union_postings(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """OR query"""
    res: list[Posting] = []
    i = j = 0
    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            res.append(p1[i])
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            res.append(p1[i])
            i += 1
        else:
            res.append(p2[j])
            j += 1
    while i < len(p1):
        res.append(p1[i])
        i += 1
    while j < len(p2):
        res.append(p2[j])
        j += 1
    return res


def difference_postings(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """AND NOT query"""
    res: list[Posting] = []
    i = j = 0
    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            res.append(p1[i])
            i += 1
        else:
            j += 1
    while i < len(p1):
        res.append(p1[i])
        i += 1
    return res


def execute_query(
    index: InvertedIndex,
    query: str,
    engine: Literal["merge", "set"] = "merge",
) -> list[int]:
    raw_tokens = query.split()
    if not raw_tokens:
        return []

    tokens: list[str] = []
    for t in raw_tokens:
        up = t.upper()
        if up in ("AND", "OR", "NOT"):
            tokens.append(up)
        else:
            tokens.append(t.casefold())

    processed: list[str] = []
    for tok in tokens:
        if (
            processed
            and processed[-1] not in ("AND", "OR", "NOT")
            and tok not in ("AND", "OR", "NOT")
        ):
            processed.append("AND")
        processed.append(tok)

    if engine == "merge":
        return _eval_merge(index, processed)
    else:
        return _eval_set(index, processed)


def _eval_merge(index: InvertedIndex, tokens: list[str]) -> list[int]:
    all_docs = [Posting(doc_id=i, tf=0) for i in sorted(index.doc_meta.keys())]

    idx = 0

    if tokens[0] == "NOT":
        if len(tokens) < 2:
            return []
        current = difference_postings(all_docs, index.postings.get(tokens[1], []))
        idx = 2
    else:
        current = index.postings.get(tokens[0], [])
        idx = 1

    while idx < len(tokens):
        op = tokens[idx]
        idx += 1
        if idx >= len(tokens):
            break

        if op == "NOT":
            target = index.postings.get(tokens[idx], [])
            current = difference_postings(current, target)
            idx += 1
        elif op == "AND":
            if tokens[idx] == "NOT":
                idx += 1
                if idx >= len(tokens):
                    break
                target = index.postings.get(tokens[idx], [])
                current = difference_postings(current, target)
            else:
                target = index.postings.get(tokens[idx], [])
                current = intersect_postings(current, target)
            idx += 1
        elif op == "OR":
            target = index.postings.get(tokens[idx], [])
            current = union_postings(current, target)
            idx += 1

    return [p.doc_id for p in current]


def _eval_set(index: InvertedIndex, tokens: list[str]) -> list[int]:
    all_doc_ids = set(index.doc_meta.keys())

    idx = 0
    if tokens[0] == "NOT":
        if len(tokens) < 2:
            return []
        current = all_doc_ids - {p.doc_id for p in index.postings.get(tokens[1], [])}
        idx = 2
    else:
        current = {p.doc_id for p in index.postings.get(tokens[0], [])}
        idx = 1

    while idx < len(tokens):
        op = tokens[idx]
        idx += 1
        if idx >= len(tokens):
            break

        if op == "NOT":
            target = {p.doc_id for p in index.postings.get(tokens[idx], [])}
            current = current - target
            idx += 1
        elif op == "AND":
            if tokens[idx] == "NOT":
                idx += 1
                if idx >= len(tokens):
                    break
                target = {p.doc_id for p in index.postings.get(tokens[idx], [])}
                current = current - target
            else:
                target = {p.doc_id for p in index.postings.get(tokens[idx], [])}
                current = current & target
            idx += 1
        elif op == "OR":
            target = {p.doc_id for p in index.postings.get(tokens[idx], [])}
            current = current | target
            idx += 1

    return sorted(current)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query an inverted index.")
    parser.add_argument("index_path", type=Path, help="Path to serialized index file")
    parser.add_argument(
        "query", type=str, help="Search expression (supports AND, OR, NOT)"
    )
    parser.add_argument(
        "--engine",
        choices=["merge", "set"],
        default="merge",
        help="Query evaluation engine",
    )
    parser.add_argument(
        "--format",
        choices=["pickle", "json"],
        default="pickle",
        help="Index file format",
    )
    args = parser.parse_args()

    if not args.index_path.exists():
        print(f"Error: index file '{args.index_path}' not found.", file=sys.stderr)
        sys.exit(1)

    tracemalloc.start()
    t0 = time.perf_counter()

    index = load(args.index_path, fmt=args.format)
    matching_ids = execute_query(index, args.query, engine=args.engine)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n" + "=" * 50)
    print(f"SEARCH RESULTS: '{args.query}' (Engine: {args.engine})")
    print("=" * 50)
    print(f"Total matching documents: {len(matching_ids)}")
    print(f"Elapsed time:             {elapsed:.4f} s")
    print(f"Peak memory:              {peak_bytes / (1024 * 1024):.2f} MB")
    print("-" * 50)
    for doc_id in matching_ids[:10]:
        meta = index.doc_meta.get(doc_id)
        title = meta.title if meta else f"Doc {doc_id}"
        print(f"  [{doc_id:4d}] {title}")
    if len(matching_ids) > 10:
        print(f"  ... and {len(matching_ids) - 10} more.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
