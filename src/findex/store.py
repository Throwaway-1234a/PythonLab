import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from findex.index import DocMeta, InvertedIndex, Posting

# ==============================================================================
# SECURITY NOTICE: UNTRUSTED PICKLE DESERIALIZATION
#
# Python's `pickle` format is an executable bytecode stack machine, NOT a passive
# data container. During unpickling, any object implementing `__reduce__` can
# instruct Python's runtime to call an arbitrary function with arbitrary arguments.
#
# An attacker can serialize a payload such as:
#
#     class Exploit:
#         def __reduce__(self):
#             import os
#             return (os.system, ('cat /etc/passwd',))
#
# Calling `pickle.load()` on untrusted input executes that code immediately with
# the privileges of the Python interpreter. Always use JSON or schema-validated
# binary protocols (like FlatBuffers/Protocol Buffers) for untrusted data.
# ==============================================================================


class _FindexUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__":
            module = "findex.index"
        return super().find_class(module, name)


def save(
    index: InvertedIndex, path: Path, fmt: Literal["pickle", "json"] = "pickle"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "pickle":
        with open(path, "wb") as f:
            pickle.dump(
                (dict(index.postings), index.doc_lengths, index.doc_meta), f, protocol=5
            )
    elif fmt == "json":
        data: dict[str, Any] = {
            "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
            "doc_meta": {
                str(k): {"doc_id": v.doc_id, "title": v.title, "path": v.path}
                for k, v in index.doc_meta.items()
            },
            "postings": {
                term: [
                    {"doc_id": p.doc_id, "tf": p.tf, "positions": list(p.positions)}
                    for p in posts
                ]
                for term, posts in index.postings.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def load(path: Path, fmt: Literal["pickle", "json"] = "pickle") -> InvertedIndex:
    if fmt == "pickle":
        with open(path, "rb") as f:
            unpickled = _FindexUnpickler(f).load()
            if len(unpickled) == 4:
                postings_dict, doc_lengths, doc_meta, raw_docs = unpickled
            else:
                postings_dict, doc_lengths, doc_meta = unpickled
                raw_docs = {}
            return InvertedIndex(
                postings=defaultdict(list, postings_dict),
                doc_lengths=doc_lengths,
                doc_meta=doc_meta,
                raw_docs=raw_docs,
            )
    elif fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            doc_lengths = {int(k): v for k, v in raw["doc_lengths"].items()}
            doc_meta = {
                int(k): DocMeta(doc_id=v["doc_id"], title=v["title"], path=v["path"])
                for k, v in raw["doc_meta"].items()
            }
            postings_dict = {
                term: [
                    Posting(
                        doc_id=p["doc_id"], tf=p["tf"], positions=tuple(p["positions"])
                    )
                    for p in posts
                ]
                for term, posts in raw["postings"].items()
            }
            return InvertedIndex(
                postings=defaultdict(list, postings_dict),
                doc_lengths=doc_lengths,
                doc_meta=doc_meta,
            )
    else:
        raise ValueError(f"Unsupported format: {fmt}")
