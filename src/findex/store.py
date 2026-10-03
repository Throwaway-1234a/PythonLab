import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

from findex.index import DocMeta, InvertedIndex, Posting

Serialized4 = tuple[
    dict[str, list[Posting]],
    dict[int, int],
    dict[int, DocMeta],
    dict[int, str],
]
Serialized3 = tuple[
    dict[str, list[Posting]],
    dict[int, int],
    dict[int, DocMeta],
]


class FindexUnpickler(pickle.Unpickler):
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
                (
                    dict(index.postings),
                    index.doc_lengths,
                    index.doc_meta,
                    index.raw_docs,
                ),
                f,
                protocol=5,
            )
    elif fmt == "json":
        data: dict[str, Any] = {
            "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
            "doc_meta": {
                str(k): {"doc_id": v.doc_id, "title": v.title, "path": v.path}
                for k, v in index.doc_meta.items()
            },
            "raw_docs": {str(k): v for k, v in index.raw_docs.items()},
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
            raw_obj: object = FindexUnpickler(f).load()

        if not isinstance(raw_obj, tuple):
            raise ValueError(f"Corrupted index in {path}: expected tuple")

        raw_tuple = cast(tuple[object, ...], raw_obj)
        postings_dict: dict[str, list[Posting]]
        doc_lengths: dict[int, int]
        doc_meta: dict[int, DocMeta]
        raw_docs: dict[int, str]

        if len(raw_tuple) == 4:
            t4 = cast(Serialized4, raw_tuple)
            postings_dict, doc_lengths, doc_meta, raw_docs = t4
        elif len(raw_tuple) == 3:
            t3 = cast(Serialized3, raw_tuple)
            postings_dict, doc_lengths, doc_meta = t3
            raw_docs = {}
        else:
            raise ValueError(
                f"Corrupted index in {path}: invalid tuple size {len(raw_tuple)}"
            )

        return InvertedIndex(
            postings=defaultdict(list, postings_dict),
            doc_lengths=doc_lengths,
            doc_meta=doc_meta,
            raw_docs=raw_docs,
        )
    elif fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            raw_json: object = json.load(f)

        if not isinstance(raw_json, dict):
            raise ValueError(f"Expected dict in {path}")

        raw = cast(dict[str, Any], raw_json)

        raw_doc_lengths = cast(dict[str, int], raw.get("doc_lengths", {}))
        doc_lengths_json: dict[int, int] = {
            int(k): v for k, v in raw_doc_lengths.items()
        }

        raw_doc_meta = cast(dict[str, dict[str, Any]], raw.get("doc_meta", {}))
        doc_meta_json: dict[int, DocMeta] = {
            int(k): DocMeta(
                doc_id=int(v["doc_id"]), title=str(v["title"]), path=str(v["path"])
            )
            for k, v in raw_doc_meta.items()
        }

        raw_raw_docs = cast(dict[str, str], raw.get("raw_docs", {}))
        raw_docs_json: dict[int, str] = {int(k): v for k, v in raw_raw_docs.items()}

        raw_postings = cast(dict[str, list[dict[str, Any]]], raw.get("postings", {}))
        postings_dict_json: dict[str, list[Posting]] = {
            term: [
                Posting(
                    doc_id=int(p["doc_id"]),
                    tf=int(p["tf"]),
                    positions=tuple(cast(list[int], p.get("positions", []))),
                )
                for p in posts
            ]
            for term, posts in raw_postings.items()
        }

        return InvertedIndex(
            postings=defaultdict(list, postings_dict_json),
            doc_lengths=doc_lengths_json,
            doc_meta=doc_meta_json,
            raw_docs=raw_docs_json,
        )
    else:
        raise ValueError(f"Unsupported format: {fmt}")
