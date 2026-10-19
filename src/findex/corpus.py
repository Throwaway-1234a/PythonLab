import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("findex.corpus")


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    if not root.exists():
        log.warning("Corpus path '%s' does not exist.", root)
        return

    if root.is_file() and root.suffix == ".jsonl":
        with open(root, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    doc_id = str(
                        data.get("doc_id") or data.get("url") or f"doc_{line_no}"
                    )
                    text = str(data.get("text", ""))
                    yield Document(doc_id=doc_id, path=root, text=text)
                except json.JSONDecodeError as err:
                    log.warning("Skipping invalid JSON line %d: %s", line_no, err)
        return

    for path in sorted(root.rglob("*.txt")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            yield Document(doc_id=path.stem, path=path, text=text)
        except OSError as err:
            log.warning("Skipping unreadable file %s: %s", path, err)
            continue
