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
        log.warning("Corpus root directory '%s' does not exist.", root)
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
