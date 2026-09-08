import re
import unicodedata
from collections.abc import Iterator

WORD_RE = re.compile(r"\b[^\W\d_]+(?:['’][^\W\d_]+)*\b")


def tokenize(text: str) -> Iterator[str]:
    normalized = unicodedata.normalize("NFC", text)

    folded = normalized.casefold()

    for match in WORD_RE.finditer(folded):
        yield match.group(0)
