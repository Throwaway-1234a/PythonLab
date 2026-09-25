import re


def make_snippet(text: str, query_terms: set[str], window_chars: int = 80) -> str:
    if not text or not query_terms:
        preview = text[: window_chars * 2].strip()
        return preview + ("..." if len(text) > len(preview) else "")

    lower_text = text.casefold()
    earliest_idx: int | None = None

    for term in query_terms:
        if not term:
            continue
        idx = lower_text.find(term.casefold())
        if idx != -1 and (earliest_idx is None or idx < earliest_idx):
            earliest_idx = idx

    if earliest_idx is None:
        preview = text[: window_chars * 2].strip()
        return preview + "..."

    start = max(0, earliest_idx - window_chars)
    end = min(len(text), earliest_idx + window_chars)

    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(text) else ""
    snippet = prefix + text[start:end].strip() + suffix

    for term in query_terms:
        if term:
            pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
            snippet = pattern.sub(r"**\1**", snippet)

    return snippet
