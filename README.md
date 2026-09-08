# findex

A full-text search engine built from scratch in modern Python.

## Lab 1 — Streams, Not Lists

### Corpus Description

The dataset is derived from Project Gutenberg public-domain literature, partitioned into chapter-length documents totaling ~1,000 files (~13 MB uncompressed text).

To fetch and prepare the corpus:
```bash
uv run python scripts/download_corpus.py
```
Files are extracted to `data/corpus/` (which is excluded from Git via `.gitignore`).

### Tokenizer Policy

The tokenizer in `findex.tokenize` applies Unicode NFC normalization followed by `str.casefold()` and streams tokens via `re.finditer`:
* **Apostrophes:** Retained within words (e.g., `don't`, `it's`, and Ukrainian `м'ясо`, `зв’язок`) to preserve contractions and language-specific orthography. Leading and trailing quotes are stripped.
* **Hyphens:** Treated as word delimiters (e.g., `state-of-the-art` produces `state`, `of`, `the`, `art`), enabling retrieval of constituent query terms.
* **Digits:** Excluded from tokens to focus index capacity on natural language vocabulary.

### Eager vs. Lazy Benchmark

Evaluated over 500 documents using Python 3.12 and `tracemalloc`:

| Version           | Documents | Peak memory | Elapsed  |
|:------------------|:---------:|:-----------:|:--------:|
| eager (lists)     |       500 |    67.70 MB |  2.857 s |
| lazy (generators) |       500 |     2.03 MB |  3.487 s |

### Where the Memory Went

The eager pipeline materializes the entire corpus in memory as a list of strings, tokenizes all documents simultaneously into a nested `list[list[str]]`, and flattens them before counting. In CPython, each individual string object has a ~50-byte header overhead in addition to the 8-byte pointer inside the list array, causing memory consumption to grow linearly O(N) with corpus size and peaking near 67 (✌️) MB.

In contrast, the lazy generator pipeline streams documents one by one via `iter_documents` and yields tokens on-the-fly using `yield` and `re.finditer`. Only a single active document buffer and one token string exist on the heap at any given moment, keeping auxiliary heap usage bounded at roughly 2.0 MB (O(1) with respect to total corpus size, bounded only by the term `Counter`).
