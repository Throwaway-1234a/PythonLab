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


## Lab 2 — Data Structures and the Inverted Index

### 1. The Memory Study (M4)

Benchmarking 1,000,000 posting records across three internal storage representations measured under `tracemalloc`:

| Postings representation | Peak memory (build) | Index file size | Load time |
| :--- | :---: | :---: | :---: |
| `list[Posting]` with plain `@dataclass` | 114.86 MB | 19.91 MB | 0.549 s |
| `list[Posting]` with `slots=True` | 84.34 MB | 16.09 MB | 2.565 s |
| `array('I')` pairs (`doc_ids`, `tfs`) | **7.81 MB** | **7.63 MB** | **0.003 s** |

#### Where the Bytes Went
In the plain `@dataclass`, each instance carries an internal `__dict__` dictionary alongside standard `PyObject` headers, adding ~100+ bytes of hash table overhead per record. Specifying `__slots__ = ('doc_id', 'tf')` eliminates the per-instance dictionary and replaces dynamic attribute tables with fixed C-struct member pointers, cutting peak memory by 30.52 MB (a 26.6% reduction).

The `array('I')` representation bypasses the Python object model entirely. Rather than storing a list of 64-bit pointers pointing to individual heap-allocated objects, it stores contiguous primitive C integers (`uint32_t`) at exactly 4 bytes per integer ($2 \times 4\text{ bytes} \times 1,000,000 \approx 7.63\text{ MB}$). This yields a **14.7× reduction in memory** and enables near-instantaneous (0.003 s) deserialization via direct buffer loading.

---

### 2. Query Engine Benchmark: Two-Pointer Merge vs. Set (M2)

Benchmarking 1,000 query evaluations over the Gutenberg index on the two most common terms (`and`, `with`) and the two rarest terms (`ushering`, `blither`):

| Term Category | Query | Engine | Latency (1,000 runs) |
| :--- | :--- | :--- | :---: |
| **Two Most Common** | `and AND with` | `merge` (two-pointer) | 0.5678 s |
| **Two Most Common** | `and AND with` | `set` (hash set) | **0.0380 s** |
| **Two Rarest** | `ushering AND blither` | `merge` (two-pointer) | 0.7472 s |
| **Two Rarest** | `ushering AND blither` | `set` (hash set) | **0.0162 s** |

#### Engine Trade-offs
In CPython, the `set` implementation outperforms the two-pointer `merge` in wall-clock time because Python sets are executed in compiled C (`setobject.c`), while the two-pointer merge runs inside CPython's bytecode evaluation loop (`BINARY_SUBSCR`, `COMPARE_OP`, `LIST_APPEND`).

However, the two-pointer merge remains the foundational algorithm for production search engines because it operates in $O(L_1 + L_2)$ time over sequential disk streams without allocating large hash sets in RAM, and directly supports positional phrase intersections and compressed postings (e.g. Variable-byte or Elias coding).

---

### 3. Persistence Formats (M3)

Full corpus index persisted and loaded in two formats:

| Format | File Size | Save Time | Load Time |
| :--- | :---: | :---: | :---: |
| **pickle (Protocol 5)** | **9.96 MB** | **1.896 s** | **1.894 s** |
| **json** | 26.01 MB | 3.910 s | 2.659 s |

#### Security Warning on Pickle
Python's `pickle` module executes a bytecode stack program during deserialization. Any serialized object implementing `__reduce__()` can instruct the Python runtime to execute arbitrary functions (such as `os.system`) with the privileges of the running process. Pickle must **never** be used to load untrusted data from external sources; JSON or schema-enforced binary formats must be used instead.

---

### 4. Position Recording Cost (M1)

Measuring the trade-off of recording token positions (`tuple[int, ...]`) per posting to enable phrase queries:

| Configuration | Build Time | Peak Memory | File Size |
| :--- | :---: | :---: | :---: |
| **Without positions (`doc_id`, `tf`)** | 7.672 s | 45.04 MB | 9.96 MB |
| **With positions (`doc_id`, `tf`, `positions`)** | 14.178 s | 135.40 MB | 16.50 MB |

Tracking token offsets roughly doubles index construction time (1.85×) and triples peak memory usage (3.0×) due to the allocation of individual integer offsets and position tuples.

## Lab 3 — The Object Model and Ranking

### 1. Three Ranking Sanity Checks (BM25)

Demonstrated on the Gutenberg corpus using `scripts/sanity_checks.py`:

#### Check 1: Rare Term Outranks Common Term
* Rare term: `'tenderer'` ($df=2$) $\to$ Score: **5.7707**
* Common term: `'all'` ($df=863$) $\to$ Score: **0.0013**
* **Result:** Rare term outranks the common term by **4,309.1×**, verifying IDF scaling.

#### Check 2: BM25 Term Frequency Saturation (Diminishing Returns)
* $tf = 1 \to 0.2877$
* $tf = 2 \to 0.4110$ (Gain: $+0.1233$)
* $tf = 5 \to 0.5532$
* $tf = 10 \to 0.6254$
* $tf = 20 \to 0.6690$ (Gain from 10 to 20: $+0.0436$)
* **Result:** Moving from $tf=1$ to $2$ yields nearly $3\times$ the score gain of moving from $tf=10$ to $20$, verifying that $k_1=1.5$ dampens term repetition.

#### Check 3: Document Length Normalization
* Short document ($dl = 100$): Score = **0.3212**
* Long document ($dl = 5,000$): Score = **0.1273**
* **Result:** A single match in a short document outranks the identical match in a bloated document by **2.5×**, verifying length penalty tuning with $b=0.75$.

---

### 2. Timing Logs & LRU Cache Hit Evidence

Decorating search with `@timed` and caching query evaluations via `@functools.lru_cache(maxsize=256)`:

```text
Index.load executed in 1932.844 ms

--- First run for: 'whale AND captain' (Cold cache) ---
Index.search executed in 5.586 ms

--- Second run for: 'whale AND captain' (Hot cache hit) ---
Index.search executed in 0.488 ms

Cache Stats: CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
```

**Result:** The hot cache hit bypasses AST parsing and postings list traversal, reducing search latency from **5.586 ms** to **0.488 ms** (**11.4× faster**).

---

### 3. Precision@5 Evaluation (TF-IDF vs. BM25)

Measured across 10 hand-labeled queries on the Gutenberg corpus:

| # | Query | Expected Matches | TF-IDF P@5 | BM25 P@5 |
|---|:---|:---|:---:|:---:|
| 1 | `whale ship captain` | Moby Dick | 1.00 | 1.00 |
| 2 | `alice rabbit hole` | Alice | 0.40 | 0.40 |
| 3 | `sense sister marriage` | Sense And Sensibility, Persuasion, Emma | 0.00 | 0.00 |
| 4 | `sea ocean storm` | Moby Dick, Ancient Mariner | 0.80 | 0.80 |
| 5 | `serpent adam eve fruit` | Paradise Lost, Bible | 0.20 | 0.20 |
| 6 | `king kingdom throne crown` | Caesar, Macbeth, Hamlet | 0.00 | 0.00 |
| 7 | `father family fortune daughters` | Parents Assistant, Sense And Sensibility | 0.00 | 0.00 |
| 8 | `ghost castle apparition` | Hamlet, Macbeth | 0.00 | 0.00 |
| 9 | `love heart beauty soul` | Sonnets, Songs Of Innocence | 0.00 | 0.00 |
| 10 | `god lord prayer heaven` | Bible, Paradise Lost | 1.00 | 1.00 |
| **Mean** | **MAP@5** | | **0.340** | **0.340** |