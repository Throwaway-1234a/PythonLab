from pathlib import Path

from findex.index import Index
from findex.ranking import BM25, TfIdf

INDEX_PATH = Path("index.bin")

GROUND_TRUTH = [
    ("whale ship captain", ["Moby Dick"]),
    ("alice rabbit hole", ["Alice"]),
    ("sense sister marriage", ["Sense And Sensibility", "Persuasion", "Emma"]),
    ("sea ocean storm", ["Moby Dick", "Ancient Mariner"]),
    ("serpent adam eve fruit", ["Paradise Lost", "Bible"]),
    ("king kingdom throne crown", ["Caesar", "Macbeth", "Hamlet"]),
    ("father family fortune daughters", ["Parents Assistant", "Sense And Sensibility"]),
    ("ghost castle apparition", ["Hamlet", "Macbeth"]),
    ("love heart beauty soul", ["Sonnets", "Songs Of Innocence"]),
    ("god lord prayer heaven", ["Bible", "Paradise Lost"]),
]


def precision_at_k(results: list, expected_substrings: list[str], k: int = 5) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = 0
    for r in top_k:
        if any(
            expected.casefold() in r.title.casefold()
            for expected in expected_substrings
        ):
            hits += 1
    return hits / k


def main() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(f"Index {INDEX_PATH} not found.")

    idx = Index.load(INDEX_PATH)
    tfidf = TfIdf()
    bm25 = BM25()

    print("\nEvaluating 10 Queries for Precision@5 (TF-IDF vs. BM25):\n")
    print("| # | Query | Expected Matches | TF-IDF P@5 | BM25 P@5 |")
    print("|---|:---|:---|:---:|:---:|")

    tfidf_scores = []
    bm25_scores = []

    for i, (query, expected) in enumerate(GROUND_TRUTH, 1):
        res_tfidf = idx.search(query, scorer=tfidf, k=5)
        res_bm25 = idx.search(query, scorer=bm25, k=5)

        p_tfidf = precision_at_k(res_tfidf, expected, k=5)
        p_bm25 = precision_at_k(res_bm25, expected, k=5)

        tfidf_scores.append(p_tfidf)
        bm25_scores.append(p_bm25)

        exp_str = ", ".join(expected)
        print(f"| {i} | `{query}` | {exp_str} | {p_tfidf:.2f} | {p_bm25:.2f} |")

    map_tfidf = sum(tfidf_scores) / len(tfidf_scores)
    map_bm25 = sum(bm25_scores) / len(bm25_scores)
    print(f"| **Mean** | **MAP@5** | | **{map_tfidf:.3f}** | **{map_bm25:.3f}** |")


if __name__ == "__main__":
    main()
