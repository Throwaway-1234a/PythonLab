from pathlib import Path

from findex.index import Index
from findex.semantic import SemanticSearcher, reciprocal_rank_fusion
from findex.vectorized import VectorizedIndex

INDEX_PATH = Path("index.bin")
NPY_PATH = Path("data/embeddings/embeddings.npy")
META_PATH = Path("data/embeddings/chunk_meta.json")

QUERIES = [
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
    ("marine creature giant monster", ["Moby Dick"]),
    ("subterranean passage wonderland", ["Alice"]),
    ("monarch royal scepter rule", ["Caesar", "Macbeth", "Hamlet"]),
    ("spectral figure battlements midnight", ["Hamlet"]),
    ("spiritual creator divine worship", ["Bible", "Paradise Lost"]),
]


def p_at_5(results: list, targets: list[str]) -> float:
    top = results[:5]
    if not top:
        return 0.0
    return (
        sum(1 for r in top if any(t.casefold() in r.title.casefold() for t in targets))
        / 5.0
    )


def main() -> None:
    if not INDEX_PATH.exists() or not NPY_PATH.exists():
        raise SystemExit("Index or embeddings missing. Run 'findex embed' first.")

    idx = Index.load(INDEX_PATH)
    vec_idx = VectorizedIndex.from_index(idx)
    searcher = SemanticSearcher(NPY_PATH, META_PATH, idx)

    print("\n| # | Query | Keyword P@5 | Semantic P@5 | Hybrid (RRF) P@5 |")
    print("|---|:---|:---:|:---:|:---:|")

    kw_scores, sem_scores, hyb_scores = [], [], []

    for i, (q, targets) in enumerate(QUERIES, start=1):
        kw_res = vec_idx.search_vectorized(q, scorer="bm25", k=10)
        sem_res = searcher.search_semantic(q, top_k=10)
        hyb_res = reciprocal_rank_fusion(kw_res, sem_res, k=60, top_k=5)

        p_kw = p_at_5(kw_res, targets)
        p_sem = p_at_5(sem_res, targets)
        p_hyb = p_at_5(hyb_res, targets)

        kw_scores.append(p_kw)
        sem_scores.append(p_sem)
        hyb_scores.append(p_hyb)

        print(f"| {i:2d} | `{q}` | {p_kw:.2f} | {p_sem:.2f} | {p_hyb:.2f} |")

    m_kw = sum(kw_scores) / len(kw_scores)
    m_sem = sum(sem_scores) / len(sem_scores)
    m_hyb = sum(hyb_scores) / len(hyb_scores)
    print(
        f"| **Mean** | **MAP@5** | **{m_kw:.3f}** | **{m_sem:.3f}** | **{m_hyb:.3f}** |"
    )


if __name__ == "__main__":
    main()
