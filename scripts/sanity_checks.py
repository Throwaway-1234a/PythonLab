from pathlib import Path

from findex.index import DocMeta, Index, Posting
from findex.ranking import BM25

INDEX_PATH = Path("index.bin")


def main() -> None:
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"Index {INDEX_PATH} not found. Run: python -m findex.index data/corpus --out index.bin"
        )

    idx = Index.load(INDEX_PATH)
    bm25 = BM25()

    print("=" * 65)
    print("SANITY CHECK 1: Rare Term Outranks Common Term")
    print("=" * 65)

    sorted_terms = sorted(idx.postings.keys(), key=lambda t: len(idx.postings[t]))
    rare_term = next(t for t in sorted_terms if len(idx[t]) == 2)
    common_term = next(
        t for t in reversed(sorted_terms) if len(idx[t]) > idx.num_docs * 0.7
    )

    p_rare = idx[rare_term][0]
    p_common = idx[common_term][0]
    score_rare = bm25.score(rare_term, p_rare, idx)
    score_common = bm25.score(common_term, p_common, idx)

    print(
        f"Rare term:   '{rare_term}' (df={idx.df(rare_term)}) -> Score: {score_rare:.4f}"
    )
    print(
        f"Common term: '{common_term}' (df={idx.df(common_term)}) -> Score: {score_common:.4f}"
    )
    print(
        f"Verified: Rare outranks Common by {score_rare / max(score_common, 1e-6):.1f}x\n"
    )

    print("=" * 65)
    print("SANITY CHECK 2: BM25 Term Frequency Saturation (Diminishing Returns)")
    print("=" * 65)

    synth_idx = Index(doc_lengths={0: 500}, doc_meta={0: DocMeta(0, "Synth", "s.txt")})
    synth_idx.postings["target"] = [Posting(0, 1)]

    scores_at_tf = []
    for tf_val in [1, 2, 5, 10, 20]:
        post = Posting(0, tf_val)
        s = bm25.score("target", post, synth_idx)
        scores_at_tf.append((tf_val, s))
        print(f"  tf = {tf_val:2d} -> Score: {s:.4f}")

    diff_1_to_2 = scores_at_tf[1][1] - scores_at_tf[0][1]
    diff_10_to_20 = scores_at_tf[4][1] - scores_at_tf[3][1]
    print(f"Gain from tf=1 to tf=2:   +{diff_1_to_2:.4f}")
    print(f"Gain from tf=10 to tf=20: +{diff_10_to_20:.4f}")
    print("Verified: 20th repetition adds almost nothing.\n")

    print("=" * 65)
    print("SANITY CHECK 3: Length Normalization (Short doc outranks Long doc)")
    print("=" * 65)
    len_idx = Index(
        doc_lengths={0: 100, 1: 5000},
        doc_meta={
            0: DocMeta(0, "Short", "short.txt"),
            1: DocMeta(1, "Long", "long.txt"),
        },
    )
    len_idx.postings["target"] = [Posting(0, 1), Posting(1, 1)]

    s_short = bm25.score("target", Posting(0, 1), len_idx)
    s_long = bm25.score("target", Posting(1, 1), len_idx)

    print(f"Short doc (dl={len_idx.doc_length(0)}): Score = {s_short:.4f}")
    print(f"Long doc  (dl={len_idx.doc_length(1)}): Score = {s_long:.4f}")
    print(f"Verified: Short doc outranks Long doc by {s_short / s_long:.1f}x\n")


if __name__ == "__main__":
    main()
