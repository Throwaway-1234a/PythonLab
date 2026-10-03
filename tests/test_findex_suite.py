from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import Index, Posting
from findex.query import And, Not, Or, Phrase, Term, parse
from findex.ranking import BM25
from findex.search import difference_postings, intersect_postings, union_postings
from findex.store import load, save
from findex.tokenize import tokenize

runner = CliRunner()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("PyThOn PrOgRaM", ["python", "program"]),
        ("Київ і Львів!", ["київ", "і", "львів"]),
        ("м'ясо та зв’язок", ["м'ясо", "та", "зв’язок"]),
        ("don't stop", ["don't", "stop"]),
        ("cafe\u0301", ["café"]),
        ("Wait... what?!", ["wait", "what"]),
        ("", []),
        ("state-of-the-art", ["state", "of", "the", "art"]),
        ("Apollo 11 in 1969", ["apollo", "in"]),
    ],
)
def test_tokenizer_parametrized(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_fixture_corpus_build(corpus_dir: Path) -> None:
    idx = Index.build(corpus_dir, record_positions=True)
    assert idx.num_docs == 3
    assert "whale" in idx
    assert "rabbit" in idx
    assert "ghost" in idx
    assert idx.df("whale") == 1
    assert idx.doc_length(0) > 0


def test_intersect_postings() -> None:
    p1 = [Posting(1, 1), Posting(3, 2), Posting(5, 1)]
    p2 = [Posting(2, 1), Posting(3, 4), Posting(6, 1)]
    assert [p.doc_id for p in intersect_postings(p1, p2)] == [3]


def test_union_postings() -> None:
    p1 = [Posting(1, 1), Posting(3, 2)]
    p2 = [Posting(2, 1), Posting(3, 4)]
    assert [p.doc_id for p in union_postings(p1, p2)] == [1, 2, 3]


def test_difference_postings() -> None:
    p1 = [Posting(1, 1), Posting(3, 2), Posting(5, 1)]
    p2 = [Posting(3, 4), Posting(6, 1)]
    assert [p.doc_id for p in difference_postings(p1, p2)] == [1, 5]


def test_parser_tree_equality() -> None:
    assert parse("a b") == And(Term("a"), Term("b"))
    assert parse("a OR b") == Or(Term("a"), Term("b"))
    assert parse("NOT a") == Not(Term("a"))
    assert parse("(a OR b) c") == And(Or(Term("a"), Term("b")), Term("c"))
    assert parse('"white whale"') == Phrase(("white", "whale"))


def test_sanity_check_rare_outranks_common(corpus_dir: Path) -> None:
    idx = Index.build(corpus_dir)
    bm25 = BM25()

    p_rare = idx["whale"][0]
    p_common = idx["the"][0]
    assert bm25.score("whale", p_rare, idx) > bm25.score("the", p_common, idx)


def test_sanity_check_term_saturation() -> None:
    bm25 = BM25()
    idx = Index()
    idx.doc_lengths[0] = 200
    idx.doc_meta[0] = None  # type: ignore[assignment]
    idx.postings["w"] = [Posting(0, 1)]

    s1 = bm25.score("w", Posting(0, 1), idx)
    s2 = bm25.score("w", Posting(0, 2), idx)
    s10 = bm25.score("w", Posting(0, 10), idx)
    s20 = bm25.score("w", Posting(0, 20), idx)
    assert (s2 - s1) > (s20 - s10)


def test_sanity_check_length_normalization() -> None:
    from findex.index import DocMeta

    bm25 = BM25()
    idx = Index()
    idx.doc_meta = {
        0: DocMeta(0, "Short", "short.txt"),
        1: DocMeta(1, "Long", "long.txt"),
    }
    idx.doc_lengths = {0: 50, 1: 5000}
    idx.postings["w"] = [Posting(0, 1), Posting(1, 1)]

    s_short = bm25.score("w", Posting(0, 1), idx)
    s_long = bm25.score("w", Posting(1, 1), idx)
    assert s_short > s_long


def test_store_roundtrip(tmp_path: Path, corpus_dir: Path) -> None:
    idx = Index.build(corpus_dir)
    pkl_file = tmp_path / "idx.bin"
    json_file = tmp_path / "idx.json"

    save(idx, pkl_file, fmt="pickle")
    loaded_pkl = load(pkl_file, fmt="pickle")
    assert len(loaded_pkl) == len(idx)

    save(idx, json_file, fmt="json")
    loaded_json = load(json_file, fmt="json")
    assert len(loaded_json) == len(idx)


def test_cli_index_and_stats(tmp_path: Path, corpus_dir: Path) -> None:
    out_file = tmp_path / "index.bin"
    res_index = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out_file)])
    assert res_index.exit_code == 0
    assert out_file.exists()

    res_stats = runner.invoke(app, ["stats", str(out_file)])
    assert res_stats.exit_code == 0
    assert "Total Documents" in res_stats.output


def test_cli_search_table(tmp_path: Path, corpus_dir: Path) -> None:
    out_file = tmp_path / "index.bin"
    runner.invoke(app, ["index", str(corpus_dir), "--out", str(out_file)])

    res = runner.invoke(app, ["search", str(out_file), "whale"])
    assert res.exit_code == 0
    assert "Moby Dick" in res.output


def test_cli_search_json_pipe(tmp_path: Path, corpus_dir: Path) -> None:
    out_file = tmp_path / "index.bin"
    runner.invoke(app, ["index", str(corpus_dir), "--out", str(out_file)])

    res = runner.invoke(app, ["search", str(out_file), "whale", "--json"])
    assert res.exit_code == 0
    import json

    data = json.loads(res.output.strip().splitlines()[0])
    assert data["title"] == "Moby Dick"
    assert data["score"] > 0


def test_cli_expected_errors_exit_nonzero(tmp_path: Path) -> None:
    res = runner.invoke(app, ["search", str(tmp_path / "nonexistent.bin"), "test"])
    assert res.exit_code != 0
