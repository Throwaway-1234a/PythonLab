import inspect

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def test_mixed_case() -> None:
    assert list(tokenize("PyThOn PrOgRaMmInG")) == ["python", "programming"]


def test_cyrillic_text() -> None:
    raw = "Київ — столиця України. Їжак та ґанок."
    expected = ["київ", "столиця", "україни", "їжак", "та", "ґанок"]
    assert list(tokenize(raw)) == expected


def test_ukrainian_apostrophe() -> None:
    raw = "м'ясо, зв’язок, п'ять"
    assert list(tokenize(raw)) == ["м'ясо", "зв’язок", "п'ять"]


def test_english_contractions() -> None:
    raw = "Don't stop, it's working!"
    assert list(tokenize(raw)) == ["don't", "stop", "it's", "working"]


def test_combining_mark_accent() -> None:
    decomposed = "cafe\u0301"
    assert list(tokenize(decomposed)) == ["café"]


def test_punctuation_stripping() -> None:
    raw = "Wait... What?! (Nothing) [Zero]: True; 'done'."
    assert list(tokenize(raw)) == ["wait", "what", "nothing", "zero", "true", "done"]


def test_empty_string_and_whitespace() -> None:
    assert list(tokenize("")) == []
    assert list(tokenize("   \n\t  \r  ")) == []


def test_hyphen_policy_splits() -> None:
    raw = "state-of-the-art full-text"
    assert list(tokenize(raw)) == ["state", "of", "the", "art", "full", "text"]


def test_digits_policy_ignored() -> None:
    raw = "Apollo 11 launched in 1969 at 09:32."
    assert list(tokenize(raw)) == ["apollo", "launched", "in", "at"]


def test_is_generator_function() -> None:
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)
