from findex.query import And, Not, Or, Phrase, Term, parse


def test_single_term() -> None:
    assert parse("python") == Term("python")


def test_implicit_and() -> None:
    assert parse("captain whale") == And(Term("captain"), Term("whale"))


def test_or_expression() -> None:
    assert parse("whale OR sea") == Or(Term("whale"), Term("sea"))


def test_prefix_not() -> None:
    assert parse("NOT captain") == Not(Term("captain"))
    assert parse("whale AND NOT captain") == And(Term("whale"), Not(Term("captain")))


def test_parentheses_precedence() -> None:
    assert parse("(a OR b) c") == And(Or(Term("a"), Term("b")), Term("c"))
    assert parse("a OR (b c)") == Or(Term("a"), And(Term("b"), Term("c")))


def test_quoted_phrase() -> None:
    assert parse('"white whale"') == Phrase(("white", "whale"))
    assert parse('"white whale" captain') == And(
        Phrase(("white", "whale")), Term("captain")
    )


def test_operator_overloading() -> None:
    t1 = Term("whale")
    t2 = Term("sea")
    assert (t1 & t2) == And(t1, t2)
    assert (t1 | t2) == Or(t1, t2)
    assert (~t1) == Not(t1)
