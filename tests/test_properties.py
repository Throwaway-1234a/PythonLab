from hypothesis import given
from hypothesis import strategies as st

from findex.index import Posting
from findex.search import intersect_postings, union_postings

postings_strategy = st.lists(
    st.integers(min_value=0, max_value=10_000),
    unique=True,
).map(lambda ids: [Posting(doc_id=i, tf=1) for i in sorted(ids)])


@given(postings_strategy, postings_strategy)
def test_property_intersection_commutativity(
    p1: list[Posting], p2: list[Posting]
) -> None:
    r1 = [p.doc_id for p in intersect_postings(p1, p2)]
    r2 = [p.doc_id for p in intersect_postings(p2, p1)]
    assert r1 == r2


@given(postings_strategy, postings_strategy)
def test_property_intersection_matches_set_model(
    p1: list[Posting], p2: list[Posting]
) -> None:
    set_intersection = sorted({p.doc_id for p in p1} & {p.doc_id for p in p2})
    engine_result = [p.doc_id for p in intersect_postings(p1, p2)]
    assert engine_result == set_intersection


@given(postings_strategy, postings_strategy)
def test_property_postings_remain_monotonically_sorted(
    p1: list[Posting], p2: list[Posting]
) -> None:
    union_res = [p.doc_id for p in union_postings(p1, p2)]
    for i in range(len(union_res) - 1):
        assert union_res[i] < union_res[i + 1]
