"""Tests for the §1-2 pydantic schema models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mission_brain.ingest.citation_floor import CitationFloorViolation
from mission_brain.schema import (
    Citation,
    CoVisibilityViolation,
    LineRangeLocator,
    PageLocator,
    ParaAnchorLocator,
    Paragraph,
    QueryResult,
    SourceId,
    SourceIdViolation,
    TimestampLocator,
    WikiPage,
)

# ---------------------------------------------------------------------------
# SourceId — grammar [a-z0-9_-]+


@pytest.mark.parametrize(
    "value",
    ["plain-md-journal-2016-09", "abc", "a_b-c-1", "0", "x-y-z_42"],
)
def test_source_id_valid(value: str) -> None:
    sid = SourceId(value)
    assert sid == value
    assert isinstance(sid, str)


@pytest.mark.parametrize(
    "value",
    ["", "BadCaps", "has space", "has/slash", "has.dot", "exclaim!"],
)
def test_source_id_invalid_raises(value: str) -> None:
    with pytest.raises(SourceIdViolation):
        SourceId(value)


# ---------------------------------------------------------------------------
# SourceId.coerce — lossy normalization for retrieval-boundary use


@pytest.mark.parametrize(
    "value",
    ["plain-md-journal-2016-09", "abc", "a_b-c-1", "0", "x-y-z_42"],
)
def test_source_id_coerce_passes_valid_through(value: str) -> None:
    sid = SourceId.coerce(value)
    assert sid == value
    assert isinstance(sid, SourceId)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain-md-legacy note title", "plain-md-legacy-note-title"),
        ("HasCaps", "hascaps"),
        ("has spaces", "has-spaces"),
        ("has/slash", "has-slash"),
        ("has.dot", "has-dot"),
        ("multi   spaces", "multi-spaces"),
        ("---leading-trailing---", "leading-trailing"),
        ("exclaim!", "exclaim"),
    ],
)
def test_source_id_coerce_slugifies(raw: str, expected: str) -> None:
    sid = SourceId.coerce(raw)
    assert sid == expected
    assert SourceId(sid) == sid


@pytest.mark.parametrize(
    "value",
    ["", "!!!", "----", "   "],
)
def test_source_id_coerce_empty_raises(value: str) -> None:
    with pytest.raises(SourceIdViolation):
        SourceId.coerce(value)


# ---------------------------------------------------------------------------
# Locator variants


def test_line_range_locator_format() -> None:
    loc = LineRangeLocator(start=45, end=52)
    assert loc.format() == "lines=45-52"


def test_line_range_locator_end_before_start_raises() -> None:
    with pytest.raises(ValidationError):
        LineRangeLocator(start=10, end=5)


def test_page_locator_format() -> None:
    assert PageLocator(page=7).format() == "page=7"


def test_page_locator_zero_raises() -> None:
    with pytest.raises(ValidationError):
        PageLocator(page=0)


def test_timestamp_locator_format() -> None:
    loc = TimestampLocator(start="00:01:30", end="00:02:45")
    assert loc.format() == "timestamp=00:01:30-00:02:45"


def test_timestamp_locator_bad_format_raises() -> None:
    with pytest.raises(ValidationError):
        TimestampLocator(start="1:30", end="2:45")


def test_para_anchor_locator_format() -> None:
    assert ParaAnchorLocator(anchor="msg-id-1234").format() == "para=msg-id-1234"


def test_para_anchor_locator_whitespace_raises() -> None:
    with pytest.raises(ValidationError):
        ParaAnchorLocator(anchor="has space")


# ---------------------------------------------------------------------------
# Citation roundtrip via render()


def test_citation_render_lines() -> None:
    c = Citation(
        source_id=SourceId("plain-md-journal-2016-09"),
        locator=LineRangeLocator(start=45, end=52),
        excerpt="a third way between victory conditions and scoring",
    )
    assert c.render() == "[ref:plain-md-journal-2016-09:lines=45-52]"


def test_citation_render_para() -> None:
    c = Citation(
        source_id=SourceId("fb-post-archive"),
        locator=ParaAnchorLocator(anchor="post-987654321"),
        excerpt="excerpt body",
    )
    assert c.render() == "[ref:fb-post-archive:para=post-987654321]"


def test_citation_discriminated_union_dispatches_by_kind() -> None:
    payload = {
        "source_id": "abc",
        "locator": {"kind": "page", "page": 3},
        "excerpt": "x",
    }
    c = Citation.model_validate(payload)
    assert isinstance(c.locator, PageLocator)
    assert c.locator.page == 3


# ---------------------------------------------------------------------------
# Paragraph citation-floor validator (§2.1)


def _cite() -> Citation:
    return Citation(
        source_id=SourceId("src-x"),
        locator=LineRangeLocator(start=1, end=3),
        excerpt="ex",
    )


def test_paragraph_with_text_and_citation_ok() -> None:
    p = Paragraph(text="A claim.", citations=[_cite()])
    assert p.text == "A claim."


def test_paragraph_with_text_no_citations_raises_citation_floor() -> None:
    with pytest.raises(CitationFloorViolation):
        Paragraph(text="A claim with no anchor.", citations=[])


def test_paragraph_blank_text_no_citations_ok() -> None:
    Paragraph(text="", citations=[])
    Paragraph(text="   \n  ", citations=[])


# ---------------------------------------------------------------------------
# WikiPage composition


def test_wiki_page_composes() -> None:
    page = WikiPage(
        title="Asymmetric Objectives",
        source_ids=[SourceId("plain-md-journal-2016-09")],
        paragraphs=[Paragraph(text="A claim.", citations=[_cite()])],
    )
    assert page.title == "Asymmetric Objectives"
    assert len(page.paragraphs) == 1


# ---------------------------------------------------------------------------
# QueryResult co-visibility (§2.4)


def test_query_result_none_page_empty_citations_ok() -> None:
    qr = QueryResult(synthesized_page=None, raw_citations=[])
    assert qr.synthesized_page is None
    assert qr.raw_citations == []


def test_query_result_none_page_with_citations_ok() -> None:
    qr = QueryResult(synthesized_page=None, raw_citations=[_cite()])
    assert qr.synthesized_page is None
    assert len(qr.raw_citations) == 1


def test_query_result_page_with_citations_ok() -> None:
    page = WikiPage(
        title="t",
        source_ids=[SourceId("src-x")],
        paragraphs=[Paragraph(text="A claim.", citations=[_cite()])],
    )
    qr = QueryResult(synthesized_page=page, raw_citations=[_cite()])
    assert qr.synthesized_page is page


def test_query_result_page_without_citations_raises_co_visibility() -> None:
    page = WikiPage(
        title="t",
        source_ids=[SourceId("src-x")],
        paragraphs=[Paragraph(text="A claim.", citations=[_cite()])],
    )
    with pytest.raises(CoVisibilityViolation):
        QueryResult(synthesized_page=page, raw_citations=[])
