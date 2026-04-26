"""Tests for the §2.1 citation-floor guard."""

from __future__ import annotations

import pytest

from mission_brain.ingest.citation_floor import (
    CitationFloorViolation,
    check_citation_floor,
)


def test_valid_paragraph_with_ref_passes():
    md = "The user's 2016 journal names asymmetric objectives [ref:journal-2016-09:lines=45-52]."
    check_citation_floor(md)


def test_paragraph_without_ref_raises():
    md = "This paragraph has no anchor and must be rejected."
    with pytest.raises(CitationFloorViolation):
        check_citation_floor(md)


def test_malformed_ref_raises_on_uppercase_source():
    md = "Bad source id [ref:BadCaps:lines=1-2]."
    with pytest.raises(CitationFloorViolation):
        check_citation_floor(md)


def test_malformed_ref_raises_on_unknown_locator():
    md = "Unknown locator [ref:my-src:chapter=3]."
    with pytest.raises(CitationFloorViolation):
        check_citation_floor(md)


def test_malformed_ref_raises_even_when_paragraph_also_has_valid_ref():
    md = "Mixed markers [ref:my-src:lines=1-2] and [ref:BadCaps:lines=3-4]."
    with pytest.raises(CitationFloorViolation):
        check_citation_floor(md)


def test_fenced_code_block_exempt():
    md = (
        "```\n"
        "def hello():\n"
        "    return 'no ref needed inside code'\n"
        "```\n"
        "\n"
        "Prose outside fence still needs a ref [ref:my-src:lines=1-3]."
    )
    check_citation_floor(md)


def test_tilde_fenced_code_block_exempt():
    md = (
        "~~~\n"
        "raw text with no refs\n"
        "~~~\n"
        "\n"
        "Prose [ref:src:page=2]."
    )
    check_citation_floor(md)


def test_indented_code_block_exempt():
    md = (
        "Leading paragraph [ref:src:lines=1-1].\n"
        "\n"
        "    indented line one\n"
        "    indented line two\n"
    )
    check_citation_floor(md)


def test_atx_heading_exempt():
    md = (
        "# Heading Without Ref\n"
        "\n"
        "Body paragraph needs one [ref:src:lines=1-1]."
    )
    check_citation_floor(md)


def test_setext_heading_exempt():
    md = (
        "Section Title\n"
        "=============\n"
        "\n"
        "Body with ref [ref:src:lines=5-6]."
    )
    check_citation_floor(md)


def test_multiple_blank_lines_between_paragraphs_handled():
    md = (
        "Para one [ref:src:lines=1-1].\n"
        "\n"
        "\n"
        "\n"
        "Para two [ref:src:lines=2-2]."
    )
    check_citation_floor(md)


def test_trailing_newline_handled():
    md = "Single paragraph with ref [ref:src:lines=1-1].\n"
    check_citation_floor(md)


def test_empty_string_passes():
    check_citation_floor("")


def test_whitespace_only_passes():
    check_citation_floor("   \n\n\t\n")


def test_multi_paragraph_one_missing_raises():
    md = (
        "Good paragraph [ref:src:lines=1-1].\n"
        "\n"
        "Bad paragraph with no citation at all."
    )
    with pytest.raises(CitationFloorViolation):
        check_citation_floor(md)


def test_timestamp_locator_accepted():
    md = "Voice memo claim [ref:voice-2018-03:timestamp=00:01:30-00:02:45]."
    check_citation_floor(md)


def test_page_locator_accepted():
    md = "PDF page claim [ref:wargame-manual-x:page=42]."
    check_citation_floor(md)


def test_para_locator_accepted():
    md = "FB post claim [ref:fb-posts-2020:para=post_abc_123]."
    check_citation_floor(md)


def test_multi_line_paragraph_with_ref_passes():
    md = (
        "A paragraph that spans\n"
        "multiple lines still counts\n"
        "as one paragraph [ref:src:lines=1-3]."
    )
    check_citation_floor(md)


def test_ref_inside_fence_does_not_count_for_surrounding_text():
    md = (
        "Prose without any anchor.\n"
        "\n"
        "```\n"
        "[ref:src:lines=1-1]\n"
        "```\n"
    )
    with pytest.raises(CitationFloorViolation):
        check_citation_floor(md)


# Thematic-break exemption — fixes the keithtracton ingest failure
# (2026-04-24 morning rayb-020 run). The LLM emits `---` as a
# section separator in long synthesized wiki pages; without this
# exemption the citation floor rejects the whole page.

def test_thematic_break_dashes_exempt():
    md = (
        "Paragraph one [ref:src:lines=1-1].\n"
        "\n"
        "---\n"
        "\n"
        "Paragraph two [ref:src:lines=2-2]."
    )
    check_citation_floor(md)


def test_thematic_break_asterisks_exempt():
    md = (
        "Paragraph one [ref:src:lines=1-1].\n"
        "\n"
        "***\n"
        "\n"
        "Paragraph two [ref:src:lines=2-2]."
    )
    check_citation_floor(md)


def test_thematic_break_underscores_exempt():
    md = (
        "Paragraph one [ref:src:lines=1-1].\n"
        "\n"
        "___\n"
        "\n"
        "Paragraph two [ref:src:lines=2-2]."
    )
    check_citation_floor(md)


def test_thematic_break_with_spaces_exempt():
    # CommonMark allows spaces between the chars; the LLM doesn't
    # always emit a tight `---`.
    md = (
        "Paragraph one [ref:src:lines=1-1].\n"
        "\n"
        "- - -\n"
        "\n"
        "Paragraph two [ref:src:lines=2-2]."
    )
    check_citation_floor(md)


def test_thematic_break_long_form_exempt():
    md = (
        "Paragraph one [ref:src:lines=1-1].\n"
        "\n"
        "----------\n"
        "\n"
        "Paragraph two [ref:src:lines=2-2]."
    )
    check_citation_floor(md)


def test_thematic_break_with_indent_exempt():
    # Up to 3 leading spaces still counts as a thematic break.
    md = (
        "Paragraph one [ref:src:lines=1-1].\n"
        "\n"
        "   ---\n"
        "\n"
        "Paragraph two [ref:src:lines=2-2]."
    )
    check_citation_floor(md)


def test_setext_heading_with_dashes_still_recognized_as_heading():
    # Regression check: the thematic-break exemption must not
    # reclassify a setext h2 underline as a thematic break. The
    # `Section\n---` pair is a heading, not a separator.
    md = (
        "Section Title\n"
        "---\n"
        "\n"
        "Body with ref [ref:src:lines=5-6]."
    )
    check_citation_floor(md)


def test_paragraph_of_text_followed_by_dashes_outside_setext_still_required():
    # If the LLM produces a real prose paragraph WITHOUT a citation,
    # AND a separate thematic break, the prose paragraph must still
    # fail. Belt-check: thematic-break exemption is per-block, not
    # whole-document.
    md = (
        "Real prose paragraph that the LLM forgot to cite.\n"
        "\n"
        "---"
    )
    with pytest.raises(CitationFloorViolation):
        check_citation_floor(md)
