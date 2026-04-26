"""Pydantic schema models — mechanical encoding of CLAUDE.md §1-2.

Per CLAUDE.md, every wiki page carries citation anchors and the
query API enforces co-visibility between synthesized pages and
their raw citations. These models translate that prose spec into
runtime-validated Python types.
"""

from __future__ import annotations

from mission_brain.schema.citation import (
    BarRangeLocator,
    Citation,
    LineRangeLocator,
    Locator,
    PageLocator,
    ParaAnchorLocator,
    TimeRangeLocator,
    TimestampLocator,
)
from mission_brain.schema.query_result import CoVisibilityViolation, QueryResult
from mission_brain.schema.source_id import SourceId, SourceIdViolation
from mission_brain.schema.wiki_page import Paragraph, Provenance, WikiPage

__all__ = [
    "BarRangeLocator",
    "Citation",
    "CoVisibilityViolation",
    "LineRangeLocator",
    "Locator",
    "PageLocator",
    "ParaAnchorLocator",
    "Paragraph",
    "Provenance",
    "QueryResult",
    "SourceId",
    "SourceIdViolation",
    "TimeRangeLocator",
    "TimestampLocator",
    "WikiPage",
]
