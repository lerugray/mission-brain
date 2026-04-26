"""QueryResult — pydantic encoding of CLAUDE.md §2.4.

Co-visibility invariant: a query that surfaces a synthesized wiki
page MUST also surface the raw citations supporting it. A
``QueryResult`` with ``synthesized_page is not None`` and
``raw_citations == []`` is rejected at construction time.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from mission_brain.schema.citation import Citation
from mission_brain.schema.wiki_page import WikiPage

__all__ = ["CoVisibilityViolation", "QueryResult"]


class CoVisibilityViolation(Exception):
    """A QueryResult contains a synthesized page but no raw citations.

    Plain :class:`Exception` (not :class:`ValueError`) so pydantic's
    ``model_validator`` lets it propagate verbatim instead of wrapping
    it in :class:`pydantic.ValidationError`.
    """


class QueryResult(BaseModel):
    """The only return type from the public query entry point.

    Either:
      * ``synthesized_page`` is None — pure raw-passage hits (or no
        hits, in which case ``raw_citations`` may also be empty), OR
      * ``synthesized_page`` is a :class:`WikiPage` AND
        ``raw_citations`` is non-empty.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    synthesized_page: WikiPage | None
    raw_citations: list[Citation]

    @model_validator(mode="after")
    def _enforce_co_visibility(self) -> QueryResult:
        if self.synthesized_page is not None and len(self.raw_citations) == 0:
            raise CoVisibilityViolation(
                "QueryResult has a synthesized_page but raw_citations is empty; "
                "co-visibility (§2.4) requires citations alongside the page"
            )
        return self
