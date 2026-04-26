"""WikiPage + Paragraph — pydantic encoding of CLAUDE.md §2.1.

A :class:`Paragraph` represents a prose block in a wiki page. The
§2.1 citation-floor invariant: any non-empty prose paragraph must
carry at least one citation. Headings and code blocks are not
modeled as Paragraphs — they are exempt by construction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from mission_brain.ingest.citation_floor import CitationFloorViolation
from mission_brain.schema.citation import Citation
from mission_brain.schema.source_id import SourceId

__all__ = ["Paragraph", "Provenance", "WikiPage"]

# "journal-private" added 2026-04-26 (rb-035) for the bullet-journal
# corpus. Downstream consumers (CLI --exclude-provenance, MCP server
# default exclusions) filter this value out by default — ingestion
# is enabled, but bot-driven drafts don't see the user's private journal
# unless explicitly opted in.
Provenance = Literal["primary", "derived", "journal-private"]


class Paragraph(BaseModel):
    """A prose paragraph in a wiki page.

    Per CLAUDE.md §2.1, any non-empty paragraph must carry at least
    one citation. Construction with ``text.strip() != ""`` and an
    empty ``citations`` list raises :class:`CitationFloorViolation`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    citations: list[Citation]

    @model_validator(mode="after")
    def _enforce_citation_floor(self) -> Paragraph:
        if self.text.strip() != "" and len(self.citations) == 0:
            preview = self.text if len(self.text) <= 120 else self.text[:117] + "..."
            raise CitationFloorViolation(
                f"Paragraph has non-empty text but zero citations: {preview!r}"
            )
        return self


class WikiPage(BaseModel):
    """A synthesized wiki page composed of cited paragraphs.

    ``provenance`` distinguishes primary sources (user-authored or
    mechanical extractions from the user's own material) from derived
    sources (content synthesized by another tool before reaching
    mission-brain). Defaults to ``"primary"`` so existing call sites
    remain unchanged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    source_ids: list[SourceId]
    paragraphs: list[Paragraph]
    provenance: Provenance = "primary"
