"""Citation models — Locator discriminated union + Citation.

CLAUDE.md §1 locator grammar:
  - ``lines=N-M`` → :class:`LineRangeLocator`
  - ``page=N`` → :class:`PageLocator`
  - ``timestamp=HH:MM:SS-HH:MM:SS`` → :class:`TimestampLocator`
  - ``time=HH:MM:SS.fff-HH:MM:SS.fff`` → :class:`TimeRangeLocator`
  - ``bar=N-M`` → :class:`BarRangeLocator`
  - ``para=<anchor>`` → :class:`ParaAnchorLocator`
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mission_brain.schema.source_id import SourceId

__all__ = [
    "BarRangeLocator",
    "Citation",
    "LineRangeLocator",
    "Locator",
    "PageLocator",
    "ParaAnchorLocator",
    "TimeRangeLocator",
    "TimestampLocator",
]


_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}$")
_PARA_ANCHOR_RE = re.compile(r"^[^\]\s]+$")


class _LocatorBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LineRangeLocator(_LocatorBase):
    kind: Literal["lines"] = "lines"
    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @field_validator("end")
    @classmethod
    def _end_ge_start(cls, v: int, info) -> int:
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"line range end ({v}) must be >= start ({start})")
        return v

    def format(self) -> str:
        return f"lines={self.start}-{self.end}"


class PageLocator(_LocatorBase):
    kind: Literal["page"] = "page"
    page: int = Field(ge=1)

    def format(self) -> str:
        return f"page={self.page}"


class TimestampLocator(_LocatorBase):
    kind: Literal["timestamp"] = "timestamp"
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _hms(cls, v: str) -> str:
        if not _TIMESTAMP_RE.fullmatch(v):
            raise ValueError(f"timestamp {v!r} must match HH:MM:SS")
        return v

    def format(self) -> str:
        return f"timestamp={self.start}-{self.end}"


class TimeRangeLocator(_LocatorBase):
    """Sub-second time range for music sources — ``HH:MM:SS.fff`` pair.

    Distinct from :class:`TimestampLocator` (whole-second, for audio
    transcripts). Introduced 2026-04-19 for MIDI-JSON music sources
    where bar-level grids aren't always inferrable but note start/end
    times are.
    """

    kind: Literal["time"] = "time"
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _hms_fff(cls, v: str) -> str:
        if not _TIME_RE.fullmatch(v):
            raise ValueError(f"time {v!r} must match HH:MM:SS.fff")
        return v

    def format(self) -> str:
        return f"time={self.start}-{self.end}"


class BarRangeLocator(_LocatorBase):
    """Bar range for music sources. Optional companion to :class:`TimeRangeLocator`
    when a bar-level grid is inferrable (music21 at ingest-time, Phase 3+).
    """

    kind: Literal["bar"] = "bar"
    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @field_validator("end")
    @classmethod
    def _end_ge_start(cls, v: int, info) -> int:
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"bar range end ({v}) must be >= start ({start})")
        return v

    def format(self) -> str:
        return f"bar={self.start}-{self.end}"


class ParaAnchorLocator(_LocatorBase):
    kind: Literal["para"] = "para"
    anchor: str

    @field_validator("anchor")
    @classmethod
    def _anchor_grammar(cls, v: str) -> str:
        if not _PARA_ANCHOR_RE.fullmatch(v):
            raise ValueError(
                f"para anchor {v!r} must not contain whitespace or `]`"
            )
        return v

    def format(self) -> str:
        return f"para={self.anchor}"


Locator = Annotated[
    LineRangeLocator
    | PageLocator
    | TimestampLocator
    | TimeRangeLocator
    | BarRangeLocator
    | ParaAnchorLocator,
    Field(discriminator="kind"),
]


class Citation(BaseModel):
    """A single citation anchor: source_id + locator + raw excerpt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    locator: Locator
    excerpt: str

    def render(self) -> str:
        """Return the inline ``[ref:<source_id>:<locator>]`` form."""
        return f"[ref:{self.source_id}:{self.locator.format()}]"
