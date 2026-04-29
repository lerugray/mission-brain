"""SourceId — the `source_id` half of every citation marker.

CLAUDE.md §1 grammar: ``source_id = [a-z0-9_-]+``. By convention
the value is the corpus-relative path with ``/`` replaced by ``-``
and the extension stripped.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

__all__ = ["SourceId", "SourceIdViolation"]


_SOURCE_ID_RE = re.compile(r"^[a-z0-9_-]+$")
_SID_SCRUB = re.compile(r"[^a-z0-9_-]+")
_SID_COLLAPSE = re.compile(r"-{2,}")


class SourceIdViolation(ValueError):
    """Raised when a value does not match the §1 source_id grammar."""


class SourceId(str):
    """A validated source identifier matching ``[a-z0-9_-]+``.

    Subclasses :class:`str` so existing code that joins / formats /
    compares the value works unchanged; pydantic validation is
    wired via ``__get_pydantic_core_schema__``.
    """

    def __new__(cls, value: str) -> SourceId:
        if not isinstance(value, str):
            raise SourceIdViolation(
                f"source_id must be a string, got {type(value).__name__}"
            )
        if not _SOURCE_ID_RE.fullmatch(value):
            raise SourceIdViolation(
                f"source_id {value!r} does not match grammar [a-z0-9_-]+"
            )
        return super().__new__(cls, value)

    @classmethod
    def coerce(cls, value: str) -> SourceId:
        """Lossy-normalize *value* into a valid SourceId.

        Use at retrieval boundaries where persisted indexes or wiki pages
        may carry legacy non-conformant source_ids. New source IDs should
        still use the strict constructor so generation bugs fail loudly.
        """
        if not isinstance(value, str):
            raise SourceIdViolation(
                f"source_id must be a string, got {type(value).__name__}"
            )
        scrubbed = _SID_SCRUB.sub("-", value.lower())
        coerced = _SID_COLLAPSE.sub("-", scrubbed).strip("-")
        if not coerced:
            raise SourceIdViolation(
                f"source_id {value!r} is empty after scrubbing"
            )
        return cls(coerced)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        def _validate(value: Any) -> SourceId:
            if isinstance(value, cls):
                return value
            return cls(value)

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )
