"""Citation-floor regex guard (CLAUDE.md §2.1 enforcement).

Every non-empty paragraph in a wiki page must carry at least one
well-formed `[ref:<source_id>:<locator>]` marker. Headings (ATX
and Setext) and code blocks (fenced or indented) are exempt.

Grammar per CLAUDE.md §1:
  source_id = [a-z0-9_-]+
  locator   = lines=N-M
            | page=N
            | timestamp=HH:MM:SS-HH:MM:SS
            | time=HH:MM:SS.fff-HH:MM:SS.fff
            | bar=N-M
            | para=<anchor>

This module is a pure function over markdown strings — no I/O,
no stdlib beyond `re`. It is the post-render sweep referenced in
CLAUDE.md §2.1 "Enforced at".
"""

from __future__ import annotations

import re

__all__ = ["CitationFloorViolation", "check_citation_floor"]


class CitationFloorViolation(Exception):
    """A paragraph lacks a valid `[ref:...]` anchor, or a marker is malformed."""


_REF_WELLFORMED = re.compile(
    r"\[ref:[a-z0-9_-]+:"
    r"(?:"
    r"lines=\d+-\d+"
    r"|page=\d+"
    r"|timestamp=\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}"
    r"|time=\d{2}:\d{2}:\d{2}\.\d{3}-\d{2}:\d{2}:\d{2}\.\d{3}"
    r"|bar=\d+-\d+"
    r"|para=[^\]\s]+"
    r")\]"
)

# Any `[ref:...]`-shaped token, whether well-formed or not. Used to
# detect malformed markers that should fail the guard even when
# another valid marker also appears in the same paragraph.
_REF_ANY = re.compile(r"\[ref:[^\]]*\]")

_ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")
_SETEXT_UNDERLINE = re.compile(r"^\s{0,3}(?:=+|-+)\s*$")

# CommonMark §4.1: a line of three or more matching `-`, `_`, or `*`
# characters (each optionally followed by spaces/tabs) is a thematic
# break. Standalone — i.e. preceded and followed by blank lines —
# they form their own block; the LLM emits these as section
# separators in long synthesized wiki pages, and the citation floor
# previously rejected them as paragraphs that "lack a [ref:...:...]
# marker." Encoding them as a structural exemption matches the way
# we exempt headings and code blocks.
_THEMATIC_BREAK = re.compile(
    r"^\s{0,3}(?:-\s*){3,}$"
    r"|^\s{0,3}(?:\*\s*){3,}$"
    r"|^\s{0,3}(?:_\s*){3,}$"
)


def _fenced_code_mask(lines: list[str]) -> list[bool]:
    """Return a per-line boolean: True iff the line is inside (or is)
    a fenced code block. Fences are ``` or ~~~ at most 3 leading spaces.
    """
    mask = [False] * len(lines)
    in_fence = False
    fence_marker: str | None = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        leading_spaces = len(line) - len(stripped)
        is_fence_line = (
            leading_spaces <= 3
            and (stripped.startswith("```") or stripped.startswith("~~~"))
        )
        if is_fence_line:
            if not in_fence:
                in_fence = True
                fence_marker = "```" if stripped.startswith("```") else "~~~"
                mask[i] = True
            elif fence_marker is not None and stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = None
                mask[i] = True
            else:
                mask[i] = True
        elif in_fence:
            mask[i] = True
    return mask


def _blocks(lines: list[str], code_mask: list[bool]) -> list[list[tuple[int, str]]]:
    """Group non-code, non-blank lines into paragraph blocks separated
    by blank lines. Each block is a list of (line_index, line) tuples.
    """
    blocks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if code_mask[i]:
            if current:
                blocks.append(current)
                current = []
            continue
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append((i, line))
    if current:
        blocks.append(current)
    return blocks


def _is_indented_code(block: list[tuple[int, str]]) -> bool:
    """Indented code block: every line starts with 4+ spaces or a tab."""
    return all(line.startswith("    ") or line.startswith("\t") for _, line in block)


def _is_heading(block: list[tuple[int, str]]) -> bool:
    """ATX (first line starts with `#`) or Setext (final line is `=`/`-`
    underline, earlier lines are the heading text).
    """
    if _ATX_HEADING.match(block[0][1]):
        return True
    if len(block) >= 2 and _SETEXT_UNDERLINE.match(block[-1][1]):
        return True
    return False


def _is_thematic_break(block: list[tuple[int, str]]) -> bool:
    """CommonMark thematic break — a single-line block of `---`, `***`,
    or `___` (3+ chars, optional spaces between). Structural separator;
    no citation needed.
    """
    if len(block) != 1:
        return False
    return bool(_THEMATIC_BREAK.match(block[0][1]))


def check_citation_floor(markdown: str) -> None:
    """Verify every non-empty paragraph in *markdown* carries at least
    one well-formed `[ref:<source_id>:<locator>]` marker, and that no
    `[ref:...]`-shaped token in any paragraph is malformed.

    Raises :class:`CitationFloorViolation` on the first failure found.
    Code blocks (fenced or indented) and headings (ATX or Setext) are
    exempt.
    """
    lines = markdown.splitlines()
    code_mask = _fenced_code_mask(lines)
    for block in _blocks(lines, code_mask):
        if (
            _is_heading(block)
            or _is_indented_code(block)
            or _is_thematic_break(block)
        ):
            continue
        text = "\n".join(line for _, line in block)

        # Every `[ref:...]`-shaped token must be well-formed, even if
        # another valid marker also appears in the paragraph.
        for m in _REF_ANY.finditer(text):
            if not _REF_WELLFORMED.fullmatch(m.group(0)):
                raise CitationFloorViolation(
                    f"Malformed citation marker {m.group(0)!r} "
                    f"in paragraph starting at line {block[0][0] + 1}"
                )

        if not _REF_WELLFORMED.search(text):
            preview = text if len(text) <= 120 else text[:117] + "..."
            raise CitationFloorViolation(
                f"Paragraph starting at line {block[0][0] + 1} "
                f"lacks a [ref:<source>:<locator>] marker: {preview!r}"
            )
