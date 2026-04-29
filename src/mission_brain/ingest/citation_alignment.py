"""Post-synthesis ``lines=`` locator realignment.

The plain-markdown ingest prompt lists ``lines=N-M`` locators, but the
model can miscount line numbers. After synthesis, we score how well
each cited line range's excerpt matches the host paragraph. Poor scores
trigger a local search for a better window in the same SourceDocument.

Non-``lines=`` locators and refs whose ``source_id`` does not match
*doc* are left unchanged.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from mission_brain.loaders.base import SourceDocument

__all__ = [
    "CitationAlignmentFailure",
    "realign_line_locators_for_document",
]

# Match pipeline._REF_MARKER. Keep regexes in sync when changing grammar.
_REF_MARKER = re.compile(
    r"\[ref:(?P<sid>[a-z0-9_-]+):"
    r"(?P<loc>"
    r"lines=\d+-\d+"
    r"|page=\d+"
    r"|timestamp=\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}"
    r"|time=\d{2}:\d{2}:\d{2}\.\d{3}-\d{2}:\d{2}:\d{2}\.\d{3}"
    r"|bar=\d+-\d+"
    r"|para=[^\]\s]+"
    r")\]"
)

# Current excerpt vs paragraph: keep the LLM's locator.
_SCORE_KEEP: float = 0.14
# A replacement must still plausibly match the paragraph.
_SCORE_MIN_BEST: float = 0.04
# If the best line window is identical to the model's but still weak, fail.
_SCORE_STUCK: float = 0.1

_MAX_RANGE_LINES: int = 32
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _word_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def _jaccard(a: str, b: str) -> float:
    wa, wb = _word_set(a), _word_set(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union else 0.0


def _ratio(a: str, b: str) -> float:
    """Token-level similarity; rewards substring matches and paraphrase."""
    if not a.strip() or not b.strip():
        return 0.0
    a2 = " ".join(_TOKEN_RE.findall(a)).lower()
    b2 = " ".join(_TOKEN_RE.findall(b)).lower()
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()


def _hybrid_score(para: str, excerpt: str) -> float:
    j = _jaccard(para, excerpt)
    r = _ratio(para, excerpt)
    return 0.5 * j + 0.5 * r


def _parse_lines_locator(raw: str) -> tuple[int, int] | None:
    if not raw.startswith("lines="):
        return None
    s, e = raw.removeprefix("lines=").split("-", 1)
    return int(s), int(e)


def _excerpt_for_range(body: str, start: int, end: int) -> str:
    lines = body.splitlines()
    s0, e0 = max(start - 1, 0), min(end, len(lines))
    if s0 >= e0:
        return ""
    return "\n".join(lines[s0:e0])


def _block_around(markdown: str, pos: int) -> str:
    """Text block, double-newline delimited, containing *pos*."""
    before = markdown.rfind("\n\n", 0, pos)
    start = 0 if before < 0 else before + 2
    after = markdown.find("\n\n", pos)
    end = len(markdown) if after < 0 else after
    return markdown[start:end]


def _best_line_range(
    body: str, paragraph_context: str
) -> tuple[tuple[int, int], float] | None:
    """Return ((start, end) 1-based, score) for the best line window."""
    lines = body.splitlines()
    n = len(lines)
    if n == 0 or not paragraph_context.strip():
        return None
    best: tuple[tuple[int, int], float] | None = None
    for start in range(1, n + 1):
        for width in range(1, _MAX_RANGE_LINES + 1):
            end = start + width - 1
            if end > n:
                break
            excerpt = "\n".join(lines[start - 1 : end])
            sc = _hybrid_score(paragraph_context, excerpt)
            t = (start, end)
            if best is None or sc > best[1] or (sc == best[1] and t < best[0]):
                best = (t, sc)
    return best


class CitationAlignmentFailure(Exception):
    """No usable ``lines=`` locator could be aligned to the source body."""


def realign_line_locators_for_document(markdown: str, doc: SourceDocument) -> str:
    """Return *markdown* with in-document ``lines=`` ref markers adjusted."""
    expected = doc.source_id or doc.source_path.stem
    body = doc.body

    replacements: list[tuple[slice, str]] = []
    for m in _REF_MARKER.finditer(markdown):
        loc_raw = m.group("loc")
        if not loc_raw.startswith("lines="):
            continue
        if m.group("sid") != expected:
            continue
        range_t = _parse_lines_locator(loc_raw)
        if range_t is None:
            continue
        ls, le = range_t
        start_ch, end_ch = m.start("loc"), m.end("loc")

        block = _block_around(markdown, m.start())
        para = _REF_MARKER.sub("", block)
        para = re.sub(r"\s+", " ", para).strip()

        current_ex = _excerpt_for_range(body, ls, le)
        cur_sc = _hybrid_score(para, current_ex)
        if cur_sc >= _SCORE_KEEP:
            continue

        best = _best_line_range(body, para)
        if best is None:
            raise CitationAlignmentFailure(f"no line range in source for ref at offset {m.start()}")
        (b_start, b_end), best_sc = best
        if best_sc < _SCORE_MIN_BEST:
            raise CitationAlignmentFailure(
                f"line ref at offset {m.start()}: best score {best_sc:.3f} "
                f"below floor (paragraph may be hallucinated w.r.t. source body)"
            )
        if (b_start, b_end) == (ls, le) and best_sc < _SCORE_STUCK:
            raise CitationAlignmentFailure(
                f"line ref at offset {m.start()}: locator lines={ls}-{le} is "
                f"the only candidate but match score {best_sc:.3f} is still weak"
            )
        if (b_start, b_end) == (ls, le):
            continue
        if best_sc <= cur_sc:
            continue
        new_loc = f"lines={b_start}-{b_end}"
        replacements.append((slice(start_ch, end_ch), new_loc))

    if not replacements:
        return markdown
    result_parts: list[str] = []
    last = 0
    for sl, new in sorted(replacements, key=lambda x: x[0].start):
        result_parts.append(markdown[last : sl.start])
        result_parts.append(new)
        last = sl.stop
    result_parts.append(markdown[last:])
    return "".join(result_parts)
