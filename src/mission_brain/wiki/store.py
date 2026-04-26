"""Render and parse WikiPage markdown files.

Format (single ``<slug>.md`` file per page under ``<vault_root>/wiki/``):

.. code-block:: text

    ---
    title: Asymmetric Objectives
    source_ids:
      - plain-md-journal-2016-09
    paragraphs:
      - text: A claim.
        citations:
          - source_id: plain-md-journal-2016-09
            locator: lines=45-52
            excerpt: a third way between victory conditions and scoring
    ---

    # Asymmetric Objectives

    A claim. [ref:plain-md-journal-2016-09:lines=45-52]

The YAML frontmatter is authoritative for :func:`read_page`; the
rendered body below is the human-readable view and satisfies the
§2.1 citation-floor regex guard.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from mission_brain.schema import (
    BarRangeLocator,
    Citation,
    LineRangeLocator,
    PageLocator,
    ParaAnchorLocator,
    Paragraph,
    SourceId,
    TimeRangeLocator,
    TimestampLocator,
    WikiPage,
)
from mission_brain.schema.citation import Locator

__all__ = ["read_page", "slugify", "write_page"]


_SLUG_NONALNUM = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """Convert a wiki-page title to its filename slug.

    Lowercase, non-alphanumeric runs collapsed to ``-``, trimmed.
    Falls back to ``"page"`` if the title has no alphanumerics.
    """
    s = _SLUG_NONALNUM.sub("-", title.lower()).strip("-")
    return s or "page"


def _parse_locator(s: str) -> Locator:
    if s.startswith("lines="):
        start_s, end_s = s[len("lines="):].split("-", 1)
        return LineRangeLocator(start=int(start_s), end=int(end_s))
    if s.startswith("page="):
        return PageLocator(page=int(s[len("page="):]))
    if s.startswith("timestamp="):
        rest = s[len("timestamp="):]
        # HH:MM:SS-HH:MM:SS — positions 0-8 and 9-17
        return TimestampLocator(start=rest[:8], end=rest[9:17])
    if s.startswith("time="):
        rest = s[len("time="):]
        # HH:MM:SS.fff-HH:MM:SS.fff — positions 0-12 and 13-25
        return TimeRangeLocator(start=rest[:12], end=rest[13:25])
    if s.startswith("bar="):
        start_s, end_s = s[len("bar="):].split("-", 1)
        return BarRangeLocator(start=int(start_s), end=int(end_s))
    if s.startswith("para="):
        return ParaAnchorLocator(anchor=s[len("para="):])
    raise ValueError(f"unknown locator format: {s!r}")


def _render_body(page: WikiPage) -> str:
    lines: list[str] = [f"# {page.title}", ""]
    for para in page.paragraphs:
        refs = " ".join(c.render() for c in para.citations)
        if refs:
            lines.append(f"{para.text} {refs}".rstrip())
        else:
            # Empty-text paragraphs with no citations are permitted by
            # the schema (blank paragraphs). Preserve the blank slot.
            lines.append(para.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _frontmatter_dict(page: WikiPage) -> dict:
    return {
        "title": page.title,
        "provenance": page.provenance,
        "source_ids": [str(sid) for sid in page.source_ids],
        "paragraphs": [
            {
                "text": para.text,
                "citations": [
                    {
                        "source_id": str(c.source_id),
                        "locator": c.locator.format(),
                        "excerpt": c.excerpt,
                    }
                    for c in para.citations
                ],
            }
            for para in page.paragraphs
        ],
    }


def _disambiguate_slug(base_slug: str, page: WikiPage, vault_root: Path) -> str:
    """Return a slug unique within *vault_root* for *page*.

    If no file exists at the title-derived ``base_slug``, returns
    it unchanged. If a file exists with the same ``source_ids`` as
    *page*, returns ``base_slug`` (legitimate update — overwrite is
    intended). If a file exists with *different* ``source_ids``, we
    have a title-collision between two distinct logical pages —
    append a deterministic 6-char hex suffix derived from *page*'s
    first source_id so both pages can coexist.

    The disambiguator is deterministic per source_id so re-ingest of
    the same source resolves to the same disambiguated slug. Order
    of writes determines which page keeps ``base_slug`` and which
    gets the suffix; the music corpus collision pattern (97/436
    sources, ~22%) was the load-bearing case for this helper.
    """
    candidate = vault_root / "wiki" / f"{base_slug}.md"
    if not candidate.exists():
        return base_slug
    existing = read_page(base_slug, vault_root)
    if existing is None:
        return base_slug
    if set(str(sid) for sid in existing.source_ids) == set(
        str(sid) for sid in page.source_ids
    ):
        return base_slug
    if not page.source_ids:
        return base_slug
    discriminator = hashlib.sha1(
        str(page.source_ids[0]).encode("utf-8")
    ).hexdigest()[:6]
    return f"{base_slug}-{discriminator}"


def write_page(page: WikiPage, vault_root: Path) -> Path:
    """Render *page* to ``<vault_root>/wiki/<slug>.md`` and return the path.

    Writes are deterministic: repeated calls with the same ``page``
    and ``vault_root`` produce byte-identical output.
    """
    wiki_dir = vault_root / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    base_slug = slugify(page.title)
    slug = _disambiguate_slug(base_slug, page, vault_root)
    path = wiki_dir / f"{slug}.md"

    fm = _frontmatter_dict(page)
    fm_yaml = yaml.safe_dump(
        fm,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=1_000_000,
    )
    body = _render_body(page)
    content = f"---\n{fm_yaml}---\n\n{body}"
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def read_page(page_id: str, vault_root: Path) -> WikiPage | None:
    """Parse ``<vault_root>/wiki/<page_id>.md`` back into a :class:`WikiPage`.

    Returns ``None`` if the file does not exist. The YAML frontmatter
    is treated as authoritative; the rendered body is ignored.
    """
    path = vault_root / "wiki" / f"{page_id}.md"
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML frontmatter header")
    end = raw.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"{path}: missing YAML frontmatter terminator")
    fm = yaml.safe_load(raw[4:end + 1])
    if not isinstance(fm, dict):
        raise ValueError(f"{path}: frontmatter is not a mapping")

    paragraphs: list[Paragraph] = []
    for pdict in fm.get("paragraphs", []):
        citations = [
            Citation(
                source_id=SourceId(cdict["source_id"]),
                locator=_parse_locator(cdict["locator"]),
                excerpt=cdict["excerpt"],
            )
            for cdict in pdict.get("citations", [])
        ]
        paragraphs.append(Paragraph(text=pdict["text"], citations=citations))

    return WikiPage(
        title=fm["title"],
        source_ids=[SourceId(s) for s in fm.get("source_ids", [])],
        paragraphs=paragraphs,
        provenance=fm.get("provenance", "primary"),
    )
