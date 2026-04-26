"""Wiki store — render and parse WikiPage markdown, maintain index + log.

The wiki layer is the Karpathy-pattern synthesis surface: every
``WikiPage`` round-trips to a single markdown file under
``<vault_root>/wiki/``. The authoritative structured data lives in
YAML frontmatter; the body below is a human-readable rendering of
the same paragraphs with inline ``[ref:...]`` markers.
"""

from __future__ import annotations

from mission_brain.wiki.index_md import update_index
from mission_brain.wiki.log_md import append_log
from mission_brain.wiki.store import read_page, slugify, write_page

__all__ = [
    "append_log",
    "read_page",
    "slugify",
    "update_index",
    "write_page",
]
