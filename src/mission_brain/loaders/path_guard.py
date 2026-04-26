"""Filesystem containment guard (extracted for mission-brain without the full privacy package).

`assert_under_corpus(path, corpus_root)` raises
`PathEscapeViolation` if the resolved path is not under the
resolved corpus root. `Path.resolve()` follows symlinks, so a
symlink pointing outside the corpus root is caught.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PathEscapeViolation", "assert_under_corpus"]


class PathEscapeViolation(Exception):
    """A path resolves outside the permitted corpus root."""


def assert_under_corpus(path: Path, corpus_root: Path) -> None:
    """Raise `PathEscapeViolation` if `path` is not inside `corpus_root`.

    Both arguments are resolved (symlinks followed, `..` collapsed)
    before comparison. `path == corpus_root` is accepted (the root
    itself is considered "under" itself).
    """

    resolved_path = Path(path).resolve()
    resolved_root = Path(corpus_root).resolve()

    if resolved_path == resolved_root:
        return

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise PathEscapeViolation(
            f"Path {resolved_path!s} is not under corpus root {resolved_root!s}"
        ) from exc
