"""Ingest orchestrator — SourceDocument + synthesis client → WikiPage.

Calls the client, runs the §2.1 citation-floor sweep, retries with
a correction nudge up to :data:`MAX_RETRIES` times on violation,
then parses the validated markdown into a :class:`WikiPage`.

Excerpt resolution is best-effort: line-range locators whose
source_id matches *doc* slice *doc.body* to populate the excerpt;
cross-document resolution is a later task's job.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from mission_brain.ingest.citation_alignment import (
    CitationAlignmentFailure,
    realign_line_locators_for_document,
)
from mission_brain.ingest.citation_floor import CitationFloorViolation, check_citation_floor
from mission_brain.ingest.claude_cli_client import ClaudeCLIClient
from mission_brain.ingest.idempotence import (
    IngestManifest,
    compute_ingest_manifest,
    write_manifest_sidecar,
)
from mission_brain.ingest.prompt import build_ingest_prompt
from mission_brain.loaders.base import SourceDocument
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
from mission_brain.schema.source_id import SourceId
from mission_brain.schema.wiki_page import Paragraph, WikiPage

__all__ = [
    "BatchResult",
    "FAILURES_FILENAME",
    "MAX_RETRIES",
    "PROGRESS_FILENAME",
    "SIDECAR_SUFFIX",
    "compute_source_manifest",
    "ingest_source",
    "manifest_matches_sidecar",
    "run_ingest_batch",
    "sidecar_path_for",
]

PROGRESS_FILENAME = ".ingest-progress.json"
FAILURES_FILENAME = ".ingest-failures.md"
SIDECAR_SUFFIX = ".manifest.json"

MAX_RETRIES = 2

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
_ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")
_SETEXT_UNDERLINE = re.compile(r"^\s{0,3}(?:=+|-+)\s*$")
_ATX_TITLE = re.compile(r"^\s{0,3}#\s+(?P<title>.+?)\s*#*\s*$")


def _parse_locator(raw: str) -> Locator:
    if raw.startswith("lines="):
        s, e = raw.removeprefix("lines=").split("-", 1)
        return LineRangeLocator(start=int(s), end=int(e))
    if raw.startswith("page="):
        return PageLocator(page=int(raw.removeprefix("page=")))
    if raw.startswith("timestamp="):
        s, e = raw.removeprefix("timestamp=").split("-", 1)
        return TimestampLocator(start=s, end=e)
    if raw.startswith("time="):
        # HH:MM:SS.fff-HH:MM:SS.fff — fixed-width 12-char halves around the `-`.
        rest = raw.removeprefix("time=")
        return TimeRangeLocator(start=rest[:12], end=rest[13:25])
    if raw.startswith("bar="):
        s, e = raw.removeprefix("bar=").split("-", 1)
        return BarRangeLocator(start=int(s), end=int(e))
    if raw.startswith("para="):
        return ParaAnchorLocator(anchor=raw.removeprefix("para="))
    raise ValueError(f"unrecognized locator syntax: {raw!r}")


def _resolve_excerpt(sid: SourceId, loc: Locator, doc: SourceDocument) -> str:
    expected = doc.source_id or doc.source_path.stem
    if str(sid) != expected or not isinstance(loc, LineRangeLocator):
        return ""
    lines = doc.body.splitlines()
    return "\n".join(lines[max(loc.start - 1, 0) : min(loc.end, len(lines))])


def _unwrap_outer_fence(markdown: str) -> str:
    """Strip a single outer ```...``` fence if it wraps the whole response.

    LLMs habitually wrap markdown output in a ```markdown ... ``` fence even
    when asked not to. Treat fenced content as the intended markdown when the
    fence encloses the entire (trimmed) response.
    """
    lines = markdown.splitlines()
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    j = len(lines) - 1
    while j > i and lines[j].strip() == "":
        j -= 1
    if i >= j:
        return markdown
    opener = lines[i].lstrip()
    closer = lines[j].lstrip()
    marker = None
    if opener.startswith("```"):
        marker = "```"
    elif opener.startswith("~~~"):
        marker = "~~~"
    if marker and closer.startswith(marker) and closer.strip() == marker:
        return "\n".join(lines[i + 1 : j])
    return markdown


def _split_blocks(markdown: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    in_fence = False
    marker: str | None = None
    for line in markdown.splitlines():
        stripped = line.lstrip()
        leading = len(line) - len(stripped)
        is_fence = leading <= 3 and (
            stripped.startswith("```") or stripped.startswith("~~~")
        )
        if is_fence:
            if current:
                blocks.append(current)
                current = []
            if not in_fence:
                in_fence, marker = True, "```" if stripped.startswith("```") else "~~~"
            elif marker and stripped.startswith(marker):
                in_fence, marker = False, None
            continue
        if in_fence:
            continue
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _is_heading(block: list[str]) -> bool:
    if _ATX_HEADING.match(block[0]):
        return True
    return len(block) >= 2 and bool(_SETEXT_UNDERLINE.match(block[-1]))


def _extract_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        m = _ATX_TITLE.match(line)
        if m:
            return m.group("title").strip()
    return fallback


def _parse_to_wiki_page(markdown: str, doc: SourceDocument) -> WikiPage:
    title = _extract_title(markdown, fallback=doc.title)
    paragraphs: list[Paragraph] = []
    seen: list[SourceId] = []
    for block in _split_blocks(markdown):
        if _is_heading(block):
            continue
        text = "\n".join(block)
        citations: list[Citation] = []
        for m in _REF_MARKER.finditer(text):
            sid = SourceId(m.group("sid"))
            loc = _parse_locator(m.group("loc"))
            citations.append(
                Citation(source_id=sid, locator=loc, excerpt=_resolve_excerpt(sid, loc, doc))
            )
            if sid not in seen:
                seen.append(sid)
        paragraphs.append(
            Paragraph(text=_REF_MARKER.sub("", text).strip(), citations=citations)
        )
    return WikiPage(
        title=title,
        source_ids=seen,
        paragraphs=paragraphs,
        provenance=doc.provenance,
    )


def ingest_source(
    doc: SourceDocument,
    prior_wiki: WikiPage | None,
    client: ClaudeCLIClient,
    claude_md_text: str = "",
) -> WikiPage:
    """Synthesize a :class:`WikiPage` from *doc* via *client*."""
    base_prompt = build_ingest_prompt(doc, prior_wiki, claude_md_text)
    last_error: CitationFloorViolation | CitationAlignmentFailure | None = None
    for attempt in range(MAX_RETRIES + 1):
        prompt = base_prompt
        if attempt > 0 and last_error is not None:
            if isinstance(last_error, CitationFloorViolation):
                extra = (
                    f"\n\nCORRECTION: previous response violated the citation floor "
                    f"({last_error}). Every non-empty paragraph must carry at least one "
                    "[ref:<source_id>:<locator>] marker. Try again.\n"
                )
            else:
                extra = (
                    f"\n\nCORRECTION: {last_error} "
                    "Cite the numbered SOURCE DOCUMENT lines that actually support each "
                    "claim — for lines=N-M, N and M are the 1-based numbers in the left "
                    "column next to the cited text. Try again.\n"
                )
            prompt = base_prompt + extra
        markdown = _unwrap_outer_fence(client.generate(prompt))
        try:
            check_citation_floor(markdown)
            markdown = realign_line_locators_for_document(markdown, doc)
        except CitationFloorViolation as exc:
            last_error = exc
            continue
        except CitationAlignmentFailure as exc:
            last_error = exc
            continue
        return _parse_to_wiki_page(markdown, doc)
    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Checkpoint-driven batch ingest (rayb-015-c)


@dataclass
class BatchResult:
    """Tally returned from :func:`run_ingest_batch`.

    ``skipped_checkpoint`` records intra-run (or resumed-run)
    dedup via ``.ingest-progress.json`` — "we already finished this
    source_id". ``skipped_manifest`` records cross-run content-aware
    dedup via the per-page manifest sidecar — "this source's inputs
    are byte-identical to what produced the existing wiki page, so
    the LLM call is redundant". Keeping the two buckets distinct
    lets telemetry distinguish "resumed work" from "cache hit".
    """

    attempted: int = 0
    completed: list[str] = field(default_factory=list)
    skipped_checkpoint: list[str] = field(default_factory=list)
    skipped_manifest: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _load_progress(vault_root: Path) -> set[str]:
    path = vault_root / PROGRESS_FILENAME
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(data, dict):
        return set()
    completed = data.get("completed")
    if not isinstance(completed, list):
        return set()
    return {str(s) for s in completed}


def _write_progress(vault_root: Path, completed: set[str]) -> None:
    """Write canonical JSON + fsync so the checkpoint survives a crash."""
    path = vault_root / PROGRESS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"completed": sorted(completed)}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass


def _append_failure(
    vault_root: Path, doc: SourceDocument, error: BaseException
) -> None:
    path = vault_root / FAILURES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sid = doc.source_id or doc.source_path.stem
    entry = (
        f"- {ts} — source_id=`{sid}` path=`{doc.source_path}` "
        f"error=`{type(error).__name__}: {error}`\n"
    )
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(entry)


def run_ingest_batch(
    sources: Iterable[SourceDocument],
    prior_fn: Callable[[SourceDocument], WikiPage | None],
    client: ClaudeCLIClient,
    vault_root: Path,
    force: bool = False,
    writer: Callable[[WikiPage, SourceDocument], None] | None = None,
    claude_md_text: str = "",
    config_digest: str | None = None,
    prior_sidecar_path_fn: Callable[[SourceDocument], Path | None] | None = None,
    new_sidecar_path_fn: Callable[[WikiPage, SourceDocument], Path] | None = None,
    overlay_path_fn: Callable[[SourceDocument], Path | None] | None = None,
    loader_version_fn: Callable[[SourceDocument], str] | None = None,
) -> BatchResult:
    """Ingest many sources with crash-safe checkpointing + §2.2 idempotence.

    Maintains ``<vault_root>/.ingest-progress.json`` with a sorted list of
    completed ``source_id``s; on re-entry, sources whose id is already in
    that list are skipped unless ``force=True``. Any exception during a
    single source's ingest is logged to ``<vault_root>/.ingest-failures.md``
    and the batch continues with the next source.

    ``writer`` runs after a successful :func:`ingest_source` call and is
    where callers persist the resulting :class:`WikiPage`. The checkpoint
    is only committed after ``writer`` returns — a writer failure marks
    the source as failed, not completed.

    When ``config_digest``, ``prior_sidecar_path_fn``, and
    ``new_sidecar_path_fn`` are all supplied, manifest-sidecar idempotence
    per CLAUDE.md §2.2 activates:

    1. Before the LLM call, compute a per-source
       :class:`IngestManifest` folding in ``claude_md_text``, loader
       version, and overlay hash.
    2. If ``prior_sidecar_path_fn(doc)`` resolves to a sidecar whose
       canonical bytes match the current manifest, skip the LLM (and
       writer) entirely and record the source_id in
       ``skipped_manifest``.
    3. Otherwise ingest normally, then write the fresh sidecar at
       ``new_sidecar_path_fn(page, doc)`` after ``writer`` succeeds.

    ``force=True`` also bypasses the manifest cache.
    """
    completed_set = _load_progress(vault_root) if not force else set()
    result = BatchResult()
    idempotence_enabled = (
        not force
        and config_digest is not None
        and prior_sidecar_path_fn is not None
        and new_sidecar_path_fn is not None
    )
    for doc in sources:
        result.attempted += 1
        sid = doc.source_id or doc.source_path.stem
        if not force and sid in completed_set:
            result.skipped_checkpoint.append(sid)
            continue

        manifest: IngestManifest | None = None
        if idempotence_enabled:
            assert config_digest is not None  # narrowed by idempotence_enabled
            loader_version = (
                loader_version_fn(doc) if loader_version_fn is not None
                else (doc.loader_version or "")
            )
            overlay_path = (
                overlay_path_fn(doc) if overlay_path_fn is not None else None
            )
            manifest = compute_source_manifest(
                doc, loader_version, config_digest, overlay_path, claude_md_text
            )
            assert prior_sidecar_path_fn is not None
            prior_sidecar = prior_sidecar_path_fn(doc)
            if prior_sidecar is not None and manifest_matches_sidecar(
                prior_sidecar, manifest
            ):
                result.skipped_manifest.append(sid)
                continue

        try:
            page = ingest_source(doc, prior_fn(doc), client, claude_md_text)
            if writer is not None:
                writer(page, doc)
        except Exception as exc:  # noqa: BLE001 — intentionally broad
            _append_failure(vault_root, doc, exc)
            result.failed.append(sid)
            continue
        if idempotence_enabled and manifest is not None:
            assert new_sidecar_path_fn is not None
            write_manifest_sidecar(manifest, new_sidecar_path_fn(page, doc))
        completed_set.add(sid)
        _write_progress(vault_root, completed_set)
        result.completed.append(sid)
    return result


# ---------------------------------------------------------------------------
# Manifest-sidecar idempotence (rayb-019; CLAUDE.md §2.2).
#
# These helpers compose the compute_ingest_manifest primitive (from
# rayb-005) with the per-page sidecar location. Callers (CLI wiring)
# use them to short-circuit the LLM when a source's inputs haven't
# changed between runs.


def compute_source_manifest(
    doc: SourceDocument,
    loader_version: str,
    config_digest: str,
    overlay_path: Path | None,
    claude_md_text: str,
) -> IngestManifest:
    """Fingerprint every input that feeds one wiki page.

    Folds the sha256 of ``claude_md_text`` into the effective config
    digest so a CLAUDE.md edit invalidates every cached manifest —
    CLAUDE.md §2.2 requires the schema file's content to participate
    in the manifest hash.
    """
    sid = doc.source_id or doc.source_path.stem
    source_hashes = {sid: doc.content_hash}
    loader_versions = {doc.source_type or "unknown": loader_version or "unknown"}
    overlay_hashes: dict[str, str] = {}
    if overlay_path is not None and overlay_path.is_file():
        overlay_hashes[overlay_path.name] = hashlib.sha256(
            overlay_path.read_bytes()
        ).hexdigest()
    claude_hash = hashlib.sha256(claude_md_text.encode("utf-8")).hexdigest()
    effective_digest = hashlib.sha256(
        f"{config_digest}|claude_md={claude_hash}".encode()
    ).hexdigest()
    return compute_ingest_manifest(
        source_hashes, loader_versions, effective_digest, overlay_hashes
    )


def sidecar_path_for(vault_root: Path, slug: str) -> Path:
    """Return the manifest sidecar path for a given wiki-page slug."""
    return vault_root / "wiki" / f"{slug}{SIDECAR_SUFFIX}"


def manifest_matches_sidecar(sidecar_path: Path, manifest: IngestManifest) -> bool:
    """True iff *sidecar_path* already holds the canonical bytes of *manifest*.

    Byte-equality is the strict check: sidecars are always written via
    :func:`write_manifest_sidecar`, which emits the canonical JSON
    unchanged. Any non-canonical or foreign sidecar content is treated
    as a miss, which forces a re-ingest — safer than trusting parsed
    content.
    """
    if not sidecar_path.is_file():
        return False
    try:
        return sidecar_path.read_bytes() == manifest.canonical_json
    except OSError:
        return False
