"""Music-data loader — reads pretty_midi-exported JSON (rayb-013-c).

Handles two shapes that catalogdna produces from the user's own audio:

* **Per-song JSON** — one file per track under ``music/per-song/``.
  Renders a deterministic plain-text body (time-labeled note lines
  sorted by start time) so the ingest LLM can emit
  ``[ref:...:time=HH:MM:SS.fff-HH:MM:SS.fff]`` citations anchored in
  the body text.
* **Catalog aggregate JSON** — a single ``catalog_analysis.json``
  under ``music/catalog/``. Rendered as a keyed flat summary
  (scale fingerprint, top chords, top motifs, register/rhythm
  distributions). Cited via ``[ref:...:para=<top-level-key>]``.

Provenance is ``"primary"``: the JSONs are mechanical extraction
from the user's audio (basic-pitch → pretty_midi), not downstream
synthesis.

This loader is pure-stdlib. pretty_midi / music21 are NOT
dependencies — we consume the already-exported JSON shape.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mission_brain.loaders.base import SourceDocument, SourceLoader
from mission_brain.schema.wiki_page import Provenance

__all__ = ["MusicDataLoader"]

_CATALOG_FILENAME = "catalog_analysis.json"
_TOP_N_CHORDS = 20
_TOP_N_MOTIFS = 20


class MusicDataLoader(SourceLoader):
    source_type = "music_data"
    supported_extensions = [".json"]
    version = "1.0"

    def __init__(self, provenance: Provenance = "primary") -> None:
        self._provenance: Provenance = provenance

    def discover(self, corpus_root: Path) -> Iterable[Path]:
        """Yield sorted per-song JSONs under ``music/per-song/``."""
        per_song = corpus_root / "music" / "per-song"
        if not per_song.is_dir():
            return []
        return sorted(p for p in per_song.glob("*.json") if p.is_file())

    def discover_catalog(self, corpus_root: Path) -> Iterable[Path]:
        """Yield the single catalog aggregate JSON if present."""
        catalog = corpus_root / "music" / "catalog" / _CATALOG_FILENAME
        if catalog.is_file():
            return [catalog]
        return []

    def load(self, path: Path) -> SourceDocument:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))

        if path.name == _CATALOG_FILENAME:
            body, frontmatter_extras, extras = _render_catalog(data)
            title = "Music Catalog Aggregate"
        else:
            body, frontmatter_extras, extras = _render_per_song(data)
            title = str(data.get("title") or path.stem)

        line_count = len(body.splitlines())

        return SourceDocument(
            title=title,
            body=body,
            frontmatter=frontmatter_extras,
            line_count=line_count,
            content_hash=self.content_hash(raw),
            source_path=path,
            source_type=self.source_type,
            loader_version=self.version,
            provenance=self._provenance,
            extras=extras,
        )

    def content_hash(self, raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def locator_for(self, span: tuple[int, int]) -> str:
        raise NotImplementedError(
            "MusicDataLoader.locator_for is deferred to Phase 3+; the ingest "
            "LLM emits [ref:...:time=...] markers from the body's time labels "
            "directly, and the citation-floor regex validates them."
        )


def _format_time(seconds: float) -> str:
    """Render a float seconds value as ``HH:MM:SS.fff`` (CLAUDE.md §1 grammar).

    Rounds to 3 decimal places deterministically. Negative values clamp
    to zero — a pretty_midi export should never produce them but the
    defensive clamp keeps the rendering total.
    """
    s = max(0.0, float(seconds))
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    secs_int = int(s % 60)
    millis = int(round((s - int(s)) * 1000))
    # Rounding can push millis to 1000; renormalize.
    if millis == 1000:
        millis = 0
        secs_int += 1
        if secs_int == 60:
            secs_int = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs_int:02d}.{millis:03d}"


def _render_per_song(
    data: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    title = str(data.get("title", "")).strip()
    tempo = data.get("tempo_estimates")
    duration = data.get("duration_seconds")
    time_sigs = data.get("time_signatures") or []
    key_sigs = data.get("key_signatures") or []
    instruments = data.get("instruments") or []

    lines: list[str] = []
    lines.append(f"title: {title}")
    lines.append(f"tempo: {tempo}")
    lines.append(f"duration_seconds: {duration}")

    lines.append("time_signatures:")
    if not time_sigs:
        lines.append("  (none)")
    else:
        for ts in time_sigs:
            num = ts.get("numerator")
            den = ts.get("denominator")
            t = ts.get("time")
            lines.append(f"  {num}/{den} @ {_format_time(t or 0.0)}")

    lines.append("key_signatures:")
    if not key_sigs:
        lines.append("  (none)")
    else:
        for ks in key_sigs:
            name = ks.get("key_name", ks.get("name", "?"))
            t = ks.get("time", 0.0)
            lines.append(f"  {name} @ {_format_time(t)}")

    # Collect all notes with instrument index for deterministic sort.
    all_notes: list[tuple[float, int, int, float, str, int]] = []
    # (start, pitch, inst_idx, end, pitch_name, velocity)

    for inst_idx, inst in enumerate(instruments):
        program = inst.get("program")
        is_drum = bool(inst.get("is_drum"))
        note_count = inst.get("note_count", len(inst.get("notes") or []))
        name = str(inst.get("name", f"Track {inst_idx}"))
        lines.append(
            f"track {inst_idx} ({name}, program {program}, "
            f"drum: {str(is_drum).lower()}): {note_count} notes"
        )
        for note in inst.get("notes") or []:
            start = float(note.get("start", 0.0))
            end = float(note.get("end", start))
            pitch = int(note.get("pitch", 0))
            pitch_name = str(note.get("pitch_name", str(pitch)))
            velocity = int(note.get("velocity", 0))
            all_notes.append((start, pitch, inst_idx, end, pitch_name, velocity))

    all_notes.sort(key=lambda t: (t[0], t[1], t[2]))

    lines.append("notes:")
    time_index: list[tuple[float, float, int]] = []
    for start, _pitch, _inst_idx, end, pitch_name, velocity in all_notes:
        body_line = len(lines) + 1  # 1-based line index in final body
        lines.append(
            f"  {_format_time(start)}-{_format_time(end)} "
            f"{pitch_name} vel={velocity}"
        )
        time_index.append((start, end, body_line))

    body = "\n".join(lines)

    frontmatter = {
        "tempo": tempo,
        "duration_seconds": duration,
        "num_instruments": len(instruments),
        "source_file": data.get("source_file"),
    }
    extras = {"time_index": time_index}
    return body, frontmatter, extras


def _render_catalog(
    data: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    lines: list[str] = []
    lines.append("title: Music Catalog Aggregate")

    stats = data.get("catalog_stats") or {}
    if stats:
        lines.append("catalog_stats:")
        for k, v in sorted(stats.items()):
            lines.append(f"  {k}: {v}")

    lines.append("scale_fingerprint:")
    scale = data.get("scale_fingerprint") or {}
    if isinstance(scale, dict):
        for k, v in sorted(scale.items(), key=lambda kv: (-_as_int(kv[1]), kv[0])):
            lines.append(f"  {k}: {v}")
    else:
        lines.append(f"  {scale!r}")

    lines.append("chord_vocabulary:")
    chords = data.get("chord_vocabulary") or {}
    if isinstance(chords, dict):
        top_chords = sorted(
            chords.items(), key=lambda kv: (-_as_int(kv[1]), kv[0])
        )[:_TOP_N_CHORDS]
        for k, v in top_chords:
            lines.append(f"  {k}: {v}")
    else:
        lines.append(f"  {chords!r}")

    lines.append("top_motifs:")
    motifs = data.get("top_motifs") or {}
    if isinstance(motifs, dict):
        top_motifs = sorted(
            motifs.items(), key=lambda kv: (-_as_int(kv[1]), kv[0])
        )[:_TOP_N_MOTIFS]
        for k, v in top_motifs:
            lines.append(f"  {k}: {v}")
    elif isinstance(motifs, list):
        for entry in motifs[:_TOP_N_MOTIFS]:
            lines.append(f"  {entry!r}")

    lines.append("register_usage:")
    reg = data.get("register_usage") or {}
    if isinstance(reg, dict):
        for k, v in sorted(reg.items()):
            lines.append(f"  {k}: {v}")

    lines.append("rhythm_distribution:")
    rhy = data.get("rhythm_distribution") or {}
    if isinstance(rhy, dict):
        for k, v in sorted(rhy.items()):
            lines.append(f"  {k}: {v}")

    body = "\n".join(lines)
    frontmatter = {
        "catalog_keys": sorted(data.keys()),
    }
    extras: dict[str, Any] = {}
    return body, frontmatter, extras


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
