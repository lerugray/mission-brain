"""Idempotent-regen manifest plumbing (CLAUDE.md §2.2 foundation).

Running ``mission-brain ingest`` twice over the same ``(corpus + config +
user-edit overlay)`` must produce byte-identical ``vault/`` output.
This module is the hash-plumbing layer: given the hashes of the
inputs that feed a wiki page, produce a deterministic manifest whose
``manifest_hash`` is sufficient to decide "regenerate or skip" at
step 2 of the ingest pipeline (CLAUDE.md §3.1).

The full §2.2 test (run ingest twice, diff outputs) requires the
schema models plus the ingest pipeline that uses them; both land
later. This module is the pure hashing primitive that pipeline will
call.

Canonical JSON rules:
  * UTF-8 encoded.
  * Keys sorted at every level (dicts-of-dicts are sorted recursively
    by ``json.dumps(sort_keys=True)``).
  * ``separators=(",", ":")`` — no insignificant whitespace.
  * No trailing newline.

``manifest_hash`` is ``sha256`` over the canonical JSON bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "IngestManifest",
    "compute_ingest_manifest",
    "write_manifest_sidecar",
]


@dataclass(frozen=True)
class IngestManifest:
    """A deterministic fingerprint of the inputs feeding one wiki page.

    ``component_hashes`` is the canonical dict that was hashed — kept
    alongside the hash so a caller can inspect which component changed
    between two manifests without re-deriving.
    """

    manifest_hash: str
    canonical_json: bytes
    component_hashes: dict[str, dict[str, str] | str]


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    """Serialize *payload* to canonical JSON bytes.

    ``sort_keys=True`` recurses into nested dicts; ``separators``
    removes insignificant whitespace; ``ensure_ascii=False`` keeps the
    output UTF-8 rather than escaping non-ASCII runes into ``\\uXXXX``
    (escaping would still be deterministic but wastes bytes).
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_ingest_manifest(
    source_hashes: dict[str, str],
    loader_versions: dict[str, str],
    config_digest: str,
    overlay_hashes: dict[str, str],
) -> IngestManifest:
    """Compute a deterministic ingest manifest.

    Parameters
    ----------
    source_hashes
        Mapping of ``source_id`` → content hash of the raw-corpus file.
    loader_versions
        Mapping of ``source_type`` → loader version string.
    config_digest
        A digest of the active mission-brain config (caller computes).
    overlay_hashes
        Mapping of wiki-page path → hash of its ``*.user-edits.md``
        overlay file, if any.

    Returns
    -------
    IngestManifest
        Frozen dataclass with ``manifest_hash``, the canonical JSON
        bytes that were hashed, and the component-hashes dict.
    """
    component_hashes: dict[str, dict[str, str] | str] = {
        "source_hashes": dict(source_hashes),
        "loader_versions": dict(loader_versions),
        "config_digest": config_digest,
        "overlay_hashes": dict(overlay_hashes),
    }
    canonical = _canonical_json_bytes(component_hashes)
    manifest_hash = hashlib.sha256(canonical).hexdigest()
    return IngestManifest(
        manifest_hash=manifest_hash,
        canonical_json=canonical,
        component_hashes=component_hashes,
    )


def write_manifest_sidecar(manifest: IngestManifest, sidecar_path: Path) -> None:
    """Write the canonical JSON of *manifest* to *sidecar_path*.

    The caller decides where the sidecar lives — this function does
    not assume ``vault/<page>.manifest.json``; that wiring is a later
    task.
    """
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_bytes(manifest.canonical_json)
