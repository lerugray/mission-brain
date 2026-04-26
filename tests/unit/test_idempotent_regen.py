"""Tests for the idempotent-regen manifest plumbing (CLAUDE.md §2.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_brain.ingest.idempotence import (
    IngestManifest,
    compute_ingest_manifest,
    write_manifest_sidecar,
)


def _sample_inputs() -> dict[str, object]:
    return {
        "source_hashes": {
            "plain-md-journal-2016-09": "a" * 64,
            "plain-md-notes-2018-rambling": "b" * 64,
        },
        "loader_versions": {"plain_markdown": "0.1.0"},
        "config_digest": "c" * 64,
        "overlay_hashes": {"vault/wiki/asymmetric-objectives.md": "d" * 64},
    }


def test_manifest_hash_stable_across_runs() -> None:
    """Running the computation 100 times over the same inputs yields
    byte-identical manifest bytes and identical hashes."""
    inputs = _sample_inputs()
    first = compute_ingest_manifest(**inputs)  # type: ignore[arg-type]
    for _ in range(100):
        again = compute_ingest_manifest(**inputs)  # type: ignore[arg-type]
        assert again.manifest_hash == first.manifest_hash
        assert again.canonical_json == first.canonical_json


def test_manifest_hash_invariant_under_key_permutation() -> None:
    """Input dicts are logically unordered — feeding them in a
    different insertion order must not change the manifest hash."""
    inputs = _sample_inputs()
    reference = compute_ingest_manifest(**inputs)  # type: ignore[arg-type]

    permuted = {
        "source_hashes": {
            "plain-md-notes-2018-rambling": "b" * 64,
            "plain-md-journal-2016-09": "a" * 64,
        },
        "loader_versions": {"plain_markdown": "0.1.0"},
        "config_digest": "c" * 64,
        "overlay_hashes": {"vault/wiki/asymmetric-objectives.md": "d" * 64},
    }
    result = compute_ingest_manifest(**permuted)  # type: ignore[arg-type]
    assert result.manifest_hash == reference.manifest_hash


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_hashes": {"plain-md-journal-2016-09": "z" * 64}},
        {"loader_versions": {"plain_markdown": "0.2.0"}},
        {"config_digest": "e" * 64},
        {"overlay_hashes": {"vault/wiki/asymmetric-objectives.md": "f" * 64}},
        {"source_hashes": {
            "plain-md-journal-2016-09": "a" * 64,
            "plain-md-notes-2018-rambling": "b" * 64,
            "plain-md-newthing": "9" * 64,
        }},
        {"overlay_hashes": {}},
    ],
)
def test_any_input_change_changes_hash(mutation: dict[str, object]) -> None:
    """Any change to any input component must produce a different hash."""
    base = _sample_inputs()
    baseline = compute_ingest_manifest(**base)  # type: ignore[arg-type]

    mutated = {**base, **mutation}
    result = compute_ingest_manifest(**mutated)  # type: ignore[arg-type]
    assert result.manifest_hash != baseline.manifest_hash


def test_canonical_json_has_no_insignificant_whitespace() -> None:
    """Canonical JSON uses compact separators — no ``, `` or ``: ``."""
    manifest = compute_ingest_manifest(**_sample_inputs())  # type: ignore[arg-type]
    raw = manifest.canonical_json
    assert b", " not in raw
    assert b": " not in raw
    assert not raw.endswith(b"\n")


def test_canonical_json_keys_sorted_top_level() -> None:
    """Top-level keys appear in sorted order in the canonical bytes."""
    manifest = compute_ingest_manifest(**_sample_inputs())  # type: ignore[arg-type]
    text = manifest.canonical_json.decode("utf-8")
    expected_order = [
        '"config_digest"',
        '"loader_versions"',
        '"overlay_hashes"',
        '"source_hashes"',
    ]
    positions = [text.index(k) for k in expected_order]
    assert positions == sorted(positions)


def test_write_manifest_sidecar_round_trip(tmp_path: Path) -> None:
    """Sidecar file contains exactly the canonical JSON bytes."""
    manifest = compute_ingest_manifest(**_sample_inputs())  # type: ignore[arg-type]
    sidecar = tmp_path / "some" / "nested" / "page.manifest.json"
    write_manifest_sidecar(manifest, sidecar)
    assert sidecar.read_bytes() == manifest.canonical_json


def test_write_manifest_sidecar_caller_picks_path(tmp_path: Path) -> None:
    """The function must not hard-code a ``vault/<page>.manifest.json``
    location — any caller-supplied path works."""
    manifest = compute_ingest_manifest(**_sample_inputs())  # type: ignore[arg-type]
    elsewhere = tmp_path / "anywhere-else.bin"
    write_manifest_sidecar(manifest, elsewhere)
    assert elsewhere.exists()
    assert elsewhere.read_bytes() == manifest.canonical_json


def test_manifest_is_frozen_dataclass() -> None:
    """IngestManifest is immutable — attempts to mutate raise."""
    manifest = compute_ingest_manifest(**_sample_inputs())  # type: ignore[arg-type]
    assert isinstance(manifest, IngestManifest)
    with pytest.raises(Exception):
        manifest.manifest_hash = "tampered"  # type: ignore[misc]
