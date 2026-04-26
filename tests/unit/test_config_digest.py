"""Tests for :func:`mission_brain.config.compute_config_digest`.

The digest is part of the CLAUDE.md §2.2 ingest manifest: it must
change iff a synthesis-relevant config field changes, and must be
stable across processes (no randomness, no floats, no dict-order
dependence).
"""

from __future__ import annotations

from pathlib import Path

from mission_brain.config import Settings, compute_config_digest


def _base(tmp_path: Path) -> Settings:
    return Settings(
        corpus_root=tmp_path / "corpus",
        vault_root=tmp_path / "vault",
        ingest_model="claude-sonnet-4-6",
        embed_model="voyage-3-lite",
        fb_msg_min_words=5000,
        fb_group_min_contribs=20,
    )


def test_digest_is_deterministic_across_instances(tmp_path: Path) -> None:
    assert compute_config_digest(_base(tmp_path)) == compute_config_digest(
        _base(tmp_path)
    )


def test_ingest_model_change_shifts_digest(tmp_path: Path) -> None:
    a = compute_config_digest(_base(tmp_path))
    s = _base(tmp_path)
    s.ingest_model = "claude-opus-4-7"
    assert compute_config_digest(s) != a


def test_embed_model_change_shifts_digest(tmp_path: Path) -> None:
    a = compute_config_digest(_base(tmp_path))
    s = _base(tmp_path)
    s.embed_model = "voyage-3-large"
    assert compute_config_digest(s) != a


def test_fb_threshold_change_shifts_digest(tmp_path: Path) -> None:
    a = compute_config_digest(_base(tmp_path))
    s = _base(tmp_path)
    s.fb_msg_min_words = 100
    assert compute_config_digest(s) != a
    s = _base(tmp_path)
    s.fb_group_min_contribs = 1
    assert compute_config_digest(s) != a


def test_unrelated_field_change_does_not_shift_digest(tmp_path: Path) -> None:
    """An API-key swap shouldn't force wholesale cache invalidation."""
    a = compute_config_digest(_base(tmp_path))
    s = _base(tmp_path)
    s.voyage_api_key = "secret-swapped"
    s.claude_cli_bin = "/elsewhere/claude"
    s.ollama_host = "http://other-host:11434"
    assert compute_config_digest(s) == a
