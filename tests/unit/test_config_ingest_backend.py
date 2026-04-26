"""Tests for :attr:`mission_brain.config.Settings.ingest_backend`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mission_brain.config import Settings


def test_ingest_backend_defaults_to_claude_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("MISSION_BRAIN_INGEST_BACKEND", raising=False)
    assert Settings().ingest_backend == "claude"


def test_ingest_backend_reads_env_cursor(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_INGEST_BACKEND", "cursor")
    assert Settings().ingest_backend == "cursor"


def test_ingest_backend_invalid_raises(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_INGEST_BACKEND", "bogus")
    with pytest.raises((ValueError, ValidationError)):
        Settings()


def test_ingest_backend_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_INGEST_BACKEND", "CURSOR")
    assert Settings().ingest_backend == "cursor"
