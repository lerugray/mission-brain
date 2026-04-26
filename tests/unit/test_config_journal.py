"""Tests for rb-035: journal config + journal-private provenance value."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mission_brain.config import Settings
from mission_brain.schema import Citation, LineRangeLocator, Paragraph, SourceId, WikiPage


# ---------------------------------------------------------------------------
# Provenance enum extension


def test_provenance_accepts_journal_private() -> None:
    page = WikiPage(
        title="Journal Reflection",
        source_ids=[SourceId("journal-2026-04-26")],
        paragraphs=[
            Paragraph(
                text="Today I noticed X.",
                citations=[
                    Citation(
                        source_id=SourceId("journal-2026-04-26"),
                        locator=LineRangeLocator(start=1, end=2),
                        excerpt="x",
                    ),
                ],
            ),
        ],
        provenance="journal-private",
    )
    assert page.provenance == "journal-private"


def test_provenance_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        WikiPage(
            title="X",
            source_ids=[SourceId("src")],
            paragraphs=[],
            provenance="bogus",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# journal_enabled (default ON per Ray's 2026-04-25 resolution)


def test_journal_enabled_defaults_true(monkeypatch) -> None:
    monkeypatch.delenv("MISSION_BRAIN_JOURNAL_ENABLED", raising=False)
    assert Settings().journal_enabled is True


def test_journal_enabled_env_false(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_JOURNAL_ENABLED", "false")
    assert Settings().journal_enabled is False


def test_journal_enabled_env_zero(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_JOURNAL_ENABLED", "0")
    assert Settings().journal_enabled is False


def test_journal_enabled_env_truthy_variants(monkeypatch) -> None:
    for raw in ("1", "true", "yes", "on", "TRUE", "Yes"):
        monkeypatch.setenv("MISSION_BRAIN_JOURNAL_ENABLED", raw)
        assert Settings().journal_enabled is True, f"failed for {raw!r}"


# ---------------------------------------------------------------------------
# journal_provider


def test_journal_provider_defaults_anthropic(monkeypatch) -> None:
    monkeypatch.delenv("MISSION_BRAIN_JOURNAL_PROVIDER", raising=False)
    assert Settings().journal_provider == "anthropic"


def test_journal_provider_env_ollama(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_JOURNAL_PROVIDER", "ollama")
    assert Settings().journal_provider == "ollama"


def test_journal_provider_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_JOURNAL_PROVIDER", "OLLAMA")
    assert Settings().journal_provider == "ollama"


def test_journal_provider_invalid_raises(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_JOURNAL_PROVIDER", "bogus")
    with pytest.raises((ValueError, ValidationError)):
        Settings()


# ---------------------------------------------------------------------------
# journal_allow_cloud (default ON; flip to false for Ollama-only mode)


def test_journal_allow_cloud_defaults_true(monkeypatch) -> None:
    monkeypatch.delenv("MISSION_BRAIN_JOURNAL_ALLOW_CLOUD", raising=False)
    assert Settings().journal_allow_cloud is True


def test_journal_allow_cloud_env_false(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_JOURNAL_ALLOW_CLOUD", "false")
    assert Settings().journal_allow_cloud is False
