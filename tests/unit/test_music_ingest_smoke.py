"""Smoke test — music source flows through ``ingest_source`` (rayb-014-c).

Uses the rayb-013-c synthetic fixtures plus a ``_StaticClient`` double
that returns a pre-built music wiki-page markdown string. Confirms the
music branch of the prompt builder plus the existing pipeline
produces a well-formed :class:`WikiPage` whose paragraphs each carry
a citation and whose provenance is ``"primary"``.

Tests that require a real ``claude -p`` invocation are marked
``pytest.mark.skip(reason='integration — runs with real LLM')`` so CI
stays green without an LLM in the loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_brain.ingest.claude_cli_client import ClaudeCLIClient
from mission_brain.ingest.pipeline import ingest_source
from mission_brain.loaders.music_data import MusicDataLoader

FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "integration"
    / "fixtures"
    / "mini-corpus"
    / "music"
)


class _StaticClient(ClaudeCLIClient):
    """Return canned markdown instead of invoking the ``claude`` CLI."""

    def __init__(self, canned_markdown: str) -> None:
        self._canned = canned_markdown
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:  # type: ignore[override]
        self.calls.append(prompt)
        return self._canned


# The synthetic per-song fixture resolves to this source_id under the
# fixtures/mini-corpus/ root. The loader derives it from the path by
# default — here we set it explicitly on the returned SourceDocument
# so the canned markdown can embed a literal marker.
_PER_SONG_SID = "fake-song"


_CANNED_PER_SONG_MARKDOWN = (
    "# Synthetic Fake Song\n"
    "\n"
    "## Tempo and meter\n"
    "\n"
    f"Tempo is 120 BPM in 4/4. [ref:{_PER_SONG_SID}:time=00:00:00.000-00:00:00.500]\n"
    "\n"
    "## Harmonic language\n"
    "\n"
    f"Opens on a C4 pitched melodic line. [ref:{_PER_SONG_SID}:time=00:00:00.000-00:00:00.500]\n"
    "\n"
    "## Rhythm and groove\n"
    "\n"
    f"Steady 8th-note subdivision. [ref:{_PER_SONG_SID}:time=00:00:00.500-00:00:01.000]\n"
    "\n"
    "## Structural arc\n"
    "\n"
    f"Single short phrase, no repetition. [ref:{_PER_SONG_SID}:time=00:00:00.000-00:00:01.250]\n"
    "\n"
    "## Notable motifs\n"
    "\n"
    f"Ascending major-third C4 to E4. [ref:{_PER_SONG_SID}:time=00:00:00.000-00:00:01.000]\n"
)


_CANNED_CATALOG_MARKDOWN = (
    "# Music Catalog Aggregate\n"
    "\n"
    "## Tempo and meter\n"
    "\n"
    "Aggregate does not expose tempo. [ref:music-catalog-analysis:para=catalog_stats]\n"
    "\n"
    "## Harmonic language\n"
    "\n"
    "C major dominates the scale fingerprint. [ref:music-catalog-analysis:para=scale_fingerprint]\n"
    "\n"
    "## Rhythm and groove\n"
    "\n"
    "8th-note durations are the largest bucket. "
    "[ref:music-catalog-analysis:para=rhythm_distribution]\n"
    "\n"
    "## Structural arc\n"
    "\n"
    "Catalog aggregate has no structural axis. [ref:music-catalog-analysis:para=catalog_stats]\n"
    "\n"
    "## Notable motifs\n"
    "\n"
    "The octave-chromatic (-12,12,-12) motif recurs most. "
    "[ref:music-catalog-analysis:para=top_motifs]\n"
)


def _load_per_song():
    path = FIXTURES / "per-song" / "fake-song.json"
    doc = MusicDataLoader().load(path)
    # Pin the source_id so the canned markdown's literal matches.
    doc.source_id = _PER_SONG_SID
    return doc


def _load_catalog():
    path = FIXTURES / "catalog" / "catalog_analysis.json"
    doc = MusicDataLoader().load(path)
    doc.source_id = "music-catalog-analysis"
    return doc


def test_per_song_ingests_to_valid_wiki_page() -> None:
    doc = _load_per_song()
    client = _StaticClient(_CANNED_PER_SONG_MARKDOWN)
    page = ingest_source(doc, prior_wiki=None, client=client)

    assert page.title == "Synthetic Fake Song"
    assert page.provenance == "primary"
    assert len(page.paragraphs) == 5
    assert all(len(p.citations) >= 1 for p in page.paragraphs)
    # Every citation points at the per-song source id.
    for para in page.paragraphs:
        for cite in para.citations:
            assert str(cite.source_id) == _PER_SONG_SID


def test_per_song_prompt_dispatch_is_music_branch() -> None:
    doc = _load_per_song()
    client = _StaticClient(_CANNED_PER_SONG_MARKDOWN)
    ingest_source(doc, prior_wiki=None, client=client)
    # The music branch emits the five section headings and time=
    # locator grammar; confirming the dispatch fired.
    prompt = client.calls[0]
    assert "## Tempo and meter" in prompt
    assert "## Notable motifs" in prompt
    assert "time=HH:MM:SS.fff-HH:MM:SS.fff" in prompt


def test_catalog_ingests_to_valid_wiki_page() -> None:
    doc = _load_catalog()
    client = _StaticClient(_CANNED_CATALOG_MARKDOWN)
    page = ingest_source(doc, prior_wiki=None, client=client)

    assert page.title == "Music Catalog Aggregate"
    assert page.provenance == "primary"
    assert len(page.paragraphs) == 5
    assert all(len(p.citations) >= 1 for p in page.paragraphs)


def test_catalog_prompt_dispatch_uses_para_locators() -> None:
    doc = _load_catalog()
    client = _StaticClient(_CANNED_CATALOG_MARKDOWN)
    ingest_source(doc, prior_wiki=None, client=client)
    prompt = client.calls[0]
    assert "para=<key>" in prompt
    assert "scale_fingerprint" in prompt


@pytest.mark.skip(reason="integration — runs with real LLM")
def test_per_song_ingest_with_real_claude_cli() -> None:
    """Real end-to-end with the actual ``claude`` CLI client.

    Skipped in CI — kicks off a live model call. Enable locally by
    removing the skip marker and pointing MISSION_BRAIN_INGEST_MODEL at a
    model the machine can serve.
    """
    from mission_brain.ingest.claude_cli_client import ClaudeCLIClient as RealClient

    doc = _load_per_song()
    page = ingest_source(doc, prior_wiki=None, client=RealClient())
    assert page.provenance == "primary"
    assert len(page.paragraphs) >= 1
