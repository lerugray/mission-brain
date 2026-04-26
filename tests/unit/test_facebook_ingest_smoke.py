"""Smoke test — FB sources flow through ``ingest_source`` (rayb-016-b).

Uses the rayb-016-a synthetic fixtures plus a ``_StaticClient`` double
that returns a pre-built wiki-page markdown string. Confirms the
three facebook branches of the prompt builder plus the existing
pipeline produce well-formed :class:`WikiPage` instances whose
paragraphs each carry a citation and whose provenance is
``"primary"``.

Matches the pattern of ``test_music_ingest_smoke.py``.
"""

from __future__ import annotations

from pathlib import Path

from mission_brain.ingest.claude_cli_client import ClaudeCLIClient
from mission_brain.ingest.pipeline import ingest_source
from mission_brain.loaders.facebook import (
    FacebookGroupsLoader,
    FacebookMessagesLoader,
    FacebookPostsLoader,
)

FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "integration"
    / "fixtures"
    / "mini-corpus"
)


class _StaticClient(ClaudeCLIClient):
    def __init__(self, canned: str) -> None:
        self._canned = canned
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:  # type: ignore[override]
        self.calls.append(prompt)
        return self._canned


# ---------------------------------------------------------------------------
# facebook_posts


_POSTS_SID = "fb-posts-2013"
_CANNED_POSTS_MD = (
    "# Facebook Posts — 2013\n"
    "\n"
    "## Recurring themes\n"
    "\n"
    f"Ray posted about his band and touring. [ref:{_POSTS_SID}:para=1356998400]\n"
    "\n"
    "## Political commentary\n"
    "\n"
    f"No political posts this year. [ref:{_POSTS_SID}:para=1356998400]\n"
    "\n"
    "## Creative projects\n"
    "\n"
    f"References a photo series. [ref:{_POSTS_SID}:para=1367452800]\n"
    "\n"
    "## Personal events\n"
    "\n"
    f"Posted a linked article later. [ref:{_POSTS_SID}:para=1367452800]\n"
    "\n"
    "## Notable verbatim posts\n"
    "\n"
    "\"synthetic post in january 2013 — a verbose prose example "
    "used to cover the year-2013 bucket in the plain-text rendering path.\" "
    f"[ref:{_POSTS_SID}:para=1356998400]\n"
)


def test_fb_posts_smoke_ingests_to_valid_page() -> None:
    loader = FacebookPostsLoader()
    path = FIXTURES / "facebook" / "posts" / "_year_2013.agg"
    doc = loader.load(path)
    client = _StaticClient(_CANNED_POSTS_MD)
    page = ingest_source(doc, prior_wiki=None, client=client)

    assert page.title == "Facebook Posts — 2013"
    assert page.provenance == "primary"
    assert len(page.paragraphs) == 5
    assert all(len(p.citations) >= 1 for p in page.paragraphs)
    # At least one para= locator hit a real post timestamp from the fixture.
    anchors = {
        str(c.locator.anchor) for p in page.paragraphs for c in p.citations
        if getattr(c.locator, "anchor", None) is not None
    }
    assert "1356998400" in anchors


def test_fb_posts_prompt_dispatch_fires_for_facebook_branch() -> None:
    loader = FacebookPostsLoader()
    path = FIXTURES / "facebook" / "posts" / "_year_2013.agg"
    doc = loader.load(path)
    client = _StaticClient(_CANNED_POSTS_MD)
    ingest_source(doc, prior_wiki=None, client=client)

    prompt = client.calls[0]
    assert "TASK (facebook posts)" in prompt
    assert "## Recurring themes" in prompt
    assert "PRESERVE VERBATIM QUOTES" in prompt


# ---------------------------------------------------------------------------
# facebook_messages


_MSG_SID = "fb-msg-synthetic-above-123456"
_CANNED_MSG_MD = (
    "# Facebook Messages — Synthetic Friend\n"
    "\n"
    "## Relationship context\n"
    "\n"
    "Ray and Synthetic Friend exchange short updates. "
    f"[ref:{_MSG_SID}:para=1500000060000]\n"
    "\n"
    "## Topics discussed\n"
    "\n"
    "Small talk and a check-in on wellbeing. "
    f"[ref:{_MSG_SID}:para=1500000000000]\n"
    "\n"
    "## Tone and register\n"
    "\n"
    "Casual and brief. "
    f"[ref:{_MSG_SID}:para=1500000120000]\n"
    "\n"
    "## Notable verbatim exchanges\n"
    "\n"
    "\"doing well thanks\" followed by a long run of numbers. "
    f"[ref:{_MSG_SID}:para=1500000060000]\n"
)


def test_fb_messages_smoke_ingests_to_valid_page() -> None:
    loader = FacebookMessagesLoader(min_words=20)
    path = next(iter(loader.discover(FIXTURES)))
    doc = loader.load(path)
    client = _StaticClient(_CANNED_MSG_MD)
    page = ingest_source(doc, prior_wiki=None, client=client)

    assert page.provenance == "primary"
    assert len(page.paragraphs) == 4
    assert all(len(p.citations) >= 1 for p in page.paragraphs)


def test_fb_messages_prompt_dispatch_fires() -> None:
    loader = FacebookMessagesLoader(min_words=20)
    path = next(iter(loader.discover(FIXTURES)))
    doc = loader.load(path)
    client = _StaticClient(_CANNED_MSG_MD)
    ingest_source(doc, prior_wiki=None, client=client)

    prompt = client.calls[0]
    assert "TASK (facebook messages)" in prompt
    assert "## Relationship context" in prompt
    assert "## Notable verbatim exchanges" in prompt


# ---------------------------------------------------------------------------
# facebook_groups


_GROUP_SID = "fb-group-active-synthetic-group"
_CANNED_GROUP_MD = (
    "# Facebook Group — Active Synthetic Group\n"
    "\n"
    "## Group purpose\n"
    "\n"
    "A synthetic group Ray contributed to repeatedly. "
    f"[ref:{_GROUP_SID}:para=1400000000]\n"
    "\n"
    "## Ray's role\n"
    "\n"
    "Regular poster across 11 entries. "
    f"[ref:{_GROUP_SID}:para=1400100000]\n"
    "\n"
    "## Recurring contributions\n"
    "\n"
    "Short prose posts touching multiple topics. "
    f"[ref:{_GROUP_SID}:para=1400200000]\n"
    "\n"
    "## Notable verbatim posts\n"
    "\n"
    "\"synthetic first post in the Active Group — prose body one.\" "
    f"[ref:{_GROUP_SID}:para=1400000000]\n"
)


def test_fb_groups_smoke_ingests_to_valid_page() -> None:
    loader = FacebookGroupsLoader(min_contribs=5)
    path = next(iter(loader.discover(FIXTURES)))
    doc = loader.load(path)
    client = _StaticClient(_CANNED_GROUP_MD)
    page = ingest_source(doc, prior_wiki=None, client=client)

    assert page.provenance == "primary"
    assert len(page.paragraphs) == 4
    assert all(len(p.citations) >= 1 for p in page.paragraphs)


def test_fb_groups_prompt_dispatch_fires() -> None:
    loader = FacebookGroupsLoader(min_contribs=5)
    path = next(iter(loader.discover(FIXTURES)))
    doc = loader.load(path)
    client = _StaticClient(_CANNED_GROUP_MD)
    ingest_source(doc, prior_wiki=None, client=client)

    prompt = client.calls[0]
    assert "TASK (facebook group)" in prompt
    assert "## Group purpose" in prompt
    assert "## Ray's role" in prompt
