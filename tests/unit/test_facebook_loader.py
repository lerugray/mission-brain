"""Tests for :mod:`mission_brain.loaders.facebook` (rayb-016-a).

Covers the three loaders' ``discover`` / ``load`` / ``content_hash``
contracts over the synthetic mini-corpus fixture, plus threshold gating,
determinism, and env-var override.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mission_brain.loaders import (
    FacebookGroupsLoader,
    FacebookMessagesLoader,
    FacebookPostsLoader,
    SourceDocument,
)
from mission_brain.loaders.facebook import (
    DEFAULT_GROUP_MIN_CONTRIBS,
    DEFAULT_MSG_MIN_WORDS,
    _extract_group_name,
    _extract_post_prose,
    _fix_mojibake,
    _slugify,
)

FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "integration"
    / "fixtures"
    / "mini-corpus"
)


# ---------------------------------------------------------------------------
# Helpers


def _posts_loader() -> FacebookPostsLoader:
    return FacebookPostsLoader()


def _messages_loader(min_words: int = 20) -> FacebookMessagesLoader:
    return FacebookMessagesLoader(min_words=min_words)


def _groups_loader(min_contribs: int = 5) -> FacebookGroupsLoader:
    return FacebookGroupsLoader(min_contribs=min_contribs)


# ---------------------------------------------------------------------------
# Posts loader


def test_posts_discover_returns_one_path_per_year_sorted() -> None:
    loader = _posts_loader()
    found = list(loader.discover(FIXTURES))
    # Fixture has posts in 2013 (3 entries) and 2014 (2 entries).
    assert [p.name for p in found] == ["_year_2013.agg", "_year_2014.agg"]


def test_posts_discover_empty_for_missing_dir(tmp_path: Path) -> None:
    loader = _posts_loader()
    assert list(loader.discover(tmp_path)) == []


def test_posts_load_2013_shape() -> None:
    loader = _posts_loader()
    path = FIXTURES / "facebook" / "posts" / "_year_2013.agg"
    doc = loader.load(path)
    assert isinstance(doc, SourceDocument)
    assert doc.source_type == "facebook_posts"
    assert doc.source_id == "fb-posts-2013"
    assert doc.provenance == "primary"
    assert doc.title == "Facebook Posts — 2013"
    assert doc.frontmatter["year"] == 2013
    # 3 entries in 2013: 1 prose, 1 photo+caption, 1 photo no caption.
    assert doc.frontmatter["post_count_total"] == 3
    assert doc.frontmatter["post_count_rendered"] == 2
    # The photo-no-caption entry is recorded as skipped.
    assert len(doc.frontmatter["post_ids_skipped"]) == 1


def test_posts_load_2013_body_has_month_headings_and_ids() -> None:
    loader = _posts_loader()
    doc = loader.load(FIXTURES / "facebook" / "posts" / "_year_2013.agg")
    assert "## January 2013" in doc.body
    assert "## May 2013" in doc.body
    # ID lines reference stable per-post anchors for citations.
    assert "[post_id=1356998400" in doc.body
    # Verbatim prose from the fixture is preserved — no paraphrase.
    assert "synthetic post in january 2013" in doc.body


def test_posts_load_2014_includes_external_link_name() -> None:
    loader = _posts_loader()
    doc = loader.load(FIXTURES / "facebook" / "posts" / "_year_2014.agg")
    assert "Linked article title in july 2014" in doc.body
    assert "synthetic link share in july 2014 with prose body." in doc.body


def test_posts_load_notes_sorted_by_timestamp() -> None:
    loader = _posts_loader()
    doc = loader.load(FIXTURES / "facebook" / "posts" / "_year_2013.agg")
    # Jan post precedes May post precedes Sep post in body order.
    jan = doc.body.find("[post_id=1356998400")
    may = doc.body.find("[post_id=1367452800")
    sep = doc.body.find("1380585600")  # skipped but id may not appear
    assert jan < may, doc.body
    # September entry is skipped (no prose) → not in body, but Jan < May holds.
    assert sep == -1 or jan < sep


def test_posts_load_rejects_non_year_agg_path(tmp_path: Path) -> None:
    loader = _posts_loader()
    with pytest.raises(ValueError, match="_year_"):
        loader.load(tmp_path / "not-a-year.agg")


def test_posts_content_hash_stable_and_sensitive(tmp_path: Path) -> None:
    loader = _posts_loader()
    doc_a = loader.load(FIXTURES / "facebook" / "posts" / "_year_2013.agg")
    doc_b = loader.load(FIXTURES / "facebook" / "posts" / "_year_2013.agg")
    assert doc_a.content_hash == doc_b.content_hash

    # Perturb a post file → hash changes.
    scratch = tmp_path / "facebook" / "posts"
    scratch.mkdir(parents=True)
    original = FIXTURES / "facebook" / "posts" / "your_posts__check_ins__photos_and_videos_1.json"
    (scratch / original.name).write_bytes(original.read_bytes() + b" ")
    perturbed = loader.load(scratch / "_year_2013.agg")
    assert perturbed.content_hash != doc_a.content_hash


def test_posts_body_rendering_is_deterministic() -> None:
    loader = _posts_loader()
    path = FIXTURES / "facebook" / "posts" / "_year_2013.agg"
    bodies = {loader.load(path).body for _ in range(20)}
    assert len(bodies) == 1


def test_posts_locator_for_raises() -> None:
    loader = _posts_loader()
    with pytest.raises(NotImplementedError):
        loader.locator_for((0, 1))


# ---------------------------------------------------------------------------
# Messages loader


def test_messages_discover_respects_threshold() -> None:
    loader = _messages_loader(min_words=20)
    found = list(loader.discover(FIXTURES))
    # Only the above-threshold thread qualifies.
    assert len(found) == 1
    assert found[0].parent.name == "synthetic-above_123456"
    assert found[0].name == "_thread.agg"


def test_messages_discover_empty_when_threshold_too_high() -> None:
    loader = _messages_loader(min_words=100_000)
    assert list(loader.discover(FIXTURES)) == []


def test_messages_load_shape() -> None:
    loader = _messages_loader(min_words=20)
    thread = (
        FIXTURES / "facebook" / "messages" / "inbox"
        / "synthetic-above_123456" / "_thread.agg"
    )
    doc = loader.load(thread)
    assert doc.source_type == "facebook_messages"
    assert doc.source_id == "fb-msg-synthetic-above-123456"
    assert doc.provenance == "primary"
    assert "Synthetic Friend" in doc.title
    assert doc.frontmatter["message_count"] == 4
    assert doc.frontmatter["ray_words"] >= 20
    assert "Ray Weiss" in doc.frontmatter["participants"]
    assert "Synthetic Friend" in doc.frontmatter["participants"]


def test_messages_body_tags_ray_and_other_messages() -> None:
    loader = _messages_loader(min_words=20)
    thread = (
        FIXTURES / "facebook" / "messages" / "inbox"
        / "synthetic-above_123456" / "_thread.agg"
    )
    doc = loader.load(thread)
    # First message (t=1500000000000) is from Synthetic Friend.
    first_idx = doc.body.find("[msg_id=1500000000000")
    assert first_idx != -1
    assert "[them]" in doc.body[first_idx : first_idx + 80]
    # Second message is Ray's.
    second_idx = doc.body.find("[msg_id=1500000060000")
    assert second_idx != -1
    assert "[Ray]" in doc.body[second_idx : second_idx + 80]


def test_messages_body_is_chronological() -> None:
    loader = _messages_loader(min_words=20)
    thread = (
        FIXTURES / "facebook" / "messages" / "inbox"
        / "synthetic-above_123456" / "_thread.agg"
    )
    doc = loader.load(thread)
    stamps = [
        int(line.split("msg_id=")[1].split(" ")[0].rstrip("]"))
        for line in doc.body.splitlines()
        if "[msg_id=" in line
    ]
    assert stamps == sorted(stamps)


def test_messages_merges_multiple_message_files(tmp_path: Path) -> None:
    """Two message_*.json files in one thread merge into a single sorted body."""
    thread_dir = tmp_path / "facebook" / "messages" / "inbox" / "multi_42"
    thread_dir.mkdir(parents=True)
    file1 = {
        "participants": [{"name": "Ray Weiss"}, {"name": "Other"}],
        "messages": [
            {
                "sender_name": "Ray Weiss",
                "timestamp_ms": 2000000000000,
                "content": "later message " + " ".join(str(i) for i in range(30)),
            },
        ],
        "title": "Other",
        "thread_path": "inbox/multi_42",
    }
    file2 = {
        "participants": [{"name": "Ray Weiss"}, {"name": "Other"}],
        "messages": [
            {
                "sender_name": "Ray Weiss",
                "timestamp_ms": 1000000000000,
                "content": "earlier message " + " ".join(str(i) for i in range(30)),
            },
        ],
        "title": "Other",
        "thread_path": "inbox/multi_42",
    }
    (thread_dir / "message_1.json").write_text(json.dumps(file1), encoding="utf-8")
    (thread_dir / "message_2.json").write_text(json.dumps(file2), encoding="utf-8")

    loader = FacebookMessagesLoader(min_words=20)
    paths = list(loader.discover(tmp_path))
    assert len(paths) == 1
    doc = loader.load(paths[0])
    assert doc.frontmatter["message_count"] == 2
    # Earlier timestamp appears before later in body.
    earlier = doc.body.find("1000000000000")
    later = doc.body.find("2000000000000")
    assert earlier != -1 and later != -1 and earlier < later


def test_messages_threshold_env_var_override(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_FB_MSG_MIN_WORDS", "5")
    loader = FacebookMessagesLoader()
    # At min_words=5 both threads qualify (even the "below" thread has
    # "thanks" from Ray which is 1 word — still below — but the
    # "above" thread easily qualifies). We verify the loader read the
    # env var by checking its internal field.
    assert loader._min_words == 5


def test_messages_default_threshold_is_5000() -> None:
    assert DEFAULT_MSG_MIN_WORDS == 5000
    loader = FacebookMessagesLoader()
    # No env var + no kwarg → default.
    if "MISSION_BRAIN_FB_MSG_MIN_WORDS" not in os.environ:
        assert loader._min_words == 5000


def test_messages_locator_for_raises() -> None:
    loader = _messages_loader()
    with pytest.raises(NotImplementedError):
        loader.locator_for((0, 1))


# ---------------------------------------------------------------------------
# Groups loader


def test_groups_discover_respects_threshold() -> None:
    loader = _groups_loader(min_contribs=5)
    found = list(loader.discover(FIXTURES))
    # Active group has 11 entries → qualifies at min_contribs=5.
    # Sparse group has 2 entries → does not qualify.
    assert len(found) == 1
    assert found[0].name == "_group_active-synthetic-group.agg"


def test_groups_discover_empty_when_threshold_too_high() -> None:
    loader = _groups_loader(min_contribs=50)
    assert list(loader.discover(FIXTURES)) == []


def test_groups_load_shape() -> None:
    loader = _groups_loader(min_contribs=5)
    path = FIXTURES / "facebook" / "groups" / "_group_active-synthetic-group.agg"
    doc = loader.load(path)
    assert doc.source_type == "facebook_groups"
    assert doc.source_id == "fb-group-active-synthetic-group"
    assert doc.provenance == "primary"
    assert doc.frontmatter["group_name"] == "Active Synthetic Group"
    assert doc.frontmatter["contribution_count"] == 11
    # 1 empty-prose entry is skipped.
    assert doc.frontmatter["contribution_count_rendered"] == 10
    assert len(doc.frontmatter["post_ids_skipped"]) == 1


def test_groups_body_contains_year_headings_and_prose() -> None:
    loader = _groups_loader(min_contribs=5)
    path = FIXTURES / "facebook" / "groups" / "_group_active-synthetic-group.agg"
    doc = loader.load(path)
    assert "## 2014" in doc.body
    assert "synthetic first post in the Active Group" in doc.body


def test_groups_threshold_env_var_override(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_BRAIN_FB_GROUP_MIN_CONTRIBS", "1")
    loader = FacebookGroupsLoader()
    # With threshold 1 both groups qualify.
    found = list(loader.discover(FIXTURES))
    assert len(found) == 2


def test_groups_default_threshold_is_20() -> None:
    assert DEFAULT_GROUP_MIN_CONTRIBS == 20


def test_groups_content_hash_deterministic() -> None:
    loader = _groups_loader(min_contribs=5)
    path = FIXTURES / "facebook" / "groups" / "_group_active-synthetic-group.agg"
    hashes = {loader.load(path).content_hash for _ in range(10)}
    assert len(hashes) == 1


def test_groups_locator_for_raises() -> None:
    loader = _groups_loader()
    with pytest.raises(NotImplementedError):
        loader.locator_for((0, 1))


# ---------------------------------------------------------------------------
# Loader identity + helpers


def test_loader_identity_fields() -> None:
    assert FacebookPostsLoader.source_type == "facebook_posts"
    assert FacebookMessagesLoader.source_type == "facebook_messages"
    assert FacebookGroupsLoader.source_type == "facebook_groups"
    for cls in (FacebookPostsLoader, FacebookMessagesLoader, FacebookGroupsLoader):
        assert ".json" in cls.supported_extensions
    # Messages loader bumped to 1.1 when chunking landed (rayb-018-a);
    # others stay on 1.0.
    assert FacebookPostsLoader.version == "1.0"
    assert FacebookMessagesLoader.version == "1.1"
    assert FacebookGroupsLoader.version == "1.0"


def test_slugify_strips_non_alnum() -> None:
    assert _slugify("Active Synthetic Group") == "active-synthetic-group"
    assert _slugify("  Hey! 100% cool.  ") == "hey-100-cool"
    assert _slugify("") == "unknown"
    # Unicode → collapses via lowercasing + non-alnum scrub.
    assert _slugify("Café Société") != ""


def test_extract_group_name_variations() -> None:
    assert _extract_group_name("Ray Weiss posted in Foo Bar.") == "Foo Bar"
    assert (
        _extract_group_name("Ray Weiss commented on Jane's post in Baz Qux.")
        == "Baz Qux"
    )
    assert _extract_group_name("Ray Weiss shared a link.") is None
    assert _extract_group_name("") is None
    assert _extract_group_name(None) is None  # type: ignore[arg-type]


def test_extract_post_prose_skips_auto_titles() -> None:
    entry = {
        "title": "Ray Weiss added a new photo.",
        "data": [{"post": "real body"}],
        "attachments": [],
    }
    assert _extract_post_prose(entry) == "real body"


def test_extract_post_prose_handles_empty() -> None:
    assert _extract_post_prose({"title": "Ray Weiss", "data": [{}], "attachments": []}) == ""


def test_fix_mojibake_roundtrips_utf8() -> None:
    # A cleanly UTF-8 string survives when re-encoded via the latin-1 path only
    # if the source was FB's double-encoded form. Pure ASCII is unchanged.
    assert _fix_mojibake("plain ascii") == "plain ascii"
    # Test with a known double-encoded sequence: 'é' is U+00E9, encoded in
    # UTF-8 as 0xC3 0xA9. FB would export those as latin-1 giving "Ã©".
    assert _fix_mojibake("caf\u00c3\u00a9") == "café"
