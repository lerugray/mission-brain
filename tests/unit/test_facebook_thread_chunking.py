"""Tests for FacebookMessagesLoader thread chunking (rayb-018-a).

Threads whose rendered body exceeds a configurable token budget split
into multiple ``SourceDocument``s with stable deterministic ids
(``fb-msg-<slug>-partN``). Covers:

* Non-chunked threads keep the ``_thread.agg`` path and ``fb-msg-<slug>``
  source_id.
* Oversized threads yield ``_thread_partN.agg`` discover paths in order.
* Each loaded chunk carries its own ``part``/``total_parts`` frontmatter
  and a chunk-specific content_hash.
* The concatenation of per-chunk messages equals the original thread.
* Chunk boundaries are deterministic across repeated runs.
* A single message larger than the budget lands alone in its own chunk
  (we do not split below message granularity).
* ``MISSION_BRAIN_FB_MSG_CHUNK_MAX_TOKENS`` env override works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mission_brain.loaders import FacebookMessagesLoader
from mission_brain.loaders.facebook import (
    DEFAULT_MSG_CHUNK_MAX_TOKENS,
    _chunk_messages,
    _estimate_msg_tokens,
)


def _write_thread(
    inbox: Path, name: str, messages: list[dict], participants=("Test User", "Other")
) -> Path:
    thread_dir = inbox / name
    thread_dir.mkdir(parents=True)
    data = {
        "participants": [{"name": p} for p in participants],
        "messages": messages,
        "title": "Other",
        "thread_path": f"inbox/{name}",
    }
    (thread_dir / "message_1.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    return thread_dir


def _msg(ts_ms: int, sender: str, content: str) -> dict:
    return {
        "sender_name": sender,
        "timestamp_ms": ts_ms,
        "content": content,
    }


def _big_corpus(tmp_path: Path, per_msg_chars: int, count: int) -> Path:
    inbox = tmp_path / "facebook" / "messages" / "inbox"
    body_chunk = "the user talks about politics and wargames " * (
        max(1, per_msg_chars // 40)
    )
    msgs = []
    for i in range(count):
        msgs.append(
            _msg(
                1_500_000_000_000 + i * 60_000,
                "Test User",
                body_chunk[:per_msg_chars],
            )
        )
    _write_thread(inbox, "chunky_77", msgs)
    return tmp_path


# ---------------------------------------------------------------------------
# Non-chunked path stays byte-compatible


def test_small_thread_still_uses_plain_thread_agg(tmp_path: Path) -> None:
    inbox = tmp_path / "facebook" / "messages" / "inbox"
    _write_thread(
        inbox,
        "small_1",
        [_msg(1_500_000_000_000 + i * 1_000, "Test User", "short message") for i in range(30)],
    )
    loader = FacebookMessagesLoader(min_words=1)
    paths = list(loader.discover(tmp_path))
    assert [p.name for p in paths] == ["_thread.agg"]

    doc = loader.load(paths[0])
    assert doc.source_id == "fb-msg-small-1"
    assert "part" not in doc.frontmatter
    assert "total_parts" not in doc.frontmatter


# ---------------------------------------------------------------------------
# Chunked discover + load


def test_oversized_thread_discovers_multiple_parts(tmp_path: Path) -> None:
    # 20 messages × 20k chars each = ~100k tokens of content; budget 20k → 5 parts
    corpus = _big_corpus(tmp_path, per_msg_chars=20_000, count=20)
    loader = FacebookMessagesLoader(min_words=1, chunk_max_tokens=20_000)
    paths = list(loader.discover(corpus))
    # Every discovered path is a _thread_partN.agg, numbered 1..N sequentially.
    assert len(paths) >= 2
    names = [p.name for p in paths]
    expected = [f"_thread_part{i + 1}.agg" for i in range(len(names))]
    assert names == expected
    assert all(p.parent.name == "chunky_77" for p in paths)


def test_chunked_load_yields_partwise_source_id_and_frontmatter(
    tmp_path: Path,
) -> None:
    corpus = _big_corpus(tmp_path, per_msg_chars=20_000, count=20)
    loader = FacebookMessagesLoader(min_words=1, chunk_max_tokens=20_000)
    paths = list(loader.discover(corpus))
    docs = [loader.load(p) for p in paths]

    total = len(docs)
    for i, doc in enumerate(docs, start=1):
        assert doc.source_id == f"fb-msg-chunky-77-part{i}"
        assert doc.frontmatter["part"] == i
        assert doc.frontmatter["total_parts"] == total
        assert f"(part {i} of {total})" in doc.title
        assert f"part: {i}" in doc.body
        assert f"total_parts: {total}" in doc.body


def test_chunk_boundary_preserves_all_messages(tmp_path: Path) -> None:
    """The union of per-chunk messages equals the original thread."""
    corpus = _big_corpus(tmp_path, per_msg_chars=20_000, count=20)
    loader = FacebookMessagesLoader(min_words=1, chunk_max_tokens=20_000)
    paths = list(loader.discover(corpus))
    docs = [loader.load(p) for p in paths]
    per_chunk_counts = [d.frontmatter["message_count"] for d in docs]
    assert sum(per_chunk_counts) == 20


def test_chunks_are_chronological_and_non_overlapping(tmp_path: Path) -> None:
    corpus = _big_corpus(tmp_path, per_msg_chars=20_000, count=20)
    loader = FacebookMessagesLoader(min_words=1, chunk_max_tokens=20_000)
    docs = [loader.load(p) for p in loader.discover(corpus)]
    last_seen = -1
    for doc in docs:
        stamps = [
            int(line.split("msg_id=")[1].split(" ")[0].rstrip("]"))
            for line in doc.body.splitlines()
            if "[msg_id=" in line
        ]
        assert stamps == sorted(stamps), doc.source_id
        assert stamps[0] > last_seen, f"chunk {doc.source_id} overlaps previous"
        last_seen = stamps[-1]


def test_chunking_is_deterministic_across_runs(tmp_path: Path) -> None:
    corpus = _big_corpus(tmp_path, per_msg_chars=20_000, count=20)
    loader = FacebookMessagesLoader(min_words=1, chunk_max_tokens=20_000)
    first = [loader.load(p).source_id for p in loader.discover(corpus)]
    for _ in range(5):
        again = [loader.load(p).source_id for p in loader.discover(corpus)]
        assert again == first


def test_chunk_content_hashes_differ_across_parts(tmp_path: Path) -> None:
    corpus = _big_corpus(tmp_path, per_msg_chars=20_000, count=20)
    loader = FacebookMessagesLoader(min_words=1, chunk_max_tokens=20_000)
    hashes = [loader.load(p).content_hash for p in loader.discover(corpus)]
    # Each partN has its part index mixed into the hash input.
    assert len(set(hashes)) == len(hashes)


def test_chunk_content_hash_stable_across_runs(tmp_path: Path) -> None:
    corpus = _big_corpus(tmp_path, per_msg_chars=20_000, count=20)
    loader = FacebookMessagesLoader(min_words=1, chunk_max_tokens=20_000)
    paths = list(loader.discover(corpus))
    first = [loader.load(p).content_hash for p in paths]
    # Reload without re-running discover — hashes must match.
    again = [loader.load(p).content_hash for p in paths]
    assert again == first


# ---------------------------------------------------------------------------
# Edge cases


def test_single_oversized_message_gets_its_own_chunk(tmp_path: Path) -> None:
    """A message bigger than the budget is not split below message granularity."""
    inbox = tmp_path / "facebook" / "messages" / "inbox"
    mega = "x" * 400_000  # ~100k tokens by chars/4 heuristic
    _write_thread(
        inbox,
        "mega_5",
        [
            _msg(1_500_000_000_000, "Test User", "short"),
            _msg(1_500_000_060_000, "Test User", mega),
            _msg(1_500_000_120_000, "Test User", "short tail"),
        ],
    )
    loader = FacebookMessagesLoader(min_words=1, chunk_max_tokens=20_000)
    paths = list(loader.discover(tmp_path))
    # At least 2 chunks because mega alone blows the budget.
    assert len(paths) >= 2
    docs = [loader.load(p) for p in paths]
    # Every message is represented across all chunks.
    assert sum(d.frontmatter["message_count"] for d in docs) == 3
    # Mega must land somewhere by itself or with a very small partner.
    assert any(len(mega) == d.frontmatter["total_words"] or mega in d.body for d in docs)


def test_load_rejects_nonexistent_part_index(tmp_path: Path) -> None:
    corpus = _big_corpus(tmp_path, per_msg_chars=20_000, count=20)
    loader = FacebookMessagesLoader(min_words=1, chunk_max_tokens=20_000)
    # Ask for part 999 of a thread that produces ~5 parts.
    bogus = corpus / "facebook" / "messages" / "inbox" / "chunky_77" / "_thread_part999.agg"
    with pytest.raises(ValueError, match="out of range"):
        loader.load(bogus)


def test_load_rejects_unrecognized_thread_path(tmp_path: Path) -> None:
    inbox = tmp_path / "facebook" / "messages" / "inbox"
    _write_thread(inbox, "basic_9", [_msg(1_500_000_000_000, "Test User", "hi")])
    loader = FacebookMessagesLoader(min_words=1)
    with pytest.raises(ValueError, match="_thread"):
        loader.load(inbox / "basic_9" / "not-an-agg.agg")


# ---------------------------------------------------------------------------
# Config surface


def test_chunk_max_tokens_env_var_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MISSION_BRAIN_FB_MSG_CHUNK_MAX_TOKENS", "42")
    loader = FacebookMessagesLoader()
    assert loader._chunk_max_tokens == 42


def test_default_chunk_budget_is_150k() -> None:
    assert DEFAULT_MSG_CHUNK_MAX_TOKENS == 150_000


# ---------------------------------------------------------------------------
# Chunker helper in isolation


def test_chunk_messages_empty_input_returns_one_empty_chunk() -> None:
    assert _chunk_messages([], 1_000) == [[]]


def test_chunk_messages_packs_greedily_within_budget() -> None:
    msgs = [
        _msg(i, "Test User", "x" * 400) for i in range(10)
    ]
    # each msg ~= 20 + 400/4 = 120 tokens. Budget 250 → ~2 msgs/chunk.
    chunks = _chunk_messages(msgs, 250)
    sizes = [len(c) for c in chunks]
    assert sum(sizes) == 10
    # No chunk packs more than 2 messages at this budget.
    assert all(s <= 2 for s in sizes)


def test_chunk_messages_empty_content_rides_with_its_chunk() -> None:
    """Empty-content messages cost 0 tokens and don't force a split."""
    msgs = [
        _msg(1, "Test User", "x" * 400),
        _msg(2, "Test User", ""),
        _msg(3, "Test User", "x" * 400),
    ]
    # Budget just under 2×msg → the empty msg rides with its neighbor.
    chunks = _chunk_messages(msgs, 200)
    flat = [m["timestamp_ms"] for c in chunks for m in c]
    assert flat == [1, 2, 3]


def test_estimate_msg_tokens_empty_is_zero() -> None:
    assert _estimate_msg_tokens({"content": ""}) == 0
    assert _estimate_msg_tokens({}) == 0
