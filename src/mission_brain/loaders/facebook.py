"""Facebook JSON loader — posts + messages + groups (rayb-016-a).

Reads Facebook's JSON export shape and emits :class:`SourceDocument`
instances shaped for the mission-brain ingest pipeline. Three loaders,
one module — they share small helpers (text normalization, slug,
timestamp rendering, user-authored filtering) so splitting into
separate modules would be premature.

**Aggregation granularity** (PLAN-facebook-ingest-2026-04-20.md §1):

* ``FacebookPostsLoader`` — one ``SourceDocument`` per calendar
  **year** with at least one post. ``discover()`` yields synthetic
  per-year paths (``corpus/facebook/posts/_year_YYYY.agg``) that do
  not exist on disk; ``load()`` parses the year out and walks the
  real ``your_posts__check_ins__photos_and_videos_*.json`` files.
* ``FacebookMessagesLoader`` — one ``SourceDocument`` per
  message thread whose user-authored word count meets the
  threshold (default 5000, override via ``MISSION_BRAIN_FB_MSG_MIN_WORDS``
  env or constructor kwarg). ``discover()`` yields synthetic paths
  ``corpus/facebook/messages/inbox/<thread>/_thread.agg`` one per
  qualifying thread.
* ``FacebookGroupsLoader`` — one ``SourceDocument`` per group
  whose user contribution count meets the threshold (default 20,
  override via ``MISSION_BRAIN_FB_GROUP_MIN_CONTRIBS``). Synthetic paths
  ``corpus/facebook/groups/_group_<slug>.agg``.

**Locator grammar.** All three emit ``para=<anchor>`` locators via
the ingest prompt; ``locator_for`` raises ``NotImplementedError``
to match the music loader's pattern.

**Post identifiers.** FB's export omits a stable post-id field, so
we anchor by unix timestamp with a ``-N`` suffix on same-second
collisions. Message anchors use ``timestamp_ms`` (no collisions
in practice). Group posts reuse the timestamp-based scheme.

**Mojibake handling.** FB exports emoji + unicode as latin-1-encoded
UTF-8 by default (documented quirk). The :func:`_fix_mojibake`
helper round-trips each string through ``latin-1 → utf-8`` and
keeps the result when it decodes cleanly.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mission_brain.loaders.base import SourceDocument, SourceLoader
from mission_brain.schema.wiki_page import Provenance

__all__ = [
    "DEFAULT_GROUP_MIN_CONTRIBS",
    "DEFAULT_MSG_CHUNK_MAX_TOKENS",
    "DEFAULT_MSG_MIN_WORDS",
    "FacebookGroupsLoader",
    "FacebookMessagesLoader",
    "FacebookPostsLoader",
    "FACEBOOK_USER_DISPLAY_NAME",
]

FACEBOOK_USER_DISPLAY_NAME = "Test User"
DEFAULT_MSG_MIN_WORDS = 5000
DEFAULT_GROUP_MIN_CONTRIBS = 20
# Threads whose rendered body exceeds this estimated token count
# (chars/4 heuristic) are chunked by discover() into multiple
# _thread_partN.agg SourceDocuments, each <= budget. Default leaves
# headroom below a 200k-token synthesis context window (rayb-018-a,
# spec per HANDOFF-2026-04-20.md).
DEFAULT_MSG_CHUNK_MAX_TOKENS = 150_000

_POSTS_GLOB = "your_posts__check_ins__photos_and_videos_*.json"
_GROUP_POSTS_FILENAME = "group_posts_and_comments.json"

_YEAR_AGG_RE = re.compile(r"_year_(?P<year>\d{4})\.agg$")
_THREAD_AGG_SUFFIX = "_thread.agg"
_THREAD_PART_RE = re.compile(r"_thread_part(?P<part>\d+)\.agg$")
_GROUP_AGG_RE = re.compile(r"_group_(?P<slug>[a-z0-9][a-z0-9_-]*)\.agg$")

# Characters per approximated token. English-text heuristic used to
# decide when to chunk; avoids depending on a tokenizer at loader
# scope. Conservative — overcounts shorter messages, undercounts
# token-dense code/URLs, both fine for a safety budget.
_CHARS_PER_TOKEN = 4

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_GROUP_TITLE_PATTERNS = (
    re.compile(r"^.*?\bposted in (?P<group>.+?)\.\s*$"),
    re.compile(r"^.*?\bin (?P<group>.+?)'s post in (?P<group2>.+?)\.\s*$"),
    re.compile(r"^.*?\bcommented on .+?'s post in (?P<group>.+?)\.\s*$"),
    re.compile(r"^.*?\breplied to .+? in (?P<group>.+?)\.\s*$"),
    re.compile(r"^.*?\bshared .+? in (?P<group>.+?)\.\s*$"),
)


def _slugify(name: str) -> str:
    """CLAUDE.md §1-compliant slug: lowercase ``[a-z0-9_-]+``."""
    slug = _NON_ALNUM.sub("-", name.lower()).strip("-")
    return slug or "unknown"


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _format_ts_utc(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_ts_ms_utc(ts_ms: int) -> str:
    s, ms = divmod(int(ts_ms), 1000)
    return (
        _dt.datetime.fromtimestamp(s, _dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
        + f".{ms:03d} UTC"
    )


def _fix_mojibake(text: str) -> str:
    """FB export double-encodes UTF-8 as latin-1. Round-trip to recover.

    Leaves ``text`` unchanged if the round-trip raises — so ASCII stays
    ASCII and anything already valid UTF-8 is unaffected.
    """
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8"))


def _hash_file_list(files: Iterable[Path], extra: dict[str, Any]) -> str:
    """Stable content_hash input covering *files*' bytes plus *extra* keys."""
    payload: dict[str, Any] = dict(extra)
    payload["files"] = sorted(
        (p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files
    )
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extract_post_prose(entry: dict[str, Any]) -> str:
    """Pull prose from a post entry: title (if author-written),
    ``data[].post`` texts, attachment descriptions + external-context names.
    Returns joined string with blank-line separators; empty if no prose.
    """
    parts: list[str] = []
    title = entry.get("title")
    if isinstance(title, str) and title.strip():
        stripped = title.strip()
        # FB auto-generates title either as bare display name (for a pure
        # prose post) or as "<name> <verb>..." (for structural
        # activity — added a photo, shared a link, checked in). Neither
        # is user-authored prose; skip both.
        u = FACEBOOK_USER_DISPLAY_NAME
        auto_prefixes = (
            f"{u} added ",
            f"{u} shared ",
            f"{u} updated ",
            f"{u} was ",
            f"{u} is ",
            f"{u} posted ",
            f"{u} wrote ",
            f"{u} changed ",
            f"{u} and ",
            f"{u} became ",
            f"{u} commented ",
            f"{u} replied ",
            f"{u} liked ",
            f"{u} checked in",
            f"{u} is now ",
        )
        is_auto = stripped == FACEBOOK_USER_DISPLAY_NAME or stripped.startswith(
            auto_prefixes
        )
        if not is_auto:
            parts.append(_fix_mojibake(stripped))
    for item in entry.get("data") or []:
        if not isinstance(item, dict):
            continue
        post = item.get("post")
        if isinstance(post, str) and post.strip():
            parts.append(_fix_mojibake(post.strip()))
    for att in entry.get("attachments") or []:
        if not isinstance(att, dict):
            continue
        for item in att.get("data") or []:
            if not isinstance(item, dict):
                continue
            media = item.get("media")
            if isinstance(media, dict):
                desc = media.get("description")
                if isinstance(desc, str) and desc.strip():
                    parts.append(_fix_mojibake(desc.strip()))
                media_title = media.get("title")
                if (
                    isinstance(media_title, str)
                    and media_title.strip()
                    and media_title.strip() not in {"Photos", "Timeline Photos"}
                ):
                    parts.append(_fix_mojibake(media_title.strip()))
            ext = item.get("external_context")
            if isinstance(ext, dict):
                name = ext.get("name")
                if isinstance(name, str) and name.strip():
                    parts.append(_fix_mojibake(name.strip()))
    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Posts loader


class FacebookPostsLoader(SourceLoader):
    source_type = "facebook_posts"
    supported_extensions = [".json"]
    version = "1.0"

    def __init__(self, provenance: Provenance = "primary") -> None:
        self._provenance: Provenance = provenance

    def discover(self, corpus_root: Path) -> Iterable[Path]:
        posts_dir = corpus_root / "facebook" / "posts"
        if not posts_dir.is_dir():
            return []
        years = _collect_post_years(posts_dir)
        return [posts_dir / f"_year_{year}.agg" for year in sorted(years)]

    def load(self, path: Path) -> SourceDocument:
        m = _YEAR_AGG_RE.search(str(path.as_posix()))
        if not m:
            raise ValueError(
                f"FacebookPostsLoader.load expects _year_YYYY.agg, got {path.name!r}"
            )
        year = int(m.group("year"))
        posts_dir = path.parent
        files = _iter_post_files(posts_dir)

        entries = _collect_year_entries(files, year)
        seen_ts: dict[int, int] = {}

        lines: list[str] = [
            f"title: Facebook Posts — {year}",
            f"year: {year}",
            f"post_count_total: {len(entries)}",
            "",
        ]
        rendered = 0
        skipped: list[str] = []
        current_month: str | None = None
        for ts, _fidx, _pos, entry in entries:
            post_id = _allocate_post_id(ts, seen_ts)
            month = _dt.datetime.fromtimestamp(ts, _dt.UTC).strftime("%B %Y")
            if month != current_month:
                current_month = month
                lines.append("")
                lines.append(f"## {month}")
                lines.append("")
            prose = _extract_post_prose(entry)
            if not prose:
                skipped.append(post_id)
                continue
            lines.append(f"[post_id={post_id} at={_format_ts_utc(ts)}]")
            lines.append(prose)
            lines.append("")
            rendered += 1

        body = "\n".join(lines).rstrip() + "\n"
        content_hash = _hash_file_list(
            [p for p, _ in files], {"kind": "facebook_posts_year", "year": year}
        )

        return SourceDocument(
            title=f"Facebook Posts — {year}",
            body=body,
            frontmatter={
                "year": year,
                "post_count_total": len(entries),
                "post_count_rendered": rendered,
                "post_ids_skipped": skipped,
            },
            line_count=len(body.splitlines()),
            content_hash=content_hash,
            source_path=path,
            source_type=self.source_type,
            loader_version=self.version,
            source_id=f"fb-posts-{year}",
            provenance=self._provenance,
        )

    def content_hash(self, raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def locator_for(self, span: tuple[int, int]) -> str:
        raise NotImplementedError(
            "FacebookPostsLoader emits para=<post_id> locators via the ingest "
            "prompt (not raw-body spans); locator_for is unused."
        )


def _iter_post_files(posts_dir: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    out: list[tuple[Path, list[dict[str, Any]]]] = []
    for p in sorted(posts_dir.glob(_POSTS_GLOB)):
        if not p.is_file():
            continue
        try:
            data = _read_json_file(p)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            out.append((p, [e for e in data if isinstance(e, dict)]))
    return out


def _collect_post_years(posts_dir: Path) -> set[int]:
    years: set[int] = set()
    for _, entries in _iter_post_files(posts_dir):
        for entry in entries:
            ts = entry.get("timestamp")
            if isinstance(ts, (int, float)) and ts > 0:
                years.add(_dt.datetime.fromtimestamp(ts, _dt.UTC).year)
    return years


def _collect_year_entries(
    files: list[tuple[Path, list[dict[str, Any]]]], year: int
) -> list[tuple[int, int, int, dict[str, Any]]]:
    out: list[tuple[int, int, int, dict[str, Any]]] = []
    for fidx, (_, entries) in enumerate(files):
        for pos, entry in enumerate(entries):
            ts = entry.get("timestamp")
            if not isinstance(ts, (int, float)) or ts <= 0:
                continue
            entry_year = _dt.datetime.fromtimestamp(ts, _dt.UTC).year
            if entry_year != year:
                continue
            out.append((int(ts), fidx, pos, entry))
    out.sort(key=lambda t: (t[0], t[1], t[2]))
    return out


def _allocate_post_id(ts: int, seen: dict[int, int]) -> str:
    count = seen.get(ts, 0)
    seen[ts] = count + 1
    return f"{ts}" if count == 0 else f"{ts}-{count}"


# ---------------------------------------------------------------------------
# Messages loader


class FacebookMessagesLoader(SourceLoader):
    source_type = "facebook_messages"
    supported_extensions = [".json"]
    version = "1.1"

    def __init__(
        self,
        min_words: int | None = None,
        chunk_max_tokens: int | None = None,
        provenance: Provenance = "primary",
    ) -> None:
        if min_words is None:
            env = os.environ.get("MISSION_BRAIN_FB_MSG_MIN_WORDS")
            min_words = int(env) if env else DEFAULT_MSG_MIN_WORDS
        if chunk_max_tokens is None:
            env_c = os.environ.get("MISSION_BRAIN_FB_MSG_CHUNK_MAX_TOKENS")
            chunk_max_tokens = (
                int(env_c) if env_c else DEFAULT_MSG_CHUNK_MAX_TOKENS
            )
        self._min_words = min_words
        self._chunk_max_tokens = chunk_max_tokens
        self._provenance: Provenance = provenance

    def discover(self, corpus_root: Path) -> Iterable[Path]:
        inbox = corpus_root / "facebook" / "messages" / "inbox"
        if not inbox.is_dir():
            return []
        out: list[Path] = []
        for thread_dir in sorted(inbox.iterdir()):
            if not thread_dir.is_dir():
                continue
            merged = _merge_thread(thread_dir)
            if merged is None:
                continue
            _parts, messages = merged
            user_words = sum(
                _word_count(m.get("content", ""))
                for m in messages
                if _is_user_msg(m)
            )
            if user_words < self._min_words:
                continue
            chunks = _chunk_messages(messages, self._chunk_max_tokens)
            if len(chunks) == 1:
                out.append(thread_dir / _THREAD_AGG_SUFFIX)
            else:
                for idx in range(1, len(chunks) + 1):
                    out.append(thread_dir / f"_thread_part{idx}.agg")
        return out

    def load(self, path: Path) -> SourceDocument:
        part_index, total_parts = _parse_thread_part(path.name)
        thread_dir = path.parent
        merged = _merge_thread(thread_dir)
        if merged is None:
            raise ValueError(f"no message_*.json files under {thread_dir}")
        participants, messages = merged

        if part_index is None:
            chunk_messages = messages
            slice_total = 1
            slice_index = 1
        else:
            chunks = _chunk_messages(messages, self._chunk_max_tokens)
            if total_parts is not None and len(chunks) != total_parts:
                # The path names a specific partN but the chunker disagrees on
                # total count — treated as an inconsistency rather than
                # silently re-slicing. Callers re-run discover when the
                # budget or corpus changes.
                raise ValueError(
                    f"chunk count mismatch for {thread_dir.name}: path "
                    f"requested part {total_parts}, chunker produced "
                    f"{len(chunks)}"
                )
            if part_index < 1 or part_index > len(chunks):
                raise ValueError(
                    f"part index {part_index} out of range 1..{len(chunks)} "
                    f"for {thread_dir.name}"
                )
            chunk_messages = chunks[part_index - 1]
            slice_total = len(chunks)
            slice_index = part_index

        thread_slug = _slugify(thread_dir.name)
        source_id = f"fb-msg-{thread_slug}"
        if slice_total > 1:
            source_id = f"{source_id}-part{slice_index}"

        other = [p for p in participants if p != FACEBOOK_USER_DISPLAY_NAME]
        title_source = ", ".join(other) if other else "self"
        title = f"Facebook Messages — {title_source}"
        if slice_total > 1:
            title = f"{title} (part {slice_index} of {slice_total})"

        user_words = sum(
            _word_count(m.get("content", ""))
            for m in chunk_messages
            if _is_user_msg(m)
        )
        total_words = sum(
            _word_count(m.get("content", "")) for m in chunk_messages
        )

        lines: list[str] = [
            f"title: {title}",
            f"thread: {thread_dir.name}",
            f"participants: {', '.join(sorted(participants))}",
            f"message_count: {len(chunk_messages)}",
            f"user_words: {user_words}",
        ]
        if slice_total > 1:
            lines.append(f"part: {slice_index}")
            lines.append(f"total_parts: {slice_total}")
        lines.append("")
        for m in chunk_messages:
            ts_ms = int(m.get("timestamp_ms", 0))
            tag = "[self]" if _is_user_msg(m) else "[them]"
            content = _fix_mojibake((m.get("content") or "").strip())
            if not content:
                continue
            lines.append(
                f"[msg_id={ts_ms} at={_format_ts_ms_utc(ts_ms)}] {tag}"
            )
            lines.append(content)
            lines.append("")

        body = "\n".join(lines).rstrip() + "\n"
        message_files = sorted(thread_dir.glob("message_*.json"))
        hash_extras: dict[str, Any] = {
            "kind": "facebook_messages_thread",
            "thread": thread_dir.name,
        }
        if slice_total > 1:
            hash_extras["part"] = slice_index
            hash_extras["total_parts"] = slice_total
            hash_extras["chunk_max_tokens"] = self._chunk_max_tokens
        content_hash = _hash_file_list(message_files, hash_extras)

        frontmatter: dict[str, Any] = {
            "thread_dir": thread_dir.name,
            "participants": sorted(participants),
            "message_count": len(chunk_messages),
            "user_words": user_words,
            "total_words": total_words,
        }
        if slice_total > 1:
            frontmatter["part"] = slice_index
            frontmatter["total_parts"] = slice_total

        return SourceDocument(
            title=title,
            body=body,
            frontmatter=frontmatter,
            line_count=len(body.splitlines()),
            content_hash=content_hash,
            source_path=path,
            source_type=self.source_type,
            loader_version=self.version,
            source_id=source_id,
            provenance=self._provenance,
        )

    def content_hash(self, raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def locator_for(self, span: tuple[int, int]) -> str:
        raise NotImplementedError(
            "FacebookMessagesLoader emits para=<timestamp_ms> locators via the "
            "ingest prompt; locator_for is unused."
        )


def _parse_thread_part(name: str) -> tuple[int | None, int | None]:
    """Return ``(part_index, total_parts)`` for a thread agg path.

    ``_thread.agg`` → ``(None, None)`` (non-chunked). ``_thread_partN.agg``
    → ``(N, None)`` — the total is not encoded in the filename; load()
    rederives it by re-chunking. Unrecognized names raise ``ValueError``.
    """
    if name == _THREAD_AGG_SUFFIX:
        return None, None
    m = _THREAD_PART_RE.search(name)
    if not m:
        raise ValueError(
            f"FacebookMessagesLoader.load expects _thread.agg or "
            f"_thread_partN.agg, got {name!r}"
        )
    return int(m.group("part")), None


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _estimate_msg_tokens(msg: dict[str, Any]) -> int:
    """Token budget contribution of one rendered message line pair.

    Covers the header ``[msg_id=... at=...] [self|them]`` plus the content
    line plus a blank separator. Messages with empty content are skipped
    during rendering, so they contribute zero tokens.
    """
    content = (msg.get("content") or "").strip()
    if not content:
        return 0
    # Header line is roughly fixed at ~60 chars (timestamp + tag); content
    # varies. Use a fixed overhead of 20 tokens per message to cover the
    # header + blank separator; the dominant term is content length.
    return 20 + _estimate_tokens(content)


def _chunk_messages(
    messages: list[dict[str, Any]], max_tokens: int
) -> list[list[dict[str, Any]]]:
    """Greedily pack *messages* (chronological) into <= ``max_tokens`` chunks.

    Deterministic: same input + budget → same chunking. Called from both
    ``discover`` (to decide how many synthetic paths to yield) and
    ``load`` (to re-derive the messages for a requested ``partN``).

    A single message whose estimated size exceeds ``max_tokens`` lands in
    its own chunk — we do not split below the message granularity.
    """
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for msg in messages:
        msg_tokens = _estimate_msg_tokens(msg)
        if msg_tokens == 0:
            current.append(msg)
            continue
        if current and current_tokens + msg_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(msg)
        current_tokens += msg_tokens
    if current:
        chunks.append(current)
    if not chunks:
        chunks.append([])
    return chunks


def _merge_thread(
    thread_dir: Path,
) -> tuple[list[str], list[dict[str, Any]]] | None:
    """Merge all ``message_*.json`` files in *thread_dir* into sorted messages."""
    files = sorted(thread_dir.glob("message_*.json"))
    if not files:
        return None
    all_msgs: list[dict[str, Any]] = []
    participants: set[str] = set()
    for p in files:
        try:
            data = _read_json_file(p)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for part in data.get("participants") or []:
            if isinstance(part, dict):
                name = part.get("name")
                if isinstance(name, str) and name.strip():
                    participants.add(_fix_mojibake(name.strip()))
        for msg in data.get("messages") or []:
            if isinstance(msg, dict) and msg.get("timestamp_ms") is not None:
                all_msgs.append(msg)
    all_msgs.sort(key=lambda m: int(m.get("timestamp_ms", 0)))
    return sorted(participants), all_msgs


def _is_user_msg(msg: dict[str, Any]) -> bool:
    return msg.get("sender_name") == FACEBOOK_USER_DISPLAY_NAME


# ---------------------------------------------------------------------------
# Groups loader


class FacebookGroupsLoader(SourceLoader):
    source_type = "facebook_groups"
    supported_extensions = [".json"]
    version = "1.0"

    def __init__(
        self,
        min_contribs: int | None = None,
        provenance: Provenance = "primary",
    ) -> None:
        if min_contribs is None:
            env = os.environ.get("MISSION_BRAIN_FB_GROUP_MIN_CONTRIBS")
            min_contribs = int(env) if env else DEFAULT_GROUP_MIN_CONTRIBS
        self._min_contribs = min_contribs
        self._provenance: Provenance = provenance

    def discover(self, corpus_root: Path) -> Iterable[Path]:
        groups_dir = corpus_root / "facebook" / "groups"
        source = groups_dir / _GROUP_POSTS_FILENAME
        if not source.is_file():
            return []
        grouped = _group_posts_by_name(source)
        out: list[Path] = []
        for group_name, entries in sorted(grouped.items()):
            if len(entries) < self._min_contribs:
                continue
            slug = _slugify(group_name)
            out.append(groups_dir / f"_group_{slug}.agg")
        return out

    def load(self, path: Path) -> SourceDocument:
        m = _GROUP_AGG_RE.search(str(path.as_posix()))
        if not m:
            raise ValueError(
                f"FacebookGroupsLoader.load expects _group_<slug>.agg, "
                f"got {path.name!r}"
            )
        slug = m.group("slug")
        groups_dir = path.parent
        source = groups_dir / _GROUP_POSTS_FILENAME
        if not source.is_file():
            raise ValueError(f"no {_GROUP_POSTS_FILENAME} under {groups_dir}")

        grouped = _group_posts_by_name(source)
        match: list[tuple[int, int, dict[str, Any]]] | None = None
        matched_name: str | None = None
        for group_name, entries in grouped.items():
            if _slugify(group_name) == slug:
                match = entries
                matched_name = group_name
                break
        if match is None or matched_name is None:
            raise ValueError(f"no group matched slug {slug!r} in {source}")

        match_sorted = sorted(match, key=lambda t: (t[0], t[1]))
        seen_ts: dict[int, int] = {}
        lines: list[str] = [
            f"title: Facebook Group — {matched_name}",
            f"group: {matched_name}",
            f"contribution_count: {len(match_sorted)}",
            "",
        ]
        rendered = 0
        skipped: list[str] = []
        current_year: int | None = None
        for ts, _pos, entry in match_sorted:
            year = _dt.datetime.fromtimestamp(ts, _dt.UTC).year
            if year != current_year:
                current_year = year
                lines.append("")
                lines.append(f"## {year}")
                lines.append("")
            post_id = _allocate_post_id(ts, seen_ts)
            prose = _extract_post_prose(entry)
            if not prose:
                skipped.append(post_id)
                continue
            lines.append(f"[post_id={post_id} at={_format_ts_utc(ts)}]")
            lines.append(prose)
            lines.append("")
            rendered += 1

        body = "\n".join(lines).rstrip() + "\n"
        content_hash = _hash_file_list(
            [source],
            {"kind": "facebook_group", "group": matched_name, "slug": slug},
        )

        return SourceDocument(
            title=f"Facebook Group — {matched_name}",
            body=body,
            frontmatter={
                "group_name": matched_name,
                "slug": slug,
                "contribution_count": len(match_sorted),
                "contribution_count_rendered": rendered,
                "post_ids_skipped": skipped,
            },
            line_count=len(body.splitlines()),
            content_hash=content_hash,
            source_path=path,
            source_type=self.source_type,
            loader_version=self.version,
            source_id=f"fb-group-{slug}",
            provenance=self._provenance,
        )

    def content_hash(self, raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def locator_for(self, span: tuple[int, int]) -> str:
        raise NotImplementedError(
            "FacebookGroupsLoader emits para=<post_id> locators via the ingest "
            "prompt; locator_for is unused."
        )


def _extract_group_name(title: str) -> str | None:
    if not isinstance(title, str):
        return None
    for pattern in _GROUP_TITLE_PATTERNS:
        m = pattern.match(title.strip())
        if m:
            return _fix_mojibake(m.group("group").strip())
    return None


def _group_posts_by_name(
    source: Path,
) -> dict[str, list[tuple[int, int, dict[str, Any]]]]:
    try:
        data = _read_json_file(source)
    except (OSError, json.JSONDecodeError):
        return {}
    entries = []
    if isinstance(data, dict):
        entries = data.get("group_posts_v2") or data.get("group_posts") or []
    elif isinstance(data, list):
        entries = data
    grouped: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
    for pos, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        ts = entry.get("timestamp")
        if not isinstance(ts, (int, float)) or ts <= 0:
            continue
        name = _extract_group_name(entry.get("title", "") or "")
        if not name:
            continue
        grouped.setdefault(name, []).append((int(ts), pos, entry))
    return grouped
