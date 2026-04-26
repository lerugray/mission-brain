"""Tests for CLAUDE.md §2.2 manifest-sidecar wiring (rayb-019).

The observation that motivated this task: every plain-md page logged
as ``updated`` on a second ingest instead of ``skipped``, which meant
the LLM call was re-running against unchanged inputs. These tests
pin the contract the wiring now enforces:

1. First ingest writes a ``<slug>.manifest.json`` sidecar.
2. Second ingest with identical inputs short-circuits the LLM and
   logs ``skipped``.
3. Second ingest with modified source content bypasses the cache
   and rewrites the sidecar.
4. The sidecar is canonical JSON (sorted keys, no insignificant
   whitespace).

No real LLM: a ``_CountingClient`` double records ``generate`` calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mission_brain import cli
from mission_brain.config import Settings
from mission_brain.ingest.claude_cli_client import ClaudeCLIClient
from mission_brain.ingest.pipeline import sidecar_path_for
from mission_brain.loaders.base import SourceDocument

_VALID_MARKDOWN = (
    "# Test Page\n"
    "\n"
    "First synthesized claim about the source. "
    "[ref:src-001:lines=1-2]\n"
    "\n"
    "Second synthesized claim covering the remaining lines. "
    "[ref:src-001:lines=2-3]\n"
)


class _CountingClient(ClaudeCLIClient):
    """Minimal ClaudeCLIClient double — counts generate() calls."""

    def __init__(self, canned: str) -> None:
        # Skip parent __init__ (which tries to resolve the CLI binary).
        self._canned = canned
        self.call_count = 0

    def generate(self, prompt: str, **_: object) -> str:  # type: ignore[override]
        self.call_count += 1
        return self._canned


def _doc(body: str = "line1\nline2\nline3\n") -> SourceDocument:
    raw = body.encode("utf-8")
    return SourceDocument(
        title="Test Page",
        body=body,
        frontmatter={},
        line_count=body.count("\n"),
        content_hash=hashlib.sha256(raw).hexdigest(),
        source_path=Path("src-001.md"),
        source_type="plain_markdown",
        loader_version="1.0",
        source_id="src-001",
        provenance="primary",
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        corpus_root=tmp_path / "corpus",
        vault_root=tmp_path / "vault",
    )


def _fresh_counters() -> tuple[dict[str, int], dict[str, int]]:
    bucket = {"new": 0, "updated": 0, "skipped": 0}
    totals = {"new": 0, "updated": 0, "skipped": 0}
    return bucket, totals


def test_first_ingest_writes_manifest_sidecar(settings: Settings) -> None:
    client = _CountingClient(_VALID_MARKDOWN)
    bucket, totals = _fresh_counters()
    cli._ingest_one(_doc(), settings, client, bucket, totals)

    assert client.call_count == 1
    assert totals["new"] == 1
    assert totals["skipped"] == 0

    sidecar = sidecar_path_for(settings.vault_root, "test-page")
    assert sidecar.is_file(), "first ingest must write the manifest sidecar"


def test_identical_second_ingest_skips_llm(settings: Settings) -> None:
    client = _CountingClient(_VALID_MARKDOWN)

    bucket1, totals1 = _fresh_counters()
    cli._ingest_one(_doc(), settings, client, bucket1, totals1)
    assert client.call_count == 1

    bucket2, totals2 = _fresh_counters()
    cli._ingest_one(_doc(), settings, client, bucket2, totals2)

    # The whole point: a second ingest over unchanged inputs must NOT
    # re-invoke the LLM. If call_count grows, the short-circuit is off.
    assert client.call_count == 1
    assert totals2["skipped"] == 1
    assert totals2["new"] == 0
    assert totals2["updated"] == 0


def test_modified_source_bypasses_cache_and_rewrites_sidecar(
    settings: Settings,
) -> None:
    client = _CountingClient(_VALID_MARKDOWN)

    bucket1, totals1 = _fresh_counters()
    cli._ingest_one(_doc(), settings, client, bucket1, totals1)
    assert client.call_count == 1

    modified = _doc(body="line1\nline2\nline3\nnew line four\n")
    bucket2, totals2 = _fresh_counters()
    cli._ingest_one(modified, settings, client, bucket2, totals2)

    assert client.call_count == 2
    assert totals2["updated"] == 1
    assert totals2["skipped"] == 0

    # Sidecar now records the new content_hash under source_hashes.
    sidecar = sidecar_path_for(settings.vault_root, "test-page")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["source_hashes"]["src-001"] == modified.content_hash


def test_sidecar_is_canonical_json(settings: Settings) -> None:
    client = _CountingClient(_VALID_MARKDOWN)
    bucket, totals = _fresh_counters()
    cli._ingest_one(_doc(), settings, client, bucket, totals)

    sidecar = sidecar_path_for(settings.vault_root, "test-page")
    raw = sidecar.read_bytes()

    # Round-trip through json + canonical dump must be byte-identical
    # to what we wrote — i.e., sorted keys + no whitespace separators.
    reserialized = json.dumps(
        json.loads(raw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert raw == reserialized
