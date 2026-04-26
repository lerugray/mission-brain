"""Checkpoint-driven batch ingest (rayb-015-c).

Covers :func:`run_ingest_batch` behavior:

* First-2-of-5 sources succeed and get recorded in
  ``.ingest-progress.json``.
* The 3rd source raises; an entry lands in ``.ingest-failures.md`` and
  the batch continues.
* 4th and 5th sources succeed and are recorded.
* A re-run with the same inputs skips completed sources and retries
  only the failed one.
* ``force=True`` re-ingests every source from scratch.
* Progress JSON bytes are canonical (sorted keys, stable across runs).
"""

from __future__ import annotations

import json
from pathlib import Path

from mission_brain.ingest.pipeline import (
    FAILURES_FILENAME,
    PROGRESS_FILENAME,
    run_ingest_batch,
)
from mission_brain.loaders.base import SourceDocument
from mission_brain.schema.citation import Citation, LineRangeLocator
from mission_brain.schema.source_id import SourceId
from mission_brain.schema.wiki_page import Paragraph, WikiPage


def _make_doc(tmp_path: Path, sid: str) -> SourceDocument:
    body = f"Paragraph one about {sid}.\nParagraph two about {sid}.\n"
    path = tmp_path / f"{sid}.md"
    path.write_text(body, encoding="utf-8")
    return SourceDocument(
        title=sid,
        body=body,
        frontmatter={},
        line_count=2,
        content_hash="hash-" + sid,
        source_path=path,
        source_type="plain_markdown",
        loader_version="1.0",
        source_id=sid,
        provenance="primary",
    )


class _ScriptedClient:
    """Canned :meth:`generate` returning markdown keyed on the source id."""

    def __init__(self, raise_on: set[str] | None = None) -> None:
        self._raise_on = raise_on or set()
        self.calls: list[str] = []

    def generate(
        self, prompt: str, model: str | None = None,
        temperature: float = 0.0, seed: int = 42,
    ) -> str:
        self.calls.append(prompt)
        for sid in self._raise_on:
            if f"source_id={sid}" in prompt:
                raise RuntimeError(f"synthetic failure for {sid}")
        # Extract the source id from the prompt's header line for canned output.
        for sid in ("src-1", "src-2", "src-3", "src-4", "src-5"):
            if f"source_id={sid}" in prompt:
                return (
                    f"# Page about {sid}\n\n"
                    f"A synthesized claim about {sid}. [ref:{sid}:lines=1-2]\n"
                )
        raise AssertionError(f"unexpected prompt: {prompt[:200]!r}")


def _prior_none(_doc: SourceDocument) -> WikiPage | None:
    return None


def _noop_writer(_page: WikiPage, _doc: SourceDocument) -> None:
    return None


def test_batch_records_successes_and_logs_failure(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    docs = [_make_doc(tmp_path, f"src-{i}") for i in range(1, 6)]
    client = _ScriptedClient(raise_on={"src-3"})

    result = run_ingest_batch(
        sources=docs,
        prior_fn=_prior_none,
        client=client,
        vault_root=vault,
        writer=_noop_writer,
    )

    assert result.attempted == 5
    assert result.completed == ["src-1", "src-2", "src-4", "src-5"]
    assert result.failed == ["src-3"]
    assert result.skipped_checkpoint == []

    progress = json.loads((vault / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert progress == {"completed": ["src-1", "src-2", "src-4", "src-5"]}

    failures = (vault / FAILURES_FILENAME).read_text(encoding="utf-8")
    assert "src-3" in failures
    assert "synthetic failure for src-3" in failures


def test_rerun_skips_completed_and_retries_failures(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    docs = [_make_doc(tmp_path, f"src-{i}") for i in range(1, 6)]

    # First run — src-3 fails.
    run_ingest_batch(
        sources=docs,
        prior_fn=_prior_none,
        client=_ScriptedClient(raise_on={"src-3"}),
        vault_root=vault,
        writer=_noop_writer,
    )

    # Second run — mock no longer raises on src-3.
    client2 = _ScriptedClient(raise_on=set())
    result = run_ingest_batch(
        sources=docs,
        prior_fn=_prior_none,
        client=client2,
        vault_root=vault,
        writer=_noop_writer,
    )

    # Only src-3 should have been (re)attempted via the client.
    assert len(client2.calls) == 1
    assert "source_id=src-3" in client2.calls[0]
    assert result.completed == ["src-3"]
    assert result.skipped_checkpoint == ["src-1", "src-2", "src-4", "src-5"]
    assert result.failed == []

    progress = json.loads((vault / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert progress == {
        "completed": ["src-1", "src-2", "src-3", "src-4", "src-5"]
    }


def test_force_reingests_all_sources(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    docs = [_make_doc(tmp_path, f"src-{i}") for i in range(1, 3)]

    run_ingest_batch(
        sources=docs,
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
    )

    client2 = _ScriptedClient()
    result = run_ingest_batch(
        sources=docs,
        prior_fn=_prior_none,
        client=client2,
        vault_root=vault,
        force=True,
        writer=_noop_writer,
    )
    assert len(client2.calls) == 2
    assert result.completed == ["src-1", "src-2"]
    assert result.skipped_checkpoint == []


def test_progress_json_is_canonical_sorted(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    # Insert out of order to prove the writer sorts.
    docs = [_make_doc(tmp_path, sid) for sid in ("src-5", "src-1", "src-3")]
    run_ingest_batch(
        sources=docs,
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
    )
    text1 = (vault / PROGRESS_FILENAME).read_text(encoding="utf-8")

    # Re-run — noop (all skipped) but must not rewrite differently.
    run_ingest_batch(
        sources=docs,
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
    )
    text2 = (vault / PROGRESS_FILENAME).read_text(encoding="utf-8")
    assert text1 == text2
    assert json.loads(text1)["completed"] == ["src-1", "src-3", "src-5"]


def test_writer_failure_marks_source_as_failed(tmp_path: Path) -> None:
    """If ``writer`` raises, the source counts as failed, not completed."""
    vault = tmp_path / "vault"
    docs = [_make_doc(tmp_path, "src-1")]

    def _bad_writer(_page: WikiPage, _doc: SourceDocument) -> None:
        raise OSError("disk full")

    result = run_ingest_batch(
        sources=docs,
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_bad_writer,
    )
    assert result.failed == ["src-1"]
    assert result.completed == []
    # No progress file should be written when nothing succeeded.
    assert not (vault / PROGRESS_FILENAME).is_file()
    assert (vault / FAILURES_FILENAME).read_text(encoding="utf-8").count("disk full") == 1


def test_prior_fn_receives_each_source(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    docs = [_make_doc(tmp_path, f"src-{i}") for i in range(1, 3)]
    seen: list[str] = []

    def _prior(doc: SourceDocument) -> WikiPage | None:
        seen.append(doc.source_id)
        return WikiPage(
            title="prior",
            source_ids=[SourceId(doc.source_id)],
            paragraphs=[
                Paragraph(
                    text="placeholder",
                    citations=[
                        Citation(
                            source_id=SourceId(doc.source_id),
                            locator=LineRangeLocator(start=1, end=1),
                            excerpt="",
                        )
                    ],
                )
            ],
            provenance="primary",
        )

    run_ingest_batch(
        sources=docs,
        prior_fn=_prior,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
    )
    assert seen == ["src-1", "src-2"]
