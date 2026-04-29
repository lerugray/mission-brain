"""Batch-level manifest-sidecar idempotence (rayb-027).

Covers :func:`run_ingest_batch` behavior when called with the
idempotence parameters wired in:

* First run writes a ``<slug>.manifest.json`` sidecar after writer.
* Second run with identical inputs short-circuits the LLM via the
  cache and records the source_id in ``skipped_manifest``.
* Changing any manifest input (source content, config_digest,
  claude_md_text, overlay) invalidates the cache and forces re-ingest.
* Omitting idempotence params preserves the pre-rayb-027 behavior —
  no sidecar is written, ``skipped_manifest`` stays empty, and every
  source runs through the LLM.
"""

from __future__ import annotations

from pathlib import Path

from mission_brain.ingest.pipeline import (
    run_ingest_batch,
    sidecar_path_for,
)
from mission_brain.loaders.base import SourceDocument
from mission_brain.schema.wiki_page import WikiPage


def _make_doc(tmp_path: Path, sid: str, body: str | None = None) -> SourceDocument:
    body = body if body is not None else (
        f"Paragraph one about {sid}.\nParagraph two about {sid}.\n"
    )
    path = tmp_path / f"{sid}.md"
    path.write_text(body, encoding="utf-8")
    return SourceDocument(
        title=sid,
        body=body,
        frontmatter={},
        line_count=body.count("\n"),
        content_hash="hash-" + sid + "-" + str(len(body)),
        source_path=path,
        source_type="plain_markdown",
        loader_version="1.0",
        source_id=sid,
        provenance="primary",
    )


class _ScriptedClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(
        self, prompt: str, model: str | None = None,
        temperature: float = 0.0, seed: int = 42,
    ) -> str:
        self.calls.append(prompt)
        for sid in ("src-1", "src-2", "src-3"):
            if f"source_id={sid}" in prompt:
                # Wording overlaps the stub corpus lines so post-synthesis
                # `lines=` realignment does not fail ingest.
                return (
                    f"# Page about {sid}\n\n"
                    f"Line one is apple, line two is banana. "
                    f"[ref:{sid}:lines=1-2]\n"
                )
        raise AssertionError(f"unexpected prompt: {prompt[:200]!r}")


def _prior_none(_doc: SourceDocument) -> WikiPage | None:
    return None


def _noop_writer(_page: WikiPage, _doc: SourceDocument) -> None:
    return None


def _prior_sidecar(vault: Path, doc: SourceDocument) -> Path | None:
    p = sidecar_path_for(vault, doc.source_id or doc.source_path.stem)
    return p if p.is_file() else None


def _new_sidecar(vault: Path, page: WikiPage, doc: SourceDocument) -> Path:
    # Use source_id as the slug so first-run path matches prior-run lookup.
    return sidecar_path_for(vault, doc.source_id or doc.source_path.stem)


def test_first_run_writes_sidecar_after_writer(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    doc = _make_doc(tmp_path, "src-1")
    client = _ScriptedClient()

    result = run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=client,
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )

    assert len(client.calls) == 1
    assert result.completed == ["src-1"]
    assert result.skipped_manifest == []
    assert sidecar_path_for(vault, "src-1").is_file()


def test_second_run_unchanged_inputs_skips_via_manifest(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    doc = _make_doc(tmp_path, "src-1")

    run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )

    # Wipe the checkpoint so we're forced to exercise the manifest path,
    # not the intra-run dedup path.
    (vault / ".ingest-progress.json").unlink()

    client2 = _ScriptedClient()
    result = run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=client2,
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )

    assert client2.calls == []  # LLM not invoked
    assert result.skipped_manifest == ["src-1"]
    assert result.completed == []


def test_changed_content_invalidates_cache(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    doc1 = _make_doc(tmp_path, "src-1", body="apple\nbanana\n")

    run_ingest_batch(
        sources=[doc1],
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )
    (vault / ".ingest-progress.json").unlink()

    doc2 = _make_doc(tmp_path, "src-1", body="apple\nbanana\ncarrot\n")
    client2 = _ScriptedClient()
    result = run_ingest_batch(
        sources=[doc2],
        prior_fn=_prior_none,
        client=client2,
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )

    assert len(client2.calls) == 1
    assert result.skipped_manifest == []
    assert result.completed == ["src-1"]


def test_changed_claude_md_invalidates_cache(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    doc = _make_doc(tmp_path, "src-1")

    run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )
    (vault / ".ingest-progress.json").unlink()

    client2 = _ScriptedClient()
    result = run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=client2,
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v2 — edited",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )

    assert len(client2.calls) == 1
    assert result.skipped_manifest == []


def test_changed_config_digest_invalidates_cache(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    doc = _make_doc(tmp_path, "src-1")

    run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )
    (vault / ".ingest-progress.json").unlink()

    client2 = _ScriptedClient()
    result = run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=client2,
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-DIFFERENT",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )

    assert len(client2.calls) == 1
    assert result.skipped_manifest == []


def test_overlay_change_invalidates_cache(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    doc = _make_doc(tmp_path, "src-1")
    overlay = tmp_path / "src-1.user-edits.md"
    overlay.write_text("original overlay text\n", encoding="utf-8")

    run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
        overlay_path_fn=lambda d: overlay,
    )
    (vault / ".ingest-progress.json").unlink()

    overlay.write_text("overlay text edited by the user\n", encoding="utf-8")
    client2 = _ScriptedClient()
    result = run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=client2,
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
        overlay_path_fn=lambda d: overlay,
    )

    assert len(client2.calls) == 1
    assert result.skipped_manifest == []


def test_idempotence_params_omitted_preserves_legacy_behavior(
    tmp_path: Path,
) -> None:
    """Backward compatibility: no idempotence params → no sidecar path."""
    vault = tmp_path / "vault"
    doc = _make_doc(tmp_path, "src-1")

    run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
    )
    (vault / ".ingest-progress.json").unlink()

    client2 = _ScriptedClient()
    result = run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=client2,
        vault_root=vault,
        writer=_noop_writer,
    )

    # Without idempotence params the LLM fires again — there's no cache.
    assert len(client2.calls) == 1
    assert result.skipped_manifest == []
    assert result.completed == ["src-1"]
    # And no sidecar is materialised.
    assert not sidecar_path_for(vault, "src-1").is_file()


def test_force_bypasses_manifest_cache(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    doc = _make_doc(tmp_path, "src-1")

    run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=_ScriptedClient(),
        vault_root=vault,
        writer=_noop_writer,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )

    client2 = _ScriptedClient()
    result = run_ingest_batch(
        sources=[doc],
        prior_fn=_prior_none,
        client=client2,
        vault_root=vault,
        writer=_noop_writer,
        force=True,
        claude_md_text="schema v1",
        config_digest="cfg-abc",
        prior_sidecar_path_fn=lambda d: _prior_sidecar(vault, d),
        new_sidecar_path_fn=lambda p, d: _new_sidecar(vault, p, d),
    )

    assert len(client2.calls) == 1
    assert result.skipped_manifest == []
    assert result.completed == ["src-1"]
