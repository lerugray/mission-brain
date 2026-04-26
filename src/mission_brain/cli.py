from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _package_version
from pathlib import Path
from typing import cast

import json

import typer

from mission_brain.config import IngestBackend


def _cli_version() -> str:
    try:
        return _package_version("mission-brain")
    except PackageNotFoundError:
        return "0.0.0"


app = typer.Typer(help="mission-brain — retrieval-only second-brain CLI.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"mission-brain {_cli_version()}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """mission-brain CLI entry point."""


@app.command("eval")
def eval_cmd(
    golden: Path = typer.Option(..., "--golden", help="Path to eval/golden/queries.yaml."),
    gate: list[str] = typer.Option([], "--gate", help="Metric gate, e.g. context_recall=0.70."),
) -> None:
    """Eval harness is not shipped in the mission-brain public template."""
    _ = (golden, gate)
    typer.echo(
        "The eval subcommand is not included in the mission-brain template.",
        err=True,
    )
    raise typer.Exit(code=1)


MUSIC_CATALOG_SOURCE_ID = "music-catalog-analysis"


@app.command("ingest")
def ingest_cmd(
    file: Path | None = typer.Option(
        None,
        "--file",
        help=(
            "Ingest a single source file by path (bypasses corpus discovery). "
            "Useful for smoke-testing one song before committing to a full batch."
        ),
    ),
    ingest_backend_opt: str | None = typer.Option(
        None,
        "--ingest-backend",
        help=(
            "Ingest LLM backend: claude or cursor. "
            "Default: MISSION_BRAIN_INGEST_BACKEND env var, else claude."
        ),
    ),
) -> None:
    """Walk the corpus, synthesize wiki pages, update index + log.

    Walks two plain-markdown subfolders with different provenance tags:
    ``plain-md/`` (primary — Ray-authored) and ``plain-md-derived/``
    (derived — content synthesized by another tool). Also walks
    ``music/per-song/`` and ``music/catalog/`` with the
    :class:`MusicDataLoader` and ``facebook/`` with the three
    Facebook loaders (all primary provenance). All feed the same
    vault; queries distinguish sources via ``WikiPage.provenance``.
    """
    from mission_brain.loaders.base import canonical_source_id
    from mission_brain.loaders.facebook import (
        FacebookGroupsLoader,
        FacebookMessagesLoader,
        FacebookPostsLoader,
    )
    from mission_brain.loaders.journal import MissionBulletLoader
    from mission_brain.loaders.music_data import MusicDataLoader
    from mission_brain.loaders.plain_markdown import PlainMarkdownLoader
    from mission_brain.schema.wiki_page import Provenance

    settings = _load_settings()
    ingest_backend_override = _parse_ingest_backend_cli(ingest_backend_opt)
    client = _build_ingest_client(settings, ingest_backend=ingest_backend_override)

    if file is not None:
        _ingest_single_file(file, settings, client)
        return

    totals = {"new": 0, "updated": 0, "skipped": 0}
    per_prov: dict[str, dict[str, int]] = {}
    music_total = 0
    fb_posts_total = 0
    fb_msg_total = 0
    fb_group_total = 0
    journal_total = 0

    subdirs: list[tuple[str, Provenance]] = [
        ("plain-md", "primary"),
        ("plain-md-derived", "derived"),
    ]

    any_found = False
    source_idx = 0
    for subdir, provenance in subdirs:
        discover_root = settings.corpus_root / subdir
        if not discover_root.is_dir():
            continue
        any_found = True
        loader = PlainMarkdownLoader(provenance=provenance)
        bucket = per_prov.setdefault(
            provenance, {"new": 0, "updated": 0, "skipped": 0}
        )
        for source_path in loader.discover(discover_root):
            doc = loader.load(source_path)
            doc.source_id = canonical_source_id(source_path, settings.corpus_root)
            source_idx += 1
            _ingest_one(doc, settings, client, bucket, totals, idx=source_idx)

    music_root = settings.corpus_root / "music"
    if music_root.is_dir():
        any_found = True
        music_loader = MusicDataLoader(provenance="primary")
        bucket = per_prov.setdefault(
            "primary", {"new": 0, "updated": 0, "skipped": 0}
        )
        for source_path in music_loader.discover(settings.corpus_root):
            doc = music_loader.load(source_path)
            doc.source_id = canonical_source_id(source_path, settings.corpus_root)
            source_idx += 1
            _ingest_one(
                doc, settings, client, bucket, totals, loader=music_loader, idx=source_idx
            )
            music_total += 1
        for source_path in music_loader.discover_catalog(settings.corpus_root):
            doc = music_loader.load(source_path)
            doc.source_id = MUSIC_CATALOG_SOURCE_ID
            source_idx += 1
            _ingest_one(
                doc, settings, client, bucket, totals, loader=music_loader, idx=source_idx
            )
            music_total += 1

    fb_root = settings.corpus_root / "facebook"
    if fb_root.is_dir():
        any_found = True
        bucket = per_prov.setdefault(
            "primary", {"new": 0, "updated": 0, "skipped": 0}
        )
        for fb_loader, counter_key in (
            (FacebookPostsLoader(provenance="primary"), "posts"),
            (
                FacebookMessagesLoader(
                    min_words=settings.fb_msg_min_words, provenance="primary"
                ),
                "msg",
            ),
            (
                FacebookGroupsLoader(
                    min_contribs=settings.fb_group_min_contribs,
                    provenance="primary",
                ),
                "group",
            ),
        ):
            for source_path in fb_loader.discover(settings.corpus_root):
                doc = fb_loader.load(source_path)
                # FB loaders set source_id explicitly in load(); trust it.
                source_idx += 1
                _ingest_one(
                    doc, settings, client, bucket, totals, loader=fb_loader, idx=source_idx
                )
                if counter_key == "posts":
                    fb_posts_total += 1
                elif counter_key == "msg":
                    fb_msg_total += 1
                elif counter_key == "group":
                    fb_group_total += 1

    # rb-036: mission-bullet journal corpus. Gated by settings.journal_enabled
    # (default True per rb-035) so a Voyage-only / Ollama-only / hands-off
    # config can disable journal ingest without code change.
    journal_root = settings.corpus_root / "journal"
    if settings.journal_enabled and journal_root.is_dir():
        any_found = True
        journal_loader = MissionBulletLoader(provenance="journal-private")
        bucket = per_prov.setdefault(
            "journal-private", {"new": 0, "updated": 0, "skipped": 0}
        )
        for source_path in journal_loader.discover(settings.corpus_root):
            doc = journal_loader.load(source_path)
            doc.source_id = canonical_source_id(source_path, settings.corpus_root)
            source_idx += 1
            _ingest_one(
                doc,
                settings,
                client,
                bucket,
                totals,
                loader=journal_loader,
                idx=source_idx,
            )
            journal_total += 1

    if not any_found:
        typer.echo(
            f"no sources under {settings.corpus_root}; nothing to ingest."
        )
        typer.echo("0 new, 0 updated, 0 skipped")
        return

    typer.echo(
        f"{totals['new']} new, {totals['updated']} updated, "
        f"{totals['skipped']} skipped"
    )
    for provenance in ("primary", "derived", "journal-private"):
        if provenance in per_prov:
            b = per_prov[provenance]
            typer.echo(
                f"  {provenance}: {b['new']} new, {b['updated']} updated, "
                f"{b['skipped']} skipped"
            )
    if music_total:
        typer.echo(f"  music: {music_total} source(s) processed")
    if fb_posts_total or fb_msg_total or fb_group_total:
        typer.echo(
            f"  facebook: {fb_posts_total} posts-year(s) / "
            f"{fb_msg_total} message-thread(s) / "
            f"{fb_group_total} group(s) processed"
        )
    if journal_total:
        typer.echo(f"  journal: {journal_total} entry(ies) processed")


def _ingest_one(
    doc,
    settings,
    client,
    bucket: dict[str, int],
    totals: dict[str, int],
    loader=None,
    idx: int | None = None,
) -> None:
    from mission_brain.ingest.idempotence import write_manifest_sidecar
    from mission_brain.ingest.pipeline import (
        compute_source_manifest,
        ingest_source,
        manifest_matches_sidecar,
        sidecar_path_for,
    )
    from mission_brain.wiki.index_md import update_index
    from mission_brain.wiki.log_md import append_log
    from mission_brain.wiki.store import read_page, slugify, write_page

    if idx is not None:
        typer.echo(f"  [{idx}] processing {doc.source_id}")

    prior_slug = _slug_for_source(doc.source_id, settings.vault_root)
    prior_page = (
        read_page(prior_slug, settings.vault_root) if prior_slug else None
    )

    # CLAUDE.md §2.2 short-circuit: compute the manifest that fingerprints
    # every input feeding this page. If the sidecar next to an existing
    # wiki page already holds the same canonical bytes, the LLM call is
    # redundant — skip it.
    claude_md_text = _read_claude_md_for_manifest()
    loader_version = (
        getattr(loader, "version", None)
        or doc.loader_version
        or ""
    )
    overlay_path = (
        settings.vault_root / "wiki" / f"{prior_slug}.user-edits.md"
        if prior_slug
        else None
    )
    manifest = compute_source_manifest(
        doc,
        loader_version,
        _settings_config_digest(settings),
        overlay_path,
        claude_md_text,
    )
    if prior_slug is not None and prior_page is not None:
        cached_sidecar = sidecar_path_for(settings.vault_root, prior_slug)
        if manifest_matches_sidecar(cached_sidecar, manifest):
            bucket["skipped"] += 1
            totals["skipped"] += 1
            append_log(settings.vault_root, doc.source_id, "skipped")
            if idx is not None:
                typer.echo(f"  [{idx}] {doc.source_id} -> skipped")
            return

    try:
        page = ingest_source(doc, prior_page, client, claude_md_text)
    except Exception as exc:
        typer.echo(f"error ingesting {doc.source_path}: {exc}", err=True)
        bucket["skipped"] += 1
        totals["skipped"] += 1
        append_log(settings.vault_root, doc.source_id, "skipped")
        if idx is not None:
            typer.echo(f"  [{idx}] {doc.source_id} -> skipped")
        return

    slug = slugify(page.title)
    exists = (settings.vault_root / "wiki" / f"{slug}.md").exists()
    write_page(page, settings.vault_root)
    update_index(settings.vault_root, page)
    write_manifest_sidecar(manifest, sidecar_path_for(settings.vault_root, slug))
    if exists:
        bucket["updated"] += 1
        totals["updated"] += 1
        append_log(settings.vault_root, doc.source_id, "updated")
        if idx is not None:
            typer.echo(f"  [{idx}] {doc.source_id} -> updated")
    else:
        bucket["new"] += 1
        totals["new"] += 1
        append_log(settings.vault_root, doc.source_id, "created")
        if idx is not None:
            typer.echo(f"  [{idx}] {doc.source_id} -> created")


def _ingest_single_file(file_path: Path, settings, client) -> None:
    """Ingest exactly one source file by path — CLI --file flag."""
    from mission_brain.loaders.base import canonical_source_id
    from mission_brain.loaders.journal import MissionBulletLoader
    from mission_brain.loaders.music_data import MusicDataLoader
    from mission_brain.loaders.plain_markdown import PlainMarkdownLoader

    if not file_path.is_file():
        typer.echo(f"error: {file_path} is not a file", err=True)
        raise typer.Exit(code=2)

    # Facebook inference — kind-by-subdir dispatch. `--file` on FB is a
    # smoke-test handle, so threshold gating is relaxed (min 1) to
    # guarantee the target slice is ingested regardless of volume.
    parts = [p.name for p in file_path.parents]
    if "facebook" in parts:
        _ingest_facebook_single(file_path, parts, settings, client)
        return

    suffix = file_path.suffix.lower()
    parts_set = set(parts)
    if suffix == ".json":
        loader = MusicDataLoader(provenance="primary")
    elif suffix in (".md", ".markdown") and "journal" in parts_set:
        # rb-036: journal corpus → MissionBulletLoader → journal-private
        # provenance (privacy floor). Hands the same loader the batch
        # walk uses so --file pilots match end-to-end behavior.
        loader = MissionBulletLoader(provenance="journal-private")
    elif suffix in (".md", ".markdown"):
        # Infer provenance from the containing subdir name if recognisable.
        provenance = "derived" if "plain-md-derived" in parts_set else "primary"
        loader = PlainMarkdownLoader(provenance=provenance)
    else:
        typer.echo(
            f"error: no loader for file extension {suffix!r}", err=True
        )
        raise typer.Exit(code=2)

    doc = loader.load(file_path)
    # Keep catalog's fixed source id consistent with batch walk.
    if suffix == ".json" and file_path.name == "catalog_analysis.json":
        doc.source_id = MUSIC_CATALOG_SOURCE_ID
    else:
        try:
            doc.source_id = canonical_source_id(file_path, settings.corpus_root)
        except ValueError:
            doc.source_id = file_path.stem

    totals = {"new": 0, "updated": 0, "skipped": 0}
    bucket = {"new": 0, "updated": 0, "skipped": 0}
    _ingest_one(doc, settings, client, bucket, totals, loader=loader)
    typer.echo(
        f"{totals['new']} new, {totals['updated']} updated, "
        f"{totals['skipped']} skipped"
    )


def _ingest_facebook_single(
    file_path: Path, parts: list[str], settings, client
) -> None:
    """Dispatch a --file invocation under corpus/facebook/ to the right loader.

    The three FB loaders aggregate (per-year, per-thread, per-group),
    so a single real file isn't addressable on its own; we infer the
    kind from the subdir and then either load the synthetic aggregate
    corresponding to that file's slice, or ingest every qualifying
    aggregate of that kind if a single slice is not inferrable.
    """
    from mission_brain.loaders.facebook import (
        FacebookGroupsLoader,
        FacebookMessagesLoader,
        FacebookPostsLoader,
    )

    totals = {"new": 0, "updated": 0, "skipped": 0}
    bucket = {"new": 0, "updated": 0, "skipped": 0}

    if "posts" in parts:
        # A real posts JSON doesn't map to a single year. Ingest ALL years
        # discovered under corpus/facebook/posts/.
        loader = FacebookPostsLoader(provenance="primary")
        for synth in loader.discover(settings.corpus_root):
            doc = loader.load(synth)
            _ingest_one(doc, settings, client, bucket, totals, loader=loader)
    elif "inbox" in parts:
        # A message file lives under inbox/<thread>/; ingest just that thread.
        # Relax threshold to 1 so --file works regardless of Ray-word count.
        # If the thread is oversized, discover yields multiple _thread_partN
        # paths — ingest each part.
        loader = FacebookMessagesLoader(min_words=1, provenance="primary")
        thread_dir = file_path.parent
        synths = [
            p
            for p in loader.discover(settings.corpus_root)
            if p.parent == thread_dir
        ]
        if not synths:
            synths = [thread_dir / "_thread.agg"]
        for synth in synths:
            doc = loader.load(synth)
            _ingest_one(doc, settings, client, bucket, totals, loader=loader)
    elif "groups" in parts:
        # groups/ currently has one known file (group_posts_and_comments.json);
        # ingest every qualifying group. Relax threshold so --file is useful
        # as a smoke test even when a group has few contributions.
        loader = FacebookGroupsLoader(min_contribs=1, provenance="primary")
        for synth in loader.discover(settings.corpus_root):
            doc = loader.load(synth)
            _ingest_one(doc, settings, client, bucket, totals, loader=loader)
    else:
        typer.echo(
            f"error: facebook --file expects a path under "
            f"posts/, messages/inbox/, or groups/; got {file_path}",
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo(
        f"{totals['new']} new, {totals['updated']} updated, "
        f"{totals['skipped']} skipped"
    )


@app.command("query")
def query_cmd(
    text: str = typer.Argument(..., help="Free-text query against wiki + corpus."),
    top_k: int = typer.Option(10, "--top-k", help="Number of retrieval hits to consider."),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit QueryResult as structured JSON (instead of markdown).",
    ),
    exclude_provenance: list[str] = typer.Option(
        [],
        "--exclude-provenance",
        help=(
            "Exclude results whose provenance matches any provided value. "
            "Repeatable: pass once per excluded value."
        ),
    ),
    snippet_words: int = typer.Option(
        60,
        "--snippet-words",
        help=(
            "With --json, truncate each raw_citation snippet to this many "
            "whitespace-delimited words (0 yields an empty snippet). Ignored "
            "when --full-snippets is set."
        ),
    ),
    full_snippets: bool = typer.Option(
        False,
        "--full-snippets",
        help="With --json, emit full citation excerpts (ignore --snippet-words).",
    ),
) -> None:
    """Run a retrieval query and print the QueryResult as markdown."""
    from mission_brain.query.engine import run_query
    from mission_brain.query.retriever import LanceDBRetriever

    settings = _load_settings()
    embedder = _build_embedder(settings)
    retriever = LanceDBRetriever(
        embedder, db_path=settings.vault_root / ".mission_brain-index"
    )
    result = run_query(
        text,
        settings.vault_root,
        settings.corpus_root,
        retriever=retriever,
        top_k=top_k,
    )

    filters_applied: list[str] = []
    if exclude_provenance and result.synthesized_page is not None:
        prov = result.synthesized_page.provenance
        if prov in exclude_provenance:
            filters_applied = [prov]
            result = result.model_copy(update={"synthesized_page": None, "raw_citations": []})

    if json_output:
        payload = {
            "query": text,
            "synthesized_page": (
                result.synthesized_page.model_dump(mode="json")
                if result.synthesized_page is not None
                else None
            ),
            "raw_citations": [
                {
                    "source_id": str(c.source_id),
                    "locator": c.locator.format(),
                    "snippet": (
                        c.excerpt
                        if full_snippets
                        else _truncate_snippet(c.excerpt, snippet_words)
                    ),
                    "score": None,
                    "provenance": (
                        result.synthesized_page.provenance
                        if result.synthesized_page is not None
                        else "primary"
                    ),
                }
                for c in result.raw_citations
            ],
            "filters_applied": filters_applied,
        }
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        typer.echo(_format_query_result(result))


# ---------------------------------------------------------------------------
# Internal hooks — tests monkeypatch these to inject mocks.


def _truncate_snippet(text: str, max_words: int) -> str:
    """Truncate ``text`` to the first ``max_words`` whitespace-delimited words.

    If ``max_words`` is zero or negative, returns an empty string. If the text
    has at most ``max_words`` words, returns ``text`` unchanged. Otherwise
    returns the first ``max_words`` words joined by a single space, plus
    ``" ..."``.
    """
    if max_words <= 0:
        return ""
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words]) + " ..."


def _load_settings():
    from mission_brain.config import Settings

    return Settings()


def _parse_ingest_backend_cli(raw: str | None) -> IngestBackend | None:
    """Return normalized backend, or None to use :attr:`Settings.ingest_backend`."""
    if raw is None or str(raw).strip() == "":
        return None
    v = str(raw).strip().lower()
    if v not in ("claude", "cursor"):
        raise typer.BadParameter(
            f"expected 'claude' or 'cursor', got {raw!r}",
            param_hint="--ingest-backend",
        )
    return cast(IngestBackend, v)


def _build_ingest_client(
    settings,
    *,
    ingest_backend: IngestBackend | None = None,
):
    backend = ingest_backend if ingest_backend is not None else settings.ingest_backend
    if backend == "cursor":
        from mission_brain.ingest.cursor_cli_client import CursorCLIClient

        return CursorCLIClient(workspace=Path.cwd())
    from mission_brain.ingest.claude_cli_client import ClaudeCLIClient

    binary = settings.claude_cli_bin or None
    return ClaudeCLIClient(binary=binary)


def _build_embedder(settings):
    from mission_brain.ingest.voyage_client import VoyageEmbedder

    return VoyageEmbedder(api_key=settings.voyage_api_key or None)


def _read_claude_md_for_manifest() -> str:
    """Return the bytes of ./CLAUDE.md as text, or '' if not present.

    CLAUDE.md §2.2 requires the schema file's content to participate in
    the ingest manifest so a schema edit invalidates every cached page.
    Read relative to the current working directory — the CLI is
    typically invoked from the repo root. Missing file is not fatal:
    tests and other invocations outside the repo root just fold an
    empty string into the digest.
    """
    path = Path("CLAUDE.md")
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _settings_config_digest(settings) -> str:
    """Thin shim — delegates to :func:`mission_brain.config.compute_config_digest`."""
    from mission_brain.config import compute_config_digest

    return compute_config_digest(settings)


def _slug_for_source(source_id: str, vault_root: Path) -> str | None:
    """Find an existing wiki-page slug that lists *source_id* in its frontmatter."""
    import yaml

    wiki_dir = vault_root / "wiki"
    if not wiki_dir.is_dir():
        return None
    for md_file in sorted(wiki_dir.glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        if not raw.startswith("---\n"):
            continue
        end = raw.find("\n---\n", 4)
        if end == -1:
            continue
        try:
            fm = yaml.safe_load(raw[4:end + 1])
        except yaml.YAMLError:
            continue
        if isinstance(fm, dict) and source_id in (fm.get("source_ids") or []):
            return md_file.stem
    return None


def _format_query_result(result) -> str:
    lines: list[str] = []
    page = result.synthesized_page
    if page is not None:
        lines.append(f"# {page.title}")
        lines.append("")
        for para in page.paragraphs:
            refs = " ".join(c.render() for c in para.citations)
            if refs:
                lines.append(f"{para.text} {refs}".rstrip())
            else:
                lines.append(para.text)
            lines.append("")
    else:
        lines.append("# (no synthesized page)")
        lines.append("")

    lines.append("## Citations")
    lines.append("")
    if not result.raw_citations:
        lines.append("_no citations_")
    else:
        for c in result.raw_citations:
            excerpt = c.excerpt.replace("\n", " ").strip()
            lines.append(f"- {c.render()} {excerpt}".rstrip())
    return "\n".join(lines)


if __name__ == "__main__":
    app()
