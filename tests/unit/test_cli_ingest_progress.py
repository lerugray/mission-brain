"""Per-source progress lines for batch ``mission-brain ingest`` (rb-045)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mission_brain import cli
from mission_brain.config import Settings

runner = CliRunner()


class _StubIngestClient:
    """Returns valid wiki markdown; keys match path stems searched in the prompt."""

    def __init__(self, canned: dict[str, str]) -> None:
        self.canned = canned

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        for stem, markdown in self.canned.items():
            if stem in prompt:
                return markdown
        raise AssertionError(f"unexpected prompt: {prompt[:200]!r}")


def _canned_page(title: str, ref_stem: str) -> str:
    return (
        f"# {title}\n\n"
        f"Synthesized line. [ref:{ref_stem}:lines=1-2]\n"
    )


def test_ingest_cmd_emits_per_source_progress_lines(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    vault = tmp_path / "vault"
    pm = corpus / "plain-md"
    pm.mkdir(parents=True)
    (pm / "one.md").write_text("# One\n\nBody one.\n", encoding="utf-8")
    (pm / "two.md").write_text("# Two\n\nBody two.\n", encoding="utf-8")

    settings = Settings(corpus_root=corpus, vault_root=vault)
    monkeypatch.setattr(cli, "_load_settings", lambda: settings)

    client = _StubIngestClient(
        canned={
            "Body one": _canned_page("Page Alpha", "plain-md-one"),
            "Body two": _canned_page("Page Beta", "plain-md-two"),
        }
    )
    monkeypatch.setattr(cli, "_build_ingest_client", lambda _s, **_k: client)

    result = runner.invoke(cli.app, ["ingest"])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "  [1] processing plain-md-one" in out
    assert "  [1] plain-md-one -> created" in out
    assert "  [2] processing plain-md-two" in out
    assert "  [2] plain-md-two -> created" in out
    assert "2 new, 0 updated, 0 skipped" in out
