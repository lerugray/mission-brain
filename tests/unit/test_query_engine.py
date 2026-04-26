"""Tests for the query engine (rayb-010).

Mock embedder returns fixed vectors keyed by substring, so the
retriever + engine can be exercised end-to-end without a live
Ollama or network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_brain.query.engine import run_query
from mission_brain.query.retriever import LanceDBRetriever, _split_paragraphs
from mission_brain.schema import (
    Citation,
    LineRangeLocator,
    Paragraph,
    SourceId,
    WikiPage,
)
from mission_brain.schema.query_result import CoVisibilityViolation, QueryResult
from mission_brain.wiki.store import write_page

# ---------------------------------------------------------------------------
# Fixtures


class FixedVectorEmbedder:
    """Return a hand-picked vector for each text based on substring match.

    Each text is scored against a list of ``(needle, vector)`` pairs in
    declaration order; the first matching needle wins. A default
    vector is returned for misses. This keeps the cosine-similarity
    geometry of the tests explicit.
    """

    def __init__(self, mapping: list[tuple[str, list[float]]], default: list[float]) -> None:
        self.mapping = mapping
        self.default = default
        self.call_count = 0

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        self.call_count += 1
        out: list[list[float]] = []
        for text in texts:
            picked = self.default
            for needle, vec in self.mapping:
                if needle in text:
                    picked = vec
                    break
            out.append(list(picked))
        return out


def _seed_wiki(vault_root: Path, slug_title: str, paragraph_text: str, source_id: str) -> WikiPage:
    page = WikiPage(
        title=slug_title,
        source_ids=[SourceId(source_id)],
        paragraphs=[
            Paragraph(
                text=paragraph_text,
                citations=[
                    Citation(
                        source_id=SourceId(source_id),
                        locator=LineRangeLocator(start=1, end=5),
                        excerpt="seeded excerpt",
                    )
                ],
            )
        ],
    )
    write_page(page, vault_root)
    return page


def _seed_corpus(corpus_root: Path, name: str, body: str) -> Path:
    pm = corpus_root / "plain-md"
    pm.mkdir(parents=True, exist_ok=True)
    path = pm / f"{name}.md"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# _split_paragraphs


def test_split_paragraphs_basic() -> None:
    text = "alpha line one\nalpha line two\n\nbeta only\n\n\ngamma last\n"
    chunks = _split_paragraphs(text)
    assert chunks == [
        (1, 2, "alpha line one\nalpha line two"),
        (4, 4, "beta only"),
        (7, 7, "gamma last"),
    ]


def test_split_paragraphs_empty() -> None:
    assert _split_paragraphs("") == []
    assert _split_paragraphs("   \n  \n") == []


# ---------------------------------------------------------------------------
# Wiki hit above threshold


def test_wiki_hit_returns_synthesized_page(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    _seed_wiki(
        vault,
        slug_title="Asymmetric Objectives",
        paragraph_text="A wiki paragraph about asymmetric objectives.",
        source_id="plain-md-journal-2016-09",
    )
    _seed_corpus(corpus, "raw-note", "Raw corpus content about scoring.\n")

    embedder = FixedVectorEmbedder(
        mapping=[
            ("wiki paragraph", [1.0, 0.0]),
            ("Raw corpus", [0.0, 1.0]),
            ("asymmetric", [1.0, 0.0]),
        ],
        default=[0.5, 0.5],
    )
    retriever = LanceDBRetriever(embedder, db_path=tmp_path / "lancedb")
    result = run_query(
        "asymmetric",
        vault,
        corpus,
        retriever=retriever,
        wiki_threshold=0.5,
    )

    assert isinstance(result, QueryResult)
    assert result.synthesized_page is not None
    assert result.synthesized_page.title == "Asymmetric Objectives"
    assert len(result.raw_citations) >= 1


# ---------------------------------------------------------------------------
# Wiki hit below threshold falls through to raw citations


def test_wiki_hit_below_threshold_returns_raw(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    _seed_wiki(
        vault,
        slug_title="Unrelated Page",
        paragraph_text="Content about something different.",
        source_id="plain-md-other",
    )
    _seed_corpus(corpus, "raw-note", "Raw scoring notes and tactics.\n")

    embedder = FixedVectorEmbedder(
        mapping=[
            ("scoring", [1.0, 0.0]),
            ("something different", [0.0, 1.0]),
        ],
        default=[0.0, 1.0],
    )
    retriever = LanceDBRetriever(embedder, db_path=tmp_path / "lancedb")
    result = run_query(
        "scoring",
        vault,
        corpus,
        retriever=retriever,
        wiki_threshold=0.99,
    )

    assert result.synthesized_page is None
    assert len(result.raw_citations) >= 1
    assert all(str(c.source_id) == "plain-md-raw-note" for c in result.raw_citations)


# ---------------------------------------------------------------------------
# Empty vault + corpus


def test_empty_returns_empty_query_result(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    corpus = tmp_path / "corpus"
    vault.mkdir()
    corpus.mkdir()

    embedder = FixedVectorEmbedder(mapping=[], default=[0.1, 0.2])
    retriever = LanceDBRetriever(embedder, db_path=tmp_path / "lancedb")
    result = run_query(
        "anything",
        vault,
        corpus,
        retriever=retriever,
    )
    assert result.synthesized_page is None
    assert result.raw_citations == []


# ---------------------------------------------------------------------------
# Co-visibility invariant enforced at QueryResult level


def test_co_visibility_violation_rejected() -> None:
    page = WikiPage(
        title="X",
        source_ids=[SourceId("src-a")],
        paragraphs=[
            Paragraph(
                text="T",
                citations=[
                    Citation(
                        source_id=SourceId("src-a"),
                        locator=LineRangeLocator(start=1, end=1),
                        excerpt="e",
                    )
                ],
            )
        ],
    )
    with pytest.raises(CoVisibilityViolation):
        QueryResult(synthesized_page=page, raw_citations=[])


# ---------------------------------------------------------------------------
# Module boundary: query/ does not import a generative LLM client


def test_query_package_no_generative_import() -> None:
    import mission_brain.query as qpkg

    for attr in dir(qpkg):
        if attr.startswith("_"):
            continue
        obj = getattr(qpkg, attr)
        mod_name = getattr(obj, "__module__", "") or ""
        assert "ingest.ollama_client" not in mod_name
        assert "ingest.pipeline" not in mod_name
