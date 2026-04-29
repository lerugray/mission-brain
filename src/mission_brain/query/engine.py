"""Query entry point — returns :class:`QueryResult`, never generates text.

The public ``run_query`` function combines retrieval hits into
the co-visibility-preserving :class:`QueryResult` shape required
by CLAUDE.md §2.4:

* If the top-1 hit is a wiki hit whose cosine similarity meets
  ``wiki_threshold``, return that page plus the citations it
  carries.
* Otherwise return ``None`` plus the raw-passage citations from
  the hit list.

The wiki-threshold default is deliberately permissive (0.0) —
the user tunes the value through the ``wiki_threshold`` argument or
wraps this function with their own gating policy. Threshold
*semantics* (cosine similarity, higher-is-better) are fixed by
CLAUDE.md's "top-1 hit exceeds a threshold" phrasing; the
*value* is a hyperparameter, not an algorithmic choice.
"""

from __future__ import annotations

from pathlib import Path

from mission_brain.query.retriever import Embedder, LanceDBRetriever, RetrievalHit
from mission_brain.schema import (
    Citation,
    LineRangeLocator,
    PageLocator,
    ParaAnchorLocator,
    QueryResult,
    SourceId,
    TimestampLocator,
)
from mission_brain.schema.citation import Locator
from mission_brain.wiki.store import read_page

__all__ = ["DEFAULT_WIKI_THRESHOLD", "run_query"]

DEFAULT_WIKI_THRESHOLD = 0.0


def _parse_locator(s: str) -> Locator:
    if s.startswith("lines="):
        start_s, end_s = s[len("lines="):].split("-", 1)
        return LineRangeLocator(start=int(start_s), end=int(end_s))
    if s.startswith("page="):
        return PageLocator(page=int(s[len("page="):]))
    if s.startswith("timestamp="):
        rest = s[len("timestamp="):]
        return TimestampLocator(start=rest[:8], end=rest[9:17])
    if s.startswith("para="):
        return ParaAnchorLocator(anchor=s[len("para="):])
    raise ValueError(f"unknown locator format: {s!r}")


def _hit_to_citation(hit: RetrievalHit) -> Citation:
    # Coerce only at the retrieval boundary so older indexes with
    # non-conformant source_ids do not crash reads. New ids remain strict.
    return Citation(
        source_id=SourceId.coerce(hit.source_id),
        locator=_parse_locator(hit.locator),
        excerpt=hit.excerpt,
    )


def run_query(
    query_text: str,
    vault_root: Path,
    corpus_root: Path,
    *,
    embedder: Embedder | None = None,
    retriever: LanceDBRetriever | None = None,
    top_k: int = 10,
    wiki_threshold: float = DEFAULT_WIKI_THRESHOLD,
    db_path: Path | None = None,
) -> QueryResult:
    """Retrieve wiki + raw hits and assemble a :class:`QueryResult`."""

    if retriever is None:
        if embedder is None:
            raise ValueError(
                "run_query requires an embedder when no retriever is supplied"
            )
        effective_db_path = db_path or (vault_root / ".mission_brain-index")
        retriever = LanceDBRetriever(embedder, db_path=effective_db_path)

    if not retriever.is_built():
        retriever.build_index(vault_root, corpus_root)

    hits = retriever.query(query_text, top_k=top_k)
    if not hits:
        return QueryResult(synthesized_page=None, raw_citations=[])

    raw_citations = [_hit_to_citation(h) for h in hits if h.kind == "raw"]

    top = hits[0]
    if top.kind == "wiki" and top.score >= wiki_threshold and top.wiki_page_slug:
        page = read_page(top.wiki_page_slug, vault_root)
        if page is not None:
            page_citations = [
                c for para in page.paragraphs for c in para.citations
            ]
            if page_citations:
                return QueryResult(
                    synthesized_page=page, raw_citations=page_citations
                )

    return QueryResult(synthesized_page=None, raw_citations=raw_citations)
