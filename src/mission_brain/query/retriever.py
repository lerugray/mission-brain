"""LanceDB retriever over the wiki layer + raw plain-markdown corpus.

Builds a single vector index over two candidate-hit shapes:

* **wiki** — one row per (paragraph, citation) pair inside each
  ``<vault_root>/wiki/*.md`` page. The paragraph text is the
  embedding input; the citation supplies the source_id / locator
  / excerpt. Each row also carries the wiki page's slug so the
  engine can fetch the full page when a wiki hit wins.
* **raw** — one row per paragraph-chunk of each
  ``<corpus_root>/plain-md/**/*.md`` file, chunked on blank-line
  boundaries. The source_id follows CLAUDE.md §1 convention
  (corpus-relative path, ``/`` → ``-``, extension stripped).

``query`` returns :class:`RetrievalHit` items sorted best-first
by cosine similarity (``score = 1 - cosine_distance``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import lancedb
import pyarrow as pa

from mission_brain.wiki.store import read_page

__all__ = ["Embedder", "LanceDBRetriever", "RetrievalHit"]


HitKind = Literal["wiki", "raw"]


@dataclass(frozen=True)
class RetrievalHit:
    source_id: str
    locator: str
    excerpt: str
    score: float
    kind: HitKind
    wiki_page_slug: str = ""


class Embedder(Protocol):
    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        ...


def _corpus_source_id(path: Path, corpus_root: Path) -> str:
    rel = path.resolve().relative_to(corpus_root.resolve())
    stem = rel.with_suffix("")
    return str(stem).replace("\\", "/").replace("/", "-").lower()


def _existing_table_names(db) -> list[str]:
    """Return existing table names regardless of LanceDB API shape.

    LanceDB's ``list_tables()`` historically returned a plain
    ``list[str]``; current versions return a ``ListTablesResponse``
    namedtuple-like with a ``.tables`` attribute. Direct ``in``
    checks against the response object always return False, which
    silently breaks both the table-exists check in
    :meth:`LanceDBRetriever.is_built` and the drop-before-create
    branch in :meth:`build_index`. This shim accepts both shapes.
    """
    raw = db.list_tables()
    if isinstance(raw, list):
        return raw
    tables_attr = getattr(raw, "tables", None)
    if isinstance(tables_attr, list):
        return tables_attr
    try:
        return list(raw)
    except TypeError:
        return []


def _split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """Split *text* on blank-line boundaries.

    Returns a list of ``(start_line, end_line, chunk)`` tuples,
    1-indexed inclusive. Empty or whitespace-only blocks are
    skipped. Matches the granularity of CLAUDE.md's paragraph
    concept and keeps the locator grammar ``lines=N-M`` honest.
    """
    lines = text.splitlines()
    chunks: list[tuple[int, int, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break
        start = i + 1
        block: list[str] = []
        while i < n and lines[i].strip() != "":
            block.append(lines[i])
            i += 1
        end = i
        chunk = "\n".join(block).strip()
        if chunk:
            chunks.append((start, end, chunk))
    return chunks


class LanceDBRetriever:
    """Build and query a LanceDB vector index over wiki + raw passages."""

    def __init__(
        self,
        embedder: Embedder,
        db_path: Path,
        table_name: str = "mission_brain_index",
    ) -> None:
        self.embedder = embedder
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._table = None

    def is_built(self) -> bool:
        """True if an index is cached in-memory OR exists on-disk.

        Without the on-disk check, every fresh `mission-brain query`
        process triggered a full embed-and-rebuild — wasteful at
        any scale, and racy on LanceDB drop+create when the
        previous on-disk table still existed (the ValueError
        "Table 'mission_brain_index' already exists" path). Looking
        for the table on disk and reattaching to it makes
        cross-process query reuse work; rebuild only happens when
        the index is genuinely missing or after a manual delete.

        Trade-off: the on-disk index is not invalidated when the
        wiki layer changes (no manifest sidecar like rb-027). A
        future task can wire content-hash-based invalidation; for
        now, an explicit `rm -rf vault/.mission_brain-index` is the
        forced-rebuild path.
        """
        if self._table is not None:
            return True
        if not self.db_path.exists():
            return False
        try:
            db = lancedb.connect(str(self.db_path))
            existing_tables = _existing_table_names(db)
            if self.table_name in existing_tables:
                self._table = db.open_table(self.table_name)
                return True
        except Exception:
            return False
        return False

    def build_index(self, vault_root: Path, corpus_root: Path) -> None:
        rows = self._collect_rows(vault_root, corpus_root)
        if not rows:
            self._table = None
            return

        texts = [r["text"] for r in rows]
        vectors = self.embedder.embed(texts)
        if len(vectors) != len(rows):
            raise ValueError(
                f"embedder returned {len(vectors)} vectors for {len(rows)} texts"
            )
        dim = len(vectors[0])
        for row, vec in zip(rows, vectors, strict=True):
            if len(vec) != dim:
                raise ValueError(
                    f"embedding dim mismatch: expected {dim}, got {len(vec)}"
                )
            row["vector"] = vec

        schema = pa.schema(
            [
                ("text", pa.string()),
                ("source_id", pa.string()),
                ("locator", pa.string()),
                ("excerpt", pa.string()),
                ("kind", pa.string()),
                ("wiki_page_slug", pa.string()),
                ("vector", pa.list_(pa.float32(), dim)),
            ]
        )

        self.db_path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(self.db_path))
        if self.table_name in _existing_table_names(db):
            db.drop_table(self.table_name)
        self._table = db.create_table(self.table_name, data=rows, schema=schema)

    def query(self, text: str, top_k: int = 10) -> list[RetrievalHit]:
        if self._table is None:
            return []
        vec = self.embedder.embed([text])[0]
        results = (
            self._table.search(vec).metric("cosine").limit(top_k).to_list()
        )
        hits: list[RetrievalHit] = []
        for row in results:
            distance = float(row.get("_distance", 1.0))
            hits.append(
                RetrievalHit(
                    source_id=row["source_id"],
                    locator=row["locator"],
                    excerpt=row["excerpt"],
                    score=1.0 - distance,
                    kind=row["kind"],
                    wiki_page_slug=row.get("wiki_page_slug", ""),
                )
            )
        return hits

    def _collect_rows(
        self, vault_root: Path, corpus_root: Path
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []

        wiki_dir = vault_root / "wiki"
        if wiki_dir.is_dir():
            for md_file in sorted(wiki_dir.glob("*.md")):
                page = read_page(md_file.stem, vault_root)
                if page is None:
                    continue
                for para in page.paragraphs:
                    if para.text.strip() == "":
                        continue
                    for cite in para.citations:
                        rows.append(
                            {
                                "text": para.text,
                                "source_id": str(cite.source_id),
                                "locator": cite.locator.format(),
                                "excerpt": cite.excerpt,
                                "kind": "wiki",
                                "wiki_page_slug": md_file.stem,
                            }
                        )

        plain_md_root = corpus_root / "plain-md"
        if plain_md_root.is_dir():
            for md_file in sorted(plain_md_root.rglob("*.md")):
                if not md_file.is_file():
                    continue
                text = md_file.read_text(encoding="utf-8")
                source_id = _corpus_source_id(md_file, corpus_root)
                for start, end, chunk in _split_paragraphs(text):
                    rows.append(
                        {
                            "text": chunk,
                            "source_id": source_id,
                            "locator": f"lines={start}-{end}",
                            "excerpt": chunk if len(chunk) <= 240 else chunk[:237] + "...",
                            "kind": "raw",
                            "wiki_page_slug": "",
                        }
                    )

        return rows
