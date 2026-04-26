"""Query engine — retrieval over wiki + raw corpus.

Per CLAUDE.md §2.4 + §3.2, the query path is retrieval-only. No
module in this package imports a generative LLM client; the
per-AST boundary guard lands in a later task
(``src/mission_brain/policy/query_gate.py``).
"""

from __future__ import annotations

from mission_brain.query.engine import run_query
from mission_brain.query.retriever import LanceDBRetriever, RetrievalHit

__all__ = [
    "LanceDBRetriever",
    "RetrievalHit",
    "run_query",
]
