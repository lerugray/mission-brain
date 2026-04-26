"""Ingest pipeline for mission-brain (corpus → wiki synthesis).

The ingest package holds the mechanisms that enforce CLAUDE.md §2
invariants at write time. See `citation_floor` for §2.1 (every
non-empty paragraph carries a `[ref:...]` anchor).
"""
