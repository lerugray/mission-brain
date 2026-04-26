# mission-brain

> **DRAFT — final voice pass pending before public push.**
> This README is a placeholder structure. The hero, privacy callout,
> and differentiators paragraph need a final voice-pass before this
> repo flips public per the strategy doc. See
> `docs/internal/STRATEGY-2026-04-26.md` (in raybrain) for the
> design spec.

Retrieval-only second brain over your own writing. Citation-grounded
synthesis. Your corpus, your keys, your control.

## What it does

Drop your writing into `corpus/`, configure your `.env` with API
keys for your chosen providers, run `mission-brain ingest` to build
a queryable wiki, then `mission-brain query` to retrieve
citation-grounded results.

Every paragraph in a query response carries a
`[ref:source_id:locator]` marker pointing at the exact passage in
your corpus. No unsourced claims; the audit trail is the contract.

## Privacy posture

Your corpus stays on disk. mission-brain never uploads source text
in bulk. Embeddings hit Voyage **only** if you provide your own
Voyage API key. Synthesis hits Anthropic **only** if you provide
your own Anthropic API key. The Ollama path is fully local —
nothing leaves your machine.

Default config is the most-private mode (Ollama for both
embeddings and synthesis). Cloud features are opt-in via API keys,
never inferred.

## Quickstart

```bash
git clone https://github.com/<your-username>/mission-brain.git
cd mission-brain
cp .env.example .env  # configure providers
# drop your writing into corpus/
mission-brain ingest
mission-brain query "what have I written about X?"
```

## Differentiators (what mission-brain does that other tools don't)

- **Retrieval-only.** Returns cited passages. The user composes
  the final draft. Other RAG tools generate text on your behalf;
  mission-brain doesn't.
- **Citation-mandatory.** Every passage carries a
  `[ref:source_id:locator]` marker. The audit trail back to your
  actual text is enforced; output without markers fails
  validation and is rejected.
- **Voice-preserving.** Synthesis prompts are calibrated NOT to
  replace your phrasing. Retrieved passages carry their original
  tone forward.
- **Self-hosted, vendor-portable.** Your data lives in plain
  markdown plus LanceDB; portable to any other tool you choose
  later.

## Stack

- Python 3.12+
- pydantic, typer, llama-index, lancedb, python-frontmatter
- Voyage (cloud embeddings, optional) / Ollama (local embeddings)
- Anthropic / OpenRouter / Ollama for synthesis
- pytest for tests

## Sibling tools

mission-brain is one of a small family of personal-data tools that
share a design floor: your data on your disk, your keys for paid
providers, no SaaS layer.

- **[GeneralStaff](https://github.com/lerugray/generalstaff)** —
  multi-project bot orchestrator with hands-off enforcement and
  audit logging.
- **[mission-bullet](https://github.com/lerugray/mission-bullet)** —
  AI-assisted bullet journal, daily-capture and weekly-review.
  Companion for "what am I thinking right now" capture while
  mission-brain handles "what have I written about X" retrieval.
- **[mission-swarm](https://github.com/lerugray/mission-swarm)** —
  swarm-simulation engine for plausible audience reactions, useful
  for rehearsing how your writing might land before publishing.

## License

[AGPL-3.0-or-later](LICENSE).
