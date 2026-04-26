# mission-brain — Mission

Retrieval-only second brain over your own writing. Citation-grounded
synthesis. BYOK. Open-source template you fork and run on your own
corpus.

The opinionated calls behind the design:

## What mission-brain IS

- **Retrieval-only.** It returns cited passages from your corpus.
  It never generates text on your behalf. Composition stays with
  you.
- **Citation-mandatory.** Every paragraph in a synthesized response
  carries a `[ref:source_id:locator]` marker pointing back to your
  actual writing. Output without markers fails validation and never
  reaches your vault.
- **Voice-preserving.** The synthesis prompts are calibrated NOT
  to replace your phrasing. The retrieved passages carry their
  original tone forward.
- **BYOK.** You provide embeddings (Voyage cloud or Ollama local)
  and synthesis (Anthropic, OpenRouter, or Ollama) credentials.
  No platform middleman, no data uploaded except for the API calls
  you configure.
- **Local-first.** Your corpus stays on disk. The vault output
  stays on disk. Nothing depends on a hosted service.

## What mission-brain is NOT

- **Not a chatbot.** No conversational layer, no memory of prior
  queries beyond what your shell history holds.
- **Not a writing assistant.** It surfaces what you've already
  written; it doesn't extend or paraphrase.
- **Not a search engine.** Embedding similarity matters but the
  goal is voice-coherent retrieval, not raw recall.
- **Not platform-locked.** Your data stays in plain markdown +
  LanceDB; portable to any other tool you choose later.

## Fork-and-customize philosophy

mission-brain is a template, not a package. Fork the repo, drop
your corpus in `corpus/`, configure your `.env`, write any custom
loaders you need, run `mission-brain ingest`, then query.

The shipped loaders cover several common shapes (plain markdown,
Facebook export, bullet journal, music metadata sidecar) but real
users will have one or two source shapes that need a custom
loader. The loader protocol is small (~30 lines) and forks are
expected.

## Architectural floors

These don't move:

1. **Citation floor.** Synthesized output without
   `[ref:source_id:locator]` markers fails validation and the
   wiki page is not written. Tested in
   `tests/unit/test_citation_floor.py`.
2. **No corpus content in commits.** `corpus/` is gitignored;
   `.gitkeep` is the only tracked file. Track your corpus in a
   separate private repo if you want version control on source
   material.
3. **No vault content in commits.** `vault/` is gitignored. The
   wiki is regenerable from the corpus + the same config; treat
   it as derived state.
4. **Provider abstraction.** Voyage, Ollama, Anthropic, OpenRouter
   are all behind a thin interface. Swapping providers is
   configuration, not code surgery.

## Sibling tools

- **[GeneralStaff](https://github.com/lerugray/generalstaff)**:
  multi-project bot orchestrator with hands-off enforcement and
  audit logging. mission-brain is a Mode B project in GS for its
  own build-out tracking.
- **[mission-bullet-oss](https://github.com/lerugray/mission-bullet-oss)**:
  AI-assisted bullet journal for daily capture and weekly review.
  Mission-bullet's daily entries are a natural mission-brain ingest
  source through an opt-in loader.
- **[mission-swarm](https://github.com/lerugray/mission-swarm)**:
  swarm-simulation engine that generates plausible audience
  reactions to a document. Sits next to mission-brain when you want
  to rehearse how a draft might land before you publish it.

All four tools follow the same posture: your data on your disk,
your keys for paid providers, no SaaS layer.
