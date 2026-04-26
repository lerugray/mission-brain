# corpus/

Drop your writing here. mission-brain reads everything in this
directory tree, runs it through configured loaders, and produces a
queryable wiki under `../vault/wiki/`.

This directory is gitignored (`.gitkeep` is the only tracked file).
Your raw writing never gets committed by this repo. Track your
corpus in a separate private repo if you want version control on
the source material itself.

## Supported source shapes (out of the box)

mission-brain ships with reference loaders for these layouts:

- **Plain markdown** — `corpus/notes/**/*.md` (any nesting)
- **Facebook export** — `corpus/facebook/<export-zip>` (drop the
  zip here, the loader unpacks + ingests posts/messages)
- **Bullet journal entries** — `corpus/journal/YYYY/MM/DD.md`
  (matches the [mission-bullet-oss](https://github.com/lerugray/mission-bullet-oss)
  layout, opt-in)
- **Music metadata sidecar** — `corpus/music/per-song/*.json` +
  `corpus/music/catalog/catalog_analysis.json` (catalog-style data)

## Adding a new loader

For source shapes mission-brain doesn't ship loaders for, write
your own following the loader protocol in
`src/mission_brain/loaders/__init__.py`. A loader yields
`SourceDocument` instances; one per logical unit you want
retrievable as a citation source.

The protocol is small (3 methods: discover, load, render). See the
shipped loaders in `src/mission_brain/loaders/` for reference
implementations across different source shapes.

## What gets indexed

- **Whole logical units** (a markdown file, a Facebook thread, a
  journal entry, a song's metadata) become `SourceDocument`s.
- Each is chunked, embedded, and stored in LanceDB.
- The wiki page is the synthesis output, not the corpus content
  itself.
- Citations in synthesized pages reference source documents by id
  plus locator (line number, paragraph index, etc).

## Privacy

mission-brain never uploads your corpus content. Embedding API
calls send chunked text to Voyage (if configured) or stay local
(if using Ollama). Synthesis API calls send retrieved chunks to
Anthropic / OpenRouter / Ollama (whichever you configured) along
with the user's query. The original files in this directory never
leave your disk.
