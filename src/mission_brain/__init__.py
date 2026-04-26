"""mission-brain — retrieval-only second brain over your own writing.

Citation-grounded retrieval. Refuses to write unsourced claims.
BYOK for embeddings (Voyage cloud or Ollama local) and synthesis
(Anthropic, OpenRouter, Ollama).

Public template version of raybrain (lerugray's private instance).
Fork-and-customize: drop your corpus into corpus/, configure your
providers, run `mission-brain ingest` to build the wiki, then
`mission-brain query` to retrieve voice-bearing citation-grounded
results.
"""
