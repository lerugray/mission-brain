"""Ingest prompt builder (CLAUDE.md §3.1 step 4).

Assembles schema + optional prior wiki page + source doc into a
single in-context prompt for the ingest LLM.

Dispatches on ``source_doc.source_type`` so structured source types
(e.g. MIDI-JSON music data, Facebook export JSON) can use a prompt
tailored to their citation grammar and synthesis shape — the default
``plain_markdown`` branch preserves the original prompt text
verbatim so existing golden-string tests keep passing.
"""

from __future__ import annotations

from mission_brain.loaders.base import SourceDocument
from mission_brain.schema.wiki_page import WikiPage

__all__ = [
    "build_facebook_groups_ingest_prompt",
    "build_facebook_messages_ingest_prompt",
    "build_facebook_posts_ingest_prompt",
    "build_ingest_prompt",
    "build_music_ingest_prompt",
    "build_plain_markdown_ingest_prompt",
]

_SECTION = "=" * 60

_CATALOG_FILENAME = "catalog_analysis.json"

# Facebook prompts share a preamble block: political fidelity, no AI-ethics
# hedging, citation-floor reminder, locator grammar. Factored out so every
# FB source-type sees the same verbatim directive.
_FACEBOOK_PREAMBLE = (
    "This source is a slice of Ray's Facebook export. Ray authored the\n"
    "posts/messages/group contributions it contains. The synthesis must\n"
    "serve as a voice/opinion retrieval surface for downstream creative\n"
    "work, which constrains how you may paraphrase.\n"
    "\n"
    "PRESERVE VERBATIM QUOTES. When Ray writes about Israel, Zionism,\n"
    "Netanyahu, antisemitism framing, or any other politically charged\n"
    "subject, quote his words directly. Do NOT sanitize, soften, or\n"
    "editorialize. Do NOT add \"while some may view X as...\" or similar\n"
    "framings. Ray's stated political identity is load-bearing context\n"
    "(CLAUDE.md §0.2 principle 1) and is entitled to verbatim retrieval\n"
    "fidelity.\n"
    "\n"
    "AI-ETHICS HEDGING IS OFF-TOPIC. Do not append caveats about how\n"
    "the reader should evaluate Ray's views, what external observers\n"
    "might think, or whether his stances are appropriate. Ray stated\n"
    "(CLAUDE.md §0.2 principle 2) that the ethical dimension of AI\n"
    "use is not open for re-litigation in these outputs. Apply the\n"
    "same rule to his political content: synthesize what he said and\n"
    "how, not what others might make of it.\n"
)


def _render_prior_page(page: WikiPage) -> str:
    lines = [f"# {page.title}", ""]
    for para in page.paragraphs:
        lines.append(para.text)
        if para.citations:
            lines.append(" ".join(c.render() for c in para.citations))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _schema_block(claude_md_text: str) -> str:
    return f"{_SECTION}\nSCHEMA (CLAUDE.md)\n{_SECTION}\n{claude_md_text}\n"


def _prior_block(prior_wiki_page: WikiPage | None) -> str | None:
    if prior_wiki_page is None:
        return None
    return (
        f"{_SECTION}\nPRIOR WIKI PAGE (update, do not re-synthesize)\n"
        f"{_SECTION}\n{_render_prior_page(prior_wiki_page)}\n"
    )


def _source_block(source_doc: SourceDocument) -> str:
    sid = source_doc.source_id or source_doc.source_path.stem
    return (
        f"{_SECTION}\nSOURCE DOCUMENT (source_id={sid})\n"
        f"{_SECTION}\ntitle: {source_doc.title}\n\n{source_doc.body}\n"
    )


def build_plain_markdown_ingest_prompt(
    source_doc: SourceDocument,
    prior_wiki_page: WikiPage | None,
    claude_md_text: str,
) -> str:
    """Plain-markdown prompt — byte-identical to the pre-rayb-014-c output."""
    parts: list[str] = [_schema_block(claude_md_text)]
    prior = _prior_block(prior_wiki_page)
    if prior is not None:
        parts.append(prior)
    sid = source_doc.source_id or source_doc.source_path.stem
    parts.append(
        f"{_SECTION}\nSOURCE DOCUMENT (source_id={sid})\n"
        f"{_SECTION}\ntitle: {source_doc.title}\n\n{source_doc.body}\n"
    )
    parts.append(
        f"{_SECTION}\nTASK\n{_SECTION}\n"
        "Emit a wiki page in markdown. Every non-empty paragraph must\n"
        f"carry at least one [ref:{sid}:<locator>] marker — use the\n"
        f"literal source_id '{sid}' shown above. Locator grammar:\n"
        "lines=N-M | page=N | timestamp=HH:MM:SS-HH:MM:SS | para=<anchor>.\n"
        "Headings and code blocks are exempt. Output raw markdown only —\n"
        "no preamble, no commentary, and DO NOT wrap the whole response\n"
        "in a ```markdown ... ``` code fence.\n"
    )
    return "\n".join(parts)


def _is_music_catalog(source_doc: SourceDocument) -> bool:
    """Catalog aggregate vs per-song MIDI-JSON.

    The catalog loader sets ``frontmatter['catalog_keys']`` and the path
    name is fixed at ``catalog_analysis.json`` — either signal suffices.
    """
    if source_doc.source_path.name == _CATALOG_FILENAME:
        return True
    return "catalog_keys" in (source_doc.frontmatter or {})


def build_music_ingest_prompt(
    source_doc: SourceDocument,
    prior_wiki_page: WikiPage | None,
    claude_md_text: str,
) -> str:
    """Music-data prompt — structured MIDI-JSON over per-song or catalog."""
    parts: list[str] = [_schema_block(claude_md_text)]
    prior = _prior_block(prior_wiki_page)
    if prior is not None:
        parts.append(prior)
    sid = source_doc.source_id or source_doc.source_path.stem
    parts.append(
        f"{_SECTION}\nSOURCE DOCUMENT (source_id={sid})\n"
        f"{_SECTION}\ntitle: {source_doc.title}\n\n{source_doc.body}\n"
    )

    if _is_music_catalog(source_doc):
        locator_instruction = (
            f"Every non-empty paragraph MUST carry at least one\n"
            f"[ref:{sid}:para=<key>] marker whose ``<key>`` is a top-level\n"
            f"key from the catalog JSON (e.g. scale_fingerprint,\n"
            f"chord_vocabulary, top_motifs, register_usage,\n"
            f"rhythm_distribution, catalog_stats). Use the literal\n"
            f"source_id '{sid}' shown above.\n"
        )
    else:
        locator_instruction = (
            f"Every non-empty paragraph MUST carry at least one\n"
            f"[ref:{sid}:time=HH:MM:SS.fff-HH:MM:SS.fff] marker whose time\n"
            f"range points into the body's time-labeled lines above.\n"
            f"Use fractional-second precision (three digits after the\n"
            f"decimal). Use the literal source_id '{sid}' shown above.\n"
        )

    parts.append(
        f"{_SECTION}\nTASK (music source)\n{_SECTION}\n"
        "Emit a wiki page in markdown with these sections, in this order,\n"
        "as ATX headings:\n"
        "  ## Tempo and meter\n"
        "  ## Harmonic language\n"
        "  ## Rhythm and groove\n"
        "  ## Structural arc\n"
        "  ## Notable motifs\n"
        "\n"
        "Tempo and meter covers tempo BPM, time signatures, and any\n"
        "meter changes. Harmonic language covers key/scale estimates and\n"
        "chord vocabulary frequency. Rhythm and groove covers note\n"
        "density and rhythmic motifs. Structural arc covers intro /\n"
        "verse / chorus / outro-style sections detectable from density\n"
        "or repetition. Notable motifs covers pitch-interval patterns\n"
        "that repeat.\n"
        "\n"
        f"{locator_instruction}"
        "\n"
        "DO NOT invent bar= locators — bar-level grids are deferred to\n"
        "Phase 3+ of mission-brain and are NOT available for this source.\n"
        "Headings and code blocks are exempt from the citation floor.\n"
        "Output raw markdown only — no preamble, no commentary, and DO\n"
        "NOT wrap the whole response in a ```markdown ... ``` code fence.\n"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Facebook prompts


def _build_facebook_prompt(
    source_doc: SourceDocument,
    prior_wiki_page: WikiPage | None,
    claude_md_text: str,
    task_header: str,
    sections: tuple[str, ...],
    section_intro: str,
    locator_label: str,
) -> str:
    """Shared renderer for all three FB source types.

    ``sections`` is the ordered list of ATX-heading section titles the
    synthesized page must contain. ``locator_label`` is the human-readable
    name of the anchor value inside ``para=<...>`` so the task text can
    name e.g. ``para=<post_id>`` or ``para=<timestamp_ms>``.
    """
    parts: list[str] = [_schema_block(claude_md_text)]
    prior = _prior_block(prior_wiki_page)
    if prior is not None:
        parts.append(prior)
    sid = source_doc.source_id or source_doc.source_path.stem
    parts.append(_source_block(source_doc))

    section_lines = "\n".join(f"  ## {s}" for s in sections)
    parts.append(
        f"{_SECTION}\n{task_header}\n{_SECTION}\n"
        f"{_FACEBOOK_PREAMBLE}\n"
        "Emit a wiki page in markdown with these sections, in this order,\n"
        "as ATX headings:\n"
        f"{section_lines}\n"
        "\n"
        f"{section_intro}\n"
        "\n"
        "Every non-empty paragraph MUST carry at least one\n"
        f"[ref:{sid}:para=<{locator_label}>] marker whose <" f"{locator_label}>\n"
        "is pulled directly from the body text above (the [post_id=...] /\n"
        "[msg_id=...] lines preceding each entry). Use the literal\n"
        f"source_id '{sid}' shown above. Headings and code blocks are\n"
        "exempt from the citation floor.\n"
        "\n"
        "DO NOT invent lines=, time=, bar=, page=, or timestamp= locators\n"
        "for this source — FB posts/messages/groups use para=<anchor>\n"
        "only. DO NOT fabricate anchor values; every <" f"{locator_label}>\n"
        "must correspond to a real entry you quoted or referenced from\n"
        "the source body.\n"
        "\n"
        "Output raw markdown only — no preamble, no commentary, and DO\n"
        "NOT wrap the whole response in a ```markdown ... ``` code fence.\n"
    )
    return "\n".join(parts)


def build_facebook_posts_ingest_prompt(
    source_doc: SourceDocument,
    prior_wiki_page: WikiPage | None,
    claude_md_text: str,
) -> str:
    return _build_facebook_prompt(
        source_doc,
        prior_wiki_page,
        claude_md_text,
        task_header="TASK (facebook posts)",
        sections=(
            "Recurring themes",
            "Political commentary",
            "Creative projects",
            "Personal events",
            "Notable verbatim posts",
        ),
        section_intro=(
            "Recurring themes covers topics Ray returned to across multiple\n"
            "posts in this year. Political commentary covers posts about\n"
            "politics, current events, and social issues — preserve verbatim\n"
            "quotes per the preamble. Creative projects covers his music,\n"
            "wargame design, and other creative work. Personal events covers\n"
            "life events, relationships, and travel. Notable verbatim posts\n"
            "lifts 3-8 posts that are the strongest voice samples for later\n"
            "retrieval — quote them in full, not in summary."
        ),
        locator_label="post_id",
    )


def build_facebook_messages_ingest_prompt(
    source_doc: SourceDocument,
    prior_wiki_page: WikiPage | None,
    claude_md_text: str,
) -> str:
    return _build_facebook_prompt(
        source_doc,
        prior_wiki_page,
        claude_md_text,
        task_header="TASK (facebook messages)",
        sections=(
            "Relationship context",
            "Topics discussed",
            "Tone and register",
            "Notable verbatim exchanges",
        ),
        section_intro=(
            "Relationship context covers who Ray was talking to and the\n"
            "nature of the relationship (friend, collaborator, family,\n"
            "romantic, professional) inferrable from the thread. Topics\n"
            "discussed covers the substantive subjects of the conversation.\n"
            "Tone and register covers how Ray writes in this thread — his\n"
            "diction, humor, seriousness, in-jokes. Notable verbatim\n"
            "exchanges lifts 2-5 exchanges that best represent Ray's voice\n"
            "in this thread — quote both sides in full for each."
        ),
        locator_label="timestamp_ms",
    )


def build_facebook_groups_ingest_prompt(
    source_doc: SourceDocument,
    prior_wiki_page: WikiPage | None,
    claude_md_text: str,
) -> str:
    return _build_facebook_prompt(
        source_doc,
        prior_wiki_page,
        claude_md_text,
        task_header="TASK (facebook group)",
        sections=(
            "Group purpose",
            "Ray's role",
            "Recurring contributions",
            "Notable verbatim posts",
        ),
        section_intro=(
            "Group purpose covers what the group is about, inferrable from\n"
            "Ray's contributions and any context in the post titles. Ray's\n"
            "role covers how he participated — lurker, regular poster,\n"
            "organizer, provocateur. Recurring contributions covers topics\n"
            "Ray raised repeatedly in this group. Notable verbatim posts\n"
            "lifts 3-6 posts that best represent his voice in this group —\n"
            "quote them in full."
        ),
        locator_label="post_id",
    )


def build_ingest_prompt(
    source_doc: SourceDocument,
    prior_wiki_page: WikiPage | None,
    claude_md_text: str,
) -> str:
    """Dispatch to a source-type-specific prompt builder.

    Unknown ``source_type`` values fall back to the plain-markdown
    prompt — existing call sites that leave ``source_type`` empty
    continue to receive the original prompt text verbatim.
    """
    st = source_doc.source_type
    if st == "music_data":
        return build_music_ingest_prompt(source_doc, prior_wiki_page, claude_md_text)
    if st == "facebook_posts":
        return build_facebook_posts_ingest_prompt(
            source_doc, prior_wiki_page, claude_md_text
        )
    if st == "facebook_messages":
        return build_facebook_messages_ingest_prompt(
            source_doc, prior_wiki_page, claude_md_text
        )
    if st == "facebook_groups":
        return build_facebook_groups_ingest_prompt(
            source_doc, prior_wiki_page, claude_md_text
        )
    return build_plain_markdown_ingest_prompt(
        source_doc, prior_wiki_page, claude_md_text
    )
