"""Source-type adapters (skeleton.md §3).

Each loader implements :class:`SourceLoader` for a specific raw-source
shape (plain markdown, MIDI-JSON music, Facebook export, ...). New
source types register here so downstream dispatch (ingest prompt,
CLI walk) can discover them via a single import.
"""

from mission_brain.loaders.base import SourceDocument, SourceLoader
from mission_brain.loaders.facebook import (
    FacebookGroupsLoader,
    FacebookMessagesLoader,
    FacebookPostsLoader,
)
from mission_brain.loaders.music_data import MusicDataLoader
from mission_brain.loaders.plain_markdown import PlainMarkdownLoader

__all__ = [
    "FacebookGroupsLoader",
    "FacebookMessagesLoader",
    "FacebookPostsLoader",
    "MusicDataLoader",
    "PlainMarkdownLoader",
    "SourceDocument",
    "SourceLoader",
]
