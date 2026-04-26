"""Runtime configuration — env-var driven settings for CLI wiring.

Loads `.env` at the repo root (if present) into ``os.environ``
without clobbering values already exported. Exposes a pydantic
:class:`Settings` object. All fields have defaults so
``Settings()`` works in a fresh repo.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

IngestBackend = Literal["claude", "cursor"]

__all__ = [
    "IngestBackend",
    "JournalProvider",
    "Settings",
    "compute_config_digest",
    "load_dotenv",
]


def load_dotenv(path: Path | None = None) -> None:
    """Best-effort loader for a simple KEY=VALUE .env file.

    Does not override values already in ``os.environ``. Ignores
    blank lines and ``#`` comments. Strips matching single/double
    quotes around values. Silent if the file is missing.
    """
    if path is not None:
        candidates = [path]
    else:
        # Try cwd first (CLI invoked from repo root) then the repo
        # root derived from this file's location (handles subprocess
        # invocation from a different cwd, e.g. mission-brain.exe spawned
        # by an MCP server running in another project's directory).
        candidates = [Path(".env"), Path(__file__).resolve().parent.parent.parent / ".env"]
    for env_path in candidates:
        if env_path.is_file():
            break
    else:
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_dotenv()


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read *name* as a bool. Truthy: 1/true/yes/on; falsy: 0/false/no/off.

    Case-insensitive. Empty / unset / unrecognized -> default. Avoids
    pydantic's permissive str-coerces-to-True default for bool fields.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


JournalProvider = Literal["anthropic", "ollama"]


def _journal_provider_from_env() -> JournalProvider:
    """Read ``MISSION_BRAIN_JOURNAL_PROVIDER`` (default anthropic). Case-insensitive."""
    raw = os.environ.get("MISSION_BRAIN_JOURNAL_PROVIDER")
    if raw is None or str(raw).strip() == "":
        return "anthropic"
    v = str(raw).strip().lower()
    if v not in ("anthropic", "ollama"):
        raise ValueError(
            f"Invalid MISSION_BRAIN_JOURNAL_PROVIDER={raw!r}; expected 'anthropic' or 'ollama'."
        )
    return v  # type: ignore[return-value]


def _ingest_backend_from_env() -> IngestBackend:
    """Read ``MISSION_BRAIN_INGEST_BACKEND`` (default claude). Case-insensitive."""
    raw = os.environ.get("MISSION_BRAIN_INGEST_BACKEND")
    if raw is None or str(raw).strip() == "":
        return "claude"
    v = str(raw).strip().lower()
    if v not in ("claude", "cursor"):
        raise ValueError(
            f"Invalid MISSION_BRAIN_INGEST_BACKEND={raw!r}; expected 'claude' or 'cursor'."
        )
    return v  # type: ignore[return-value]


class Settings(BaseModel):
    corpus_root: Path = Field(
        default_factory=lambda: _env_path("MISSION_BRAIN_CORPUS_ROOT", "./corpus")
    )
    vault_root: Path = Field(
        default_factory=lambda: _env_path("MISSION_BRAIN_VAULT_ROOT", "./vault")
    )
    ingest_model: str = Field(
        default_factory=lambda: _env_str("MISSION_BRAIN_INGEST_MODEL", "claude-sonnet-4-6")
    )
    ingest_backend: IngestBackend = Field(default_factory=_ingest_backend_from_env)
    embed_model: str = Field(
        default_factory=lambda: _env_str("MISSION_BRAIN_EMBED_MODEL", "voyage-3-lite")
    )
    voyage_api_key: str = Field(
        default_factory=lambda: _env_str("VOYAGE_API_KEY", "")
    )
    claude_cli_bin: str = Field(
        default_factory=lambda: _env_str("MISSION_BRAIN_CLAUDE_BIN", "")
    )
    ollama_host: str = Field(
        default_factory=lambda: _env_str("OLLAMA_HOST", "http://localhost:11434")
    )
    fb_msg_min_words: int = Field(
        default_factory=lambda: _env_int("MISSION_BRAIN_FB_MSG_MIN_WORDS", 5000)
    )
    fb_group_min_contribs: int = Field(
        default_factory=lambda: _env_int("MISSION_BRAIN_FB_GROUP_MIN_CONTRIBS", 20)
    )
    # rb-035: journal-corpus ingest config (default ON; 2026-04-25
    # resolution). Privacy is enforced downstream by the
    # ``journal-private`` provenance tag — bot consumers default-
    # exclude it via --exclude-provenance even when ingestion runs.
    journal_enabled: bool = Field(
        default_factory=lambda: _env_bool("MISSION_BRAIN_JOURNAL_ENABLED", True)
    )
    journal_provider: JournalProvider = Field(
        default_factory=_journal_provider_from_env
    )
    journal_allow_cloud: bool = Field(
        default_factory=lambda: _env_bool("MISSION_BRAIN_JOURNAL_ALLOW_CLOUD", True)
    )


def compute_config_digest(settings: Settings) -> str:
    """Digest the subset of settings fields that affect synthesis output.

    CLAUDE.md §2.2 folds this into the ingest manifest so a config
    change whose effect is visible in a wiki page invalidates the
    cached sidecar. Fields unrelated to synthesis (e.g., API keys,
    host URLs) are deliberately excluded — including them would
    force wholesale re-ingest on any env shuffle.
    """
    payload = {
        "ingest_model": settings.ingest_model,
        "embed_model": settings.embed_model,
        "fb_msg_min_words": settings.fb_msg_min_words,
        "fb_group_min_contribs": settings.fb_group_min_contribs,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
