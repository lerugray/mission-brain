"""Cursor-agent CLI client — synthesis via ``cursor-agent -p``.

Sibling to :mod:`mission_brain.ingest.claude_cli_client`. Implements the
same ``generate(prompt) -> str`` contract so :func:`pipeline.ingest_source`
treats the two interchangeably. Runs cursor-agent inside WSL because
that's where Ray's authenticated Cursor Pro install lives (the Windows
build is GUI-only).

Default model is ``composer-2-fast`` — Cursor's free-unlimited tier.
The generation pattern is "prompt-in via stdin file, markdown-out via
stdout"; cursor-agent is asked NOT to edit any files in the workspace.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

__all__ = ["DEFAULT_MODEL", "CursorCLIClient", "CursorCLIError"]

DEFAULT_MODEL = "composer-2-fast"
DEFAULT_TIMEOUT = 1200.0


class CursorCLIError(RuntimeError):
    """Raised when ``cursor-agent`` exits non-zero or can't be reached."""


def _windows_to_wsl(path: str) -> str:
    """Translate ``C:\\foo`` or ``C:/foo`` to ``/mnt/c/foo`` for WSL.

    Mirrors ``scripts/cursor/cursor-launch.sh:to_wsl_path``.
    """
    p = path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"
    return p


def _resolve_wsl_binary() -> str:
    binary = os.environ.get("MISSION_BRAIN_WSL_BIN")
    if binary:
        return binary
    found = shutil.which("wsl") or shutil.which("wsl.exe")
    if found:
        return found
    raise CursorCLIError(
        "could not locate `wsl` on PATH. "
        "Set MISSION_BRAIN_WSL_BIN to the absolute path of wsl.exe."
    )


_REFUSAL_PREFIX = (
    "DO NOT edit, create, or delete any files in this workspace. DO NOT run\n"
    "shell commands. DO NOT use any tools — your only job is to print the\n"
    "requested markdown to stdout. The text after this preamble is the\n"
    "real task. Output the markdown response directly — no commentary,\n"
    "no preamble, no acknowledgement, no \"here's the markdown\" framing.\n"
    "Begin your response with the first character of the requested\n"
    "markdown.\n\n"
    "=" * 60 + "\n"
    "REAL TASK BEGINS BELOW\n"
    + "=" * 60 + "\n\n"
)


class CursorCLIClient:
    """Mirror of :class:`ClaudeCLIClient` that routes to cursor-agent."""

    def __init__(
        self,
        model: str | None = None,
        workspace: Path | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.wsl_binary = _resolve_wsl_binary()
        self.model = model or os.environ.get(
            "MISSION_BRAIN_CURSOR_MODEL", DEFAULT_MODEL
        )
        # Cursor needs a workspace; default to the cwd. The workspace is
        # only a sandboxing context — we don't ask cursor to touch it.
        self.workspace = workspace or Path.cwd()
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        """Invoke ``cursor-agent`` and return its stdout."""
        del temperature, seed
        chosen_model = model or self.model
        # Write the prompt to a temp file. Stdin pipes through PowerShell
        # → subprocess → wsl → bash get mangled by every shell-quoting
        # layer; a file is the only durable medium.
        tmp_dir = Path(tempfile.gettempdir())
        prompt_path = tmp_dir / f"mission-brain-cursor-{uuid.uuid4().hex}.txt"
        full_prompt = _REFUSAL_PREFIX + prompt
        prompt_path.write_text(full_prompt, encoding="utf-8")

        wsl_workspace = _windows_to_wsl(str(self.workspace))
        wsl_prompt = _windows_to_wsl(str(prompt_path))

        # bash -lc so cursor-agent inherits Ray's WSL login PATH
        # (~/.local/bin/cursor-agent isn't on the default non-login PATH).
        cmd_inner = (
            f"cd {wsl_workspace!s} && "
            f"cursor-agent -p --trust "
            f"--model {chosen_model} "
            f"--output-format text "
            f"< {wsl_prompt!s}"
        )
        cmd = [self.wsl_binary, "bash", "-lc", cmd_inner]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CursorCLIError(
                f"failed to execute {self.wsl_binary!r}: {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CursorCLIError(
                f"cursor-agent exceeded {self.timeout}s timeout"
            ) from exc
        finally:
            prompt_path.unlink(missing_ok=True)

        if result.returncode != 0:
            raise CursorCLIError(
                f"cursor-agent exited {result.returncode}: "
                f"{(result.stderr or '').strip()[:500]}"
            )
        return result.stdout
