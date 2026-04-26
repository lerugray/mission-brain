"""Claude Code CLI client — synthesis via ``claude -p``.

Shells out to the user's installed Claude Code CLI and returns
its stdout. Uses the user's existing subscription instead of the
Anthropic API. No HTTP egress of our own; network_guard is not
invoked because the CLI handles its own transport.

The ``temperature`` and ``seed`` parameters of :meth:`generate`
are accepted for interface parity but ignored — ``claude -p``
does not expose them. Determinism is
preserved at a higher layer by the §2.2 manifest-hash
short-circuit, which skips the LLM entirely when inputs are
unchanged.
"""

from __future__ import annotations

import os
import shutil
import subprocess

__all__ = ["DEFAULT_MODEL", "ClaudeCLIClient", "ClaudeCLIError"]

DEFAULT_MODEL = "claude-sonnet-4-6"

# rayb-029: bumped from 600s after the 2026-04-24 evening watchdog run
# saw chrisrakowski_part1, madeleinemcmillan_part1, and the 2020 FB-posts
# threads exceed 600s. Two paths considered:
#   (a) bump DEFAULT_TIMEOUT to cover the longest threads
#   (b) detect oversized inputs (estimated tokens > N) and per-call route
#       through a 1M-context model variant via --model override
# (b) is more durable but adds surface area: token estimator, model
# allowlist, per-call routing logic, and a fallback ladder when the
# 1M-context model is rate-limited. (a) is one constant + one test.
# Going with (a) — Hammerstein "minimum viable durable" call. Revisit (b)
# only if 1200s also proves insufficient on real-world threads.
DEFAULT_TIMEOUT = 1200.0


class ClaudeCLIError(RuntimeError):
    """Raised when the ``claude`` CLI exits non-zero or can't be found."""


def _resolve_claude_binary() -> str:
    binary = os.environ.get("MISSION_BRAIN_CLAUDE_BIN")
    if binary:
        return binary
    for candidate in ("claude", "claude.cmd", "claude.exe"):
        found = shutil.which(candidate)
        if found:
            return found
    raise ClaudeCLIError(
        "could not locate the `claude` CLI on PATH. "
        "Install Claude Code or set MISSION_BRAIN_CLAUDE_BIN to its absolute path."
    )


class ClaudeCLIClient:
    def __init__(
        self,
        binary: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.binary = binary or _resolve_claude_binary()
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        """Invoke ``claude -p`` and return its stdout."""
        del temperature, seed
        chosen_model = model or os.environ.get(
            "MISSION_BRAIN_INGEST_MODEL", DEFAULT_MODEL
        )
        cmd = [
            self.binary,
            "-p",
            "--output-format", "text",
            "--model", chosen_model,
        ]
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ClaudeCLIError(f"failed to execute {self.binary!r}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCLIError(
                f"claude -p exceeded {self.timeout}s timeout"
            ) from exc
        if result.returncode != 0:
            raise ClaudeCLIError(
                f"claude -p exited {result.returncode}: "
                f"{(result.stderr or '').strip()[:500]}"
            )
        return result.stdout
