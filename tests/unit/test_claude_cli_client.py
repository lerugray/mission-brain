"""Tests for :mod:`mission_brain.ingest.claude_cli_client` (rayb-029).

Covers the timeout-config path. Don't actually run a 1200s claude call
in tests — mock the subprocess and assert the timeout argument was
passed through correctly. Also guards the DEFAULT_TIMEOUT regression
(must be >= 1200s after rayb-029) and the timeout-exceeded error path.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from mission_brain.ingest.claude_cli_client import (
    DEFAULT_TIMEOUT,
    ClaudeCLIClient,
    ClaudeCLIError,
)


def test_default_timeout_at_least_1200s() -> None:
    """rayb-029 regression guard: long FB threads need >= 1200s."""
    assert DEFAULT_TIMEOUT >= 1200.0


def test_default_timeout_passed_to_subprocess() -> None:
    """When no per-call timeout override, DEFAULT_TIMEOUT reaches subprocess.run."""
    with patch(
        "mission_brain.ingest.claude_cli_client.subprocess.run"
    ) as mock_run, patch(
        "mission_brain.ingest.claude_cli_client._resolve_claude_binary",
        return_value="/fake/claude",
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/fake/claude"],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        client = ClaudeCLIClient()
        client.generate("hello")
        assert mock_run.called
        kwargs = mock_run.call_args.kwargs
        assert kwargs["timeout"] == DEFAULT_TIMEOUT


def test_explicit_timeout_overrides_default() -> None:
    """Per-instance timeout override is plumbed through to subprocess.run."""
    with patch(
        "mission_brain.ingest.claude_cli_client.subprocess.run"
    ) as mock_run, patch(
        "mission_brain.ingest.claude_cli_client._resolve_claude_binary",
        return_value="/fake/claude",
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/fake/claude"],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        client = ClaudeCLIClient(timeout=42.0)
        client.generate("hello")
        kwargs = mock_run.call_args.kwargs
        assert kwargs["timeout"] == 42.0


def test_timeout_expired_raises_claude_cli_error() -> None:
    """A subprocess.TimeoutExpired turns into ClaudeCLIError with the timeout value in the message."""
    with patch(
        "mission_brain.ingest.claude_cli_client.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=1200.0),
    ), patch(
        "mission_brain.ingest.claude_cli_client._resolve_claude_binary",
        return_value="/fake/claude",
    ):
        client = ClaudeCLIClient()
        with pytest.raises(ClaudeCLIError) as exc_info:
            client.generate("hello")
        # Error message should mention the configured timeout so an
        # operator skimming logs immediately knows whether to bump it.
        assert "1200" in str(exc_info.value)


def test_nonzero_exit_raises_claude_cli_error() -> None:
    """Non-zero subprocess return code raises ClaudeCLIError with stderr trimmed in."""
    with patch(
        "mission_brain.ingest.claude_cli_client.subprocess.run"
    ) as mock_run, patch(
        "mission_brain.ingest.claude_cli_client._resolve_claude_binary",
        return_value="/fake/claude",
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/fake/claude"],
            returncode=2,
            stdout="",
            stderr="upstream rate-limited",
        )
        client = ClaudeCLIClient()
        with pytest.raises(ClaudeCLIError) as exc_info:
            client.generate("hello")
        assert "exited 2" in str(exc_info.value)
        assert "upstream rate-limited" in str(exc_info.value)


def test_subprocess_called_with_expected_args() -> None:
    """Sanity check the cmd vector — model + output-format + -p flags."""
    with patch(
        "mission_brain.ingest.claude_cli_client.subprocess.run"
    ) as mock_run, patch(
        "mission_brain.ingest.claude_cli_client._resolve_claude_binary",
        return_value="/fake/claude",
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/fake/claude"],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        client = ClaudeCLIClient()
        client.generate("hello", model="claude-opus-4-7")
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "/fake/claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert "text" in cmd
        assert "--model" in cmd
        assert "claude-opus-4-7" in cmd
