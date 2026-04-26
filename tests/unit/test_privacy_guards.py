"""Tests for privacy guards — network egress + path containment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mission_brain.ingest.network_guard import NetworkEgressViolation, assert_local
from mission_brain.loaders.path_guard import PathEscapeViolation, assert_under_corpus


class TestAssertLocal:
    def test_ollama_default_url_passes(self) -> None:
        assert_local("http://localhost:11434")

    def test_localhost_with_path_passes(self) -> None:
        assert_local("http://localhost:11434/api/generate")

    def test_127_0_0_1_passes(self) -> None:
        assert_local("http://127.0.0.1/")

    def test_127_0_0_1_with_port_passes(self) -> None:
        assert_local("http://127.0.0.1:8080/api")

    def test_ipv6_loopback_passes(self) -> None:
        assert_local("http://[::1]:11434/api/generate")

    def test_public_domain_raises(self) -> None:
        with pytest.raises(NetworkEgressViolation):
            assert_local("http://example.com/")

    def test_public_ip_raises(self) -> None:
        with pytest.raises(NetworkEgressViolation):
            assert_local("http://8.8.8.8/")

    def test_https_public_raises(self) -> None:
        with pytest.raises(NetworkEgressViolation):
            assert_local("https://api.openai.com/v1/chat")

    def test_private_rfc1918_raises(self) -> None:
        # 10.0.0.0/8 is private but not loopback — we guard against egress
        # off the local machine, so non-loopback private ranges still raise.
        with pytest.raises(NetworkEgressViolation):
            assert_local("http://10.0.0.1/")

    def test_allowlist_override_works(self) -> None:
        assert_local("http://example.com/", allowlist=["example.com"])

    def test_allowlist_does_not_affect_unrelated_hosts(self) -> None:
        with pytest.raises(NetworkEgressViolation):
            assert_local("http://evil.com/", allowlist=["example.com"])

    def test_missing_host_raises(self) -> None:
        with pytest.raises(NetworkEgressViolation):
            assert_local("not-a-url")

    def test_unresolvable_host_raises(self) -> None:
        with pytest.raises(NetworkEgressViolation):
            # .invalid is RFC 2606 reserved — guaranteed not to resolve.
            assert_local("http://nonexistent.invalid/")


class TestAssertUnderCorpus:
    def test_file_inside_corpus_passes(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        inner = corpus / "note.md"
        inner.write_text("x")
        assert_under_corpus(inner, corpus)

    def test_nested_file_inside_corpus_passes(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        (corpus / "a" / "b").mkdir(parents=True)
        inner = corpus / "a" / "b" / "note.md"
        inner.write_text("x")
        assert_under_corpus(inner, corpus)

    def test_root_itself_is_allowed(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        assert_under_corpus(corpus, corpus)

    def test_traversal_raises(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        outside = corpus / ".." / ".." / "etc" / "passwd"
        with pytest.raises(PathEscapeViolation):
            assert_under_corpus(outside, corpus)

    def test_absolute_outside_raises(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        with pytest.raises(PathEscapeViolation):
            assert_under_corpus(elsewhere / "x.md", corpus)

    @pytest.mark.skipif(
        sys.platform == "win32" and not hasattr(os, "symlink"),
        reason="symlink test requires symlink support",
    )
    def test_symlink_escape_raises(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        secret_target = tmp_path / "secrets"
        secret_target.mkdir()
        (secret_target / "leak.txt").write_text("ssh-key")
        link = corpus / "leak"
        try:
            os.symlink(secret_target, link, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation unsupported: {exc}")
        with pytest.raises(PathEscapeViolation):
            assert_under_corpus(link / "leak.txt", corpus)

    def test_sibling_directory_raises(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        sibling = tmp_path / "corpus-sneaky"
        sibling.mkdir()
        (sibling / "x.md").write_text("x")
        # Prefix-match false positive: "corpus-sneaky" starts with
        # "corpus" textually but is NOT under it. `relative_to`
        # catches this correctly.
        with pytest.raises(PathEscapeViolation):
            assert_under_corpus(sibling / "x.md", corpus)
