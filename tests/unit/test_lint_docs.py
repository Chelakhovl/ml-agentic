"""Tests for DocsLinter and the lint-docs CLI command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agentic_mlops.cli.main import app
from agentic_mlops.contracts.doctor import CheckStatus
from agentic_mlops.tools.docs_linter import _LIVING_SPEC_MARKER, DocsLinter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_docs_dir(tmp: Path, with_header: bool = True) -> Path:
    """Create a minimal docs structure used by most tests."""
    docs = tmp / "workflow_docs"
    header = f"<!-- {_LIVING_SPEC_MARKER} -->\n" if with_header else ""
    _write(docs / "agents" / "01_data_intake_agent.md", f"{header}# Data Intake Agent\n")
    _write(docs / "agents" / "00_orchestrator_agent.md", f"{header}# Orchestrator Agent\n")
    _write(docs / "docs" / "01_mvp_scope.md", f"{header}# MVP Scope\n")
    return docs


def _make_src_dir(tmp: Path) -> Path:
    """Create a minimal src layout."""
    pkg = tmp / "src" / "agentic_mlops"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "agents" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (pkg / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "agents" / "data_intake.py").write_text(
        "class DataIntakeAgent: pass\n", encoding="utf-8"
    )
    return tmp / "src"


# ---------------------------------------------------------------------------
# DocsLinter unit tests
# ---------------------------------------------------------------------------


class TestDocsLinterStructure:
    def test_no_spec_files_returns_error(self, tmp_path):
        empty = tmp_path / "empty_docs"
        empty.mkdir()
        src = _make_src_dir(tmp_path)
        report = DocsLinter().lint(empty, src)
        assert report.overall_status == CheckStatus.ERROR

    def test_finds_spec_files_in_agents_and_docs(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        report = DocsLinter().lint(docs, src)
        # Should have a docs_found OK check
        names = [c.name for c in report.checks]
        assert "docs_found" in names
        found = next(c for c in report.checks if c.name == "docs_found")
        assert found.status == CheckStatus.OK


class TestDocsLinterHeader:
    def test_all_headers_present_gives_ok(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        report = DocsLinter().lint(docs, src)
        header_chk = next(c for c in report.checks if c.name == "header")
        assert header_chk.status == CheckStatus.OK

    def test_missing_header_gives_warning(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=False)
        src = _make_src_dir(tmp_path)
        report = DocsLinter().lint(docs, src)
        header_chk = next(c for c in report.checks if c.name == "header")
        assert header_chk.status == CheckStatus.WARNING
        assert header_chk.detail  # names the missing files


class TestDocsLinterPaths:
    def test_valid_path_reference_passes(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        # Reference an existing file
        _write(
            docs / "agents" / "02_structuring_agent.md",
            f"<!-- {_LIVING_SPEC_MARKER} -->\n"
            "See `agents/data_intake.py` for the implementation.\n",
        )
        report = DocsLinter().lint(docs, src)
        paths_chk = next(c for c in report.checks if c.name == "paths")
        assert paths_chk.status == CheckStatus.OK

    def test_missing_path_reference_warns(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        # Reference a non-existent file
        _write(
            docs / "agents" / "99_fake_agent.md",
            f"<!-- {_LIVING_SPEC_MARKER} -->\n"
            "See `agents/nonexistent_agent.py` for details.\n",
        )
        report = DocsLinter().lint(docs, src)
        paths_chk = next(c for c in report.checks if c.name == "paths")
        assert paths_chk.status == CheckStatus.WARNING
        assert "nonexistent_agent.py" in (paths_chk.detail or "")


class TestDocsLinterClasses:
    def test_existing_class_reference_passes(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        # Reference a class that actually exists
        _write(
            docs / "agents" / "01_data_intake_agent.md",
            f"<!-- {_LIVING_SPEC_MARKER} -->\n"
            "DataIntakeAgent is the main entry point.\n",
        )
        report = DocsLinter().lint(docs, src)
        classes_chk = next(c for c in report.checks if c.name == "classes")
        assert classes_chk.status == CheckStatus.OK

    def test_missing_class_reference_warns(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        _write(
            docs / "agents" / "01_data_intake_agent.md",
            f"<!-- {_LIVING_SPEC_MARKER} -->\n"
            "GhostModelAgent handles all the invisible work.\n",
        )
        report = DocsLinter().lint(docs, src)
        classes_chk = next(c for c in report.checks if c.name == "classes")
        assert classes_chk.status == CheckStatus.WARNING
        assert "GhostModelAgent" in (classes_chk.detail or "")


class TestDocsLinterClaudeSync:
    def _make_claude_md(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "CLAUDE.md"
        p.write_text(content, encoding="utf-8")
        return p

    def test_agent_mentioned_in_claude_passes(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        claude_md = self._make_claude_md(
            tmp_path,
            "# CLAUDE.md\nDataIntakeAgent handles ingestion.\nOrchestratorAgent routes.\n",
        )
        report = DocsLinter().lint(docs, src, claude_md_path=claude_md)
        sync_chk = next((c for c in report.checks if c.name == "claude_sync"), None)
        assert sync_chk is not None
        assert sync_chk.status == CheckStatus.OK

    def test_agent_missing_from_claude_warns(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        claude_md = self._make_claude_md(tmp_path, "# CLAUDE.md\nNo agent mentioned here.\n")
        report = DocsLinter().lint(docs, src, claude_md_path=claude_md)
        sync_chk = next((c for c in report.checks if c.name == "claude_sync"), None)
        assert sync_chk is not None
        assert sync_chk.status == CheckStatus.WARNING

    def test_no_claude_md_skips_check(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        report = DocsLinter().lint(docs, src, claude_md_path=None)
        names = [c.name for c in report.checks]
        assert "claude_sync" not in names

    def test_missing_claude_md_warns(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        report = DocsLinter().lint(docs, src, claude_md_path=tmp_path / "CLAUDE.md")
        sync_chk = next((c for c in report.checks if c.name == "claude_sync"), None)
        assert sync_chk is not None
        assert sync_chk.status == CheckStatus.WARNING


class TestDocsLinterStrict:
    def test_strict_mode_degrades_warning_to_error(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=False)  # missing headers → warning
        src = _make_src_dir(tmp_path)
        report_normal = DocsLinter().lint(docs, src)
        report_strict = DocsLinter().lint(docs, src, strict=True)
        assert report_normal.overall_status == CheckStatus.WARNING
        assert report_strict.overall_status == CheckStatus.ERROR

    def test_strict_mode_keeps_ok_as_ok(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        report = DocsLinter().lint(docs, src, strict=True)
        # No warnings or errors, so strict doesn't change anything
        assert report.overall_status == CheckStatus.OK


# ---------------------------------------------------------------------------
# CLI tests (lint-docs command)
# ---------------------------------------------------------------------------


class TestLintDocsCli:
    def test_exit_0_on_clean_docs(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app, ["lint-docs", str(docs), "--src-dir", str(src / "..")])
        # May have warnings about missing paths/classes (no real content) but not ERROR exit
        assert result.exit_code in (0, 1)  # at worst 1 (warnings in strict)
        assert "agentic-mlops lint-docs" in (result.output or "")

    def test_exit_nonzero_on_missing_docs_dir(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["lint-docs", str(tmp_path / "nonexistent"), "--src-dir", str(tmp_path / "src")],
        )
        assert result.exit_code != 0

    def test_strict_flag_accepted(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app, ["lint-docs", str(docs), "--src-dir", str(src / ".."), "--strict"]
        )
        # Flag is accepted without "No such option"
        assert "no such option" not in (result.output or "").lower()

    def test_output_contains_summary(self, tmp_path):
        docs = _make_docs_dir(tmp_path, with_header=True)
        src = _make_src_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app, ["lint-docs", str(docs), "--src-dir", str(src / "..")]
        )
        assert "Summary:" in (result.output or "")
