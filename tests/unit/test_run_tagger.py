"""Unit tests for WorkflowRunTagger and the tag-run CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_mlops.tools.run_tagger import WorkflowRunTagger

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_run(runs_dir: Path, workflow_id: str, existing_tags: dict | None = None) -> Path:
    d = runs_dir / workflow_id
    d.mkdir(parents=True, exist_ok=True)
    state: dict = {"workflow_id": workflow_id, "status": "completed"}
    if existing_tags is not None:
        state["tags"] = existing_tags
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return d


def _load_state(runs_dir: Path, workflow_id: str) -> dict:
    return json.loads((runs_dir / workflow_id / "state.json").read_text(encoding="utf-8"))


def _tagger() -> WorkflowRunTagger:
    return WorkflowRunTagger()


# ── WorkflowRunTagger.add ─────────────────────────────────────────────────────


class TestAdd:
    def test_add_single_tag(self, tmp_path):
        _make_run(tmp_path, "wf_a")
        result = _tagger().add(tmp_path, "wf_a", {"env": "prod"})
        assert result == {"env": "prod"}
        assert _load_state(tmp_path, "wf_a")["tags"] == {"env": "prod"}

    def test_add_multiple_tags(self, tmp_path):
        _make_run(tmp_path, "wf_a")
        result = _tagger().add(tmp_path, "wf_a", {"env": "prod", "version": "2"})
        assert result["env"] == "prod"
        assert result["version"] == "2"

    def test_add_merges_with_existing(self, tmp_path):
        _make_run(tmp_path, "wf_a", existing_tags={"env": "staging"})
        result = _tagger().add(tmp_path, "wf_a", {"version": "3"})
        assert result == {"env": "staging", "version": "3"}

    def test_add_overwrites_existing_key(self, tmp_path):
        _make_run(tmp_path, "wf_a", existing_tags={"env": "staging"})
        result = _tagger().add(tmp_path, "wf_a", {"env": "prod"})
        assert result["env"] == "prod"

    def test_add_does_not_overwrite_other_state_keys(self, tmp_path):
        _make_run(tmp_path, "wf_a")
        _tagger().add(tmp_path, "wf_a", {"env": "prod"})
        state = _load_state(tmp_path, "wf_a")
        assert state["status"] == "completed"
        assert state["workflow_id"] == "wf_a"

    def test_add_raises_for_missing_workflow(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no_such"):
            _tagger().add(tmp_path, "no_such", {"k": "v"})


# ── WorkflowRunTagger.remove ──────────────────────────────────────────────────


class TestRemove:
    def test_remove_existing_key(self, tmp_path):
        _make_run(tmp_path, "wf_a", existing_tags={"env": "prod", "version": "2"})
        result = _tagger().remove(tmp_path, "wf_a", ["env"])
        assert "env" not in result
        assert result["version"] == "2"

    def test_remove_nonexistent_key_is_noop(self, tmp_path):
        _make_run(tmp_path, "wf_a", existing_tags={"env": "prod"})
        result = _tagger().remove(tmp_path, "wf_a", ["ghost"])
        assert result == {"env": "prod"}

    def test_remove_all_keys_leaves_empty(self, tmp_path):
        _make_run(tmp_path, "wf_a", existing_tags={"a": "1", "b": "2"})
        result = _tagger().remove(tmp_path, "wf_a", ["a", "b"])
        assert result == {}

    def test_remove_persists_to_disk(self, tmp_path):
        _make_run(tmp_path, "wf_a", existing_tags={"env": "prod"})
        _tagger().remove(tmp_path, "wf_a", ["env"])
        assert _load_state(tmp_path, "wf_a").get("tags") == {}

    def test_remove_raises_for_missing_workflow(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _tagger().remove(tmp_path, "no_such", ["k"])


# ── WorkflowRunTagger.get ─────────────────────────────────────────────────────


class TestGet:
    def test_get_returns_existing_tags(self, tmp_path):
        _make_run(tmp_path, "wf_a", existing_tags={"env": "prod"})
        assert _tagger().get(tmp_path, "wf_a") == {"env": "prod"}

    def test_get_no_tags_returns_empty(self, tmp_path):
        _make_run(tmp_path, "wf_a")
        assert _tagger().get(tmp_path, "wf_a") == {}

    def test_get_returns_copy_not_reference(self, tmp_path):
        _make_run(tmp_path, "wf_a", existing_tags={"k": "v"})
        t = _tagger()
        result = t.get(tmp_path, "wf_a")
        result["k"] = "mutated"
        # disk not affected
        assert _load_state(tmp_path, "wf_a")["tags"]["k"] == "v"

    def test_get_raises_for_missing_workflow(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _tagger().get(tmp_path, "no_such")


# ── CLI tests ─────────────────────────────────────────────────────────────────


class TestTagRunCLI:
    def _setup(self, tmp_path: Path, tags: dict | None = None) -> Path:
        _make_run(tmp_path, "wf_1", existing_tags=tags)
        return tmp_path

    def test_show_tags_exit_0(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        self._setup(tmp_path, tags={"env": "prod"})
        result = CliRunner().invoke(app, ["tag-run", "wf_1", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0

    def test_show_tags_displays_key_value(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        self._setup(tmp_path, tags={"env": "prod", "version": "3"})
        result = CliRunner().invoke(app, ["tag-run", "wf_1", "--runs-dir", str(tmp_path)])
        assert "env" in result.output
        assert "prod" in result.output

    def test_show_no_tags_message(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        self._setup(tmp_path)
        result = CliRunner().invoke(app, ["tag-run", "wf_1", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "no tags" in result.output.lower()

    def test_set_tag_via_cli(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        self._setup(tmp_path)
        result = CliRunner().invoke(
            app, ["tag-run", "wf_1", "env=staging", "--runs-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert _load_state(tmp_path, "wf_1")["tags"]["env"] == "staging"

    def test_set_multiple_tags(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        self._setup(tmp_path)
        result = CliRunner().invoke(
            app,
            ["tag-run", "wf_1", "env=prod", "dataset=v5", "--runs-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        tags = _load_state(tmp_path, "wf_1")["tags"]
        assert tags["env"] == "prod"
        assert tags["dataset"] == "v5"

    def test_malformed_tag_exits_1(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        self._setup(tmp_path)
        result = CliRunner().invoke(
            app, ["tag-run", "wf_1", "noequalsign", "--runs-dir", str(tmp_path)]
        )
        assert result.exit_code == 1

    def test_remove_tag_via_cli(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        self._setup(tmp_path, tags={"env": "prod", "keep": "yes"})
        result = CliRunner().invoke(
            app, ["tag-run", "wf_1", "--remove", "env", "--runs-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        tags = _load_state(tmp_path, "wf_1").get("tags", {})
        assert "env" not in tags
        assert tags.get("keep") == "yes"

    def test_missing_workflow_exits_1(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        result = CliRunner().invoke(app, ["tag-run", "no_such_wf", "--runs-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "no_such_wf" in result.output or "not found" in result.output.lower()
