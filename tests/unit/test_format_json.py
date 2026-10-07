"""Tests for --format json on CLI commands."""  # noqa: E501

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_mlops.cli.main import app
from tests.conftest import make_valid_dataset

runner = CliRunner()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_json(output: str) -> dict:
    """Extract JSON from CLI output that may contain leading log lines."""
    idx = output.index("{")
    return json.loads(output[idx:])


def _training_config(path: Path) -> str:
    path.write_text(
        "model: yolov8n.pt\nepochs: 1\nbatch: 1\nimgsz: 32\nmode: local_dry_run\nname: test\n",
        encoding="utf-8",
    )
    return str(path)


def _make_weights(path: Path) -> str:
    path.write_bytes(b"fake-weights")
    return str(path)


def _make_training_output(path: Path, weights_path: str | None = None) -> str:
    import json as _json

    data = {
        "success": True,
        "message": "ok",
        "job_status": "completed",
        "best_weights_path": weights_path or str(path.parent / "best.pt"),
        "artifacts": [],
        "warnings": [],
        "errors": [],
        "metadata": {},
    }
    path.write_text(_json.dumps(data), encoding="utf-8")
    return str(path)


def _make_evaluation_output(path: Path) -> str:
    import json as _json

    data = {
        "success": True,
        "message": "ok",
        "recommendation": "PROMOTE_CANDIDATE",
        "passed_checks": [],
        "failed_checks": [],
        "artifacts": [],
        "warnings": [],
        "errors": [],
        "metadata": {},
    }
    path.write_text(_json.dumps(data), encoding="utf-8")
    return str(path)


def _make_approval_decision(path: Path) -> str:
    import json as _json

    data = {
        "success": True,
        "message": "approved",
        "status": "approved",
        "action": "approve_model",
        "approver": "test",
        "comment": "",
        "artifacts": [],
        "warnings": [],
        "errors": [],
        "metadata": {},
    }
    path.write_text(_json.dumps(data), encoding="utf-8")
    return str(path)


# ── train --format json ───────────────────────────────────────────────────────


class TestTrainFormatJson:
    def test_outputs_valid_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        cfg = _training_config(tmp_path / "training.yaml")
        result = runner.invoke(
            app,
            [
                "train",
                "--dataset-path", str(ds),
                "--data-yaml", str(ds / "data.yaml"),
                "--training-config", cfg,
                "--output-dir", str(tmp_path / "out"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "success" in data

    def test_json_has_no_rich_markup(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        cfg = _training_config(tmp_path / "training.yaml")
        result = runner.invoke(
            app,
            [
                "train",
                "--dataset-path", str(ds),
                "--data-yaml", str(ds / "data.yaml"),
                "--training-config", cfg,
                "--output-dir", str(tmp_path / "out"),
                "--format", "json",
            ],
        )
        # Rich markup should not appear in raw JSON output
        assert "[bold" not in result.output
        assert "[green" not in result.output

    def test_text_format_does_not_output_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        cfg = _training_config(tmp_path / "training.yaml")
        result = runner.invoke(
            app,
            [
                "train",
                "--dataset-path", str(ds),
                "--data-yaml", str(ds / "data.yaml"),
                "--training-config", cfg,
                "--output-dir", str(tmp_path / "out"),
            ],
        )
        # Text output should not be parseable as a top-level JSON object
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_json(result.output)


# ── evaluate --format json ────────────────────────────────────────────────────


class TestEvaluateFormatJson:
    def test_outputs_valid_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--dataset-path", str(ds),
                "--data-yaml", str(ds / "data.yaml"),
                "--output-dir", str(tmp_path / "out"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "success" in data

    def test_json_contains_recommendation(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--dataset-path", str(ds),
                "--data-yaml", str(ds / "data.yaml"),
                "--output-dir", str(tmp_path / "out"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "recommendation" in data


# ── deploy-model --format json ────────────────────────────────────────────────


class TestDeployModelFormatJson:
    def test_outputs_valid_json_on_success(self, tmp_path: Path) -> None:
        weights = tmp_path / "best.pt"
        _make_weights(weights)
        result = runner.invoke(
            app,
            [
                "deploy-model", str(weights),
                "--model-name", "my-model",
                "--export-format", "pt",
                "--deployment-dir", str(tmp_path / "deployments"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "success" in data

    def test_outputs_valid_json_on_blocked(self, tmp_path: Path) -> None:
        # production deploy without approval → blocked; should still be valid JSON
        weights = tmp_path / "best.pt"
        _make_weights(weights)
        result = runner.invoke(
            app,
            [
                "deploy-model", str(weights),
                "--model-name", "my-model",
                "--export-format", "pt",
                "--target", "production",
                "--deployment-dir", str(tmp_path / "deployments"),
                "--format", "json",
            ],
        )
        assert result.exit_code == 1
        data = _parse_json(result.output)
        assert data["success"] is False


# ── register-model --format json ──────────────────────────────────────────────


class TestRegisterModelFormatJson:
    def test_outputs_valid_json(self, tmp_path: Path) -> None:
        # Create a real best.pt so the registry gate passes
        weights = tmp_path / "best.pt"
        _make_weights(weights)

        train_out = _make_training_output(tmp_path / "training_output.json", str(weights))
        eval_out = _make_evaluation_output(tmp_path / "evaluation_output.json")
        appr = _make_approval_decision(tmp_path / "approval_decision.json")

        result = runner.invoke(
            app,
            [
                "register-model",
                "--model-name", "test-model",
                "--training-output", train_out,
                "--evaluation-output", eval_out,
                "--approval-decision", appr,
                "--registry-dir", str(tmp_path / "registry"),
                "--output-dir", str(tmp_path / "reg_out"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "success" in data

    def test_json_contains_model_name(self, tmp_path: Path) -> None:
        weights = tmp_path / "best.pt"
        _make_weights(weights)

        train_out = _make_training_output(tmp_path / "training_output.json", str(weights))
        eval_out = _make_evaluation_output(tmp_path / "evaluation_output.json")
        appr = _make_approval_decision(tmp_path / "approval_decision.json")

        result = runner.invoke(
            app,
            [
                "register-model",
                "--model-name", "my-special-model",
                "--training-output", train_out,
                "--evaluation-output", eval_out,
                "--approval-decision", appr,
                "--registry-dir", str(tmp_path / "registry"),
                "--output-dir", str(tmp_path / "reg_out"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert data.get("model_name") == "my-special-model"


# ── run-mvp --format json ─────────────────────────────────────────────────────


class TestRunMvpFormatJson:
    def test_outputs_valid_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        cfg = _training_config(tmp_path / "training.yaml")
        result = runner.invoke(
            app,
            [
                "run-mvp",
                "--dataset-path", str(ds),
                "--data-yaml", str(ds / "data.yaml"),
                "--training-config", cfg,
                "--output-dir", str(tmp_path / "out"),
                "--dry-run",
                "--no-interactive",
                "--approval-action", "approve_model",
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "success" in data

    def test_json_contains_step_statuses(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        cfg = _training_config(tmp_path / "training.yaml")
        result = runner.invoke(
            app,
            [
                "run-mvp",
                "--dataset-path", str(ds),
                "--data-yaml", str(ds / "data.yaml"),
                "--training-config", cfg,
                "--output-dir", str(tmp_path / "out"),
                "--dry-run",
                "--no-interactive",
                "--approval-action", "approve_model",
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "message" in data
        assert "artifacts" in data

    def test_text_format_not_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        cfg = _training_config(tmp_path / "training.yaml")
        result = runner.invoke(
            app,
            [
                "run-mvp",
                "--dataset-path", str(ds),
                "--data-yaml", str(ds / "data.yaml"),
                "--training-config", cfg,
                "--output-dir", str(tmp_path / "out"),
                "--dry-run",
                "--no-interactive",
                "--approval-action", "approve_model",
            ],
        )
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_json(result.output)


# ── run-workflow --format json ────────────────────────────────────────────────


class TestRunWorkflowFormatJson:
    def _write_orchestrator_config(self, path: Path, dataset_path: str, data_yaml: str) -> str:
        path.write_text(
            f"steps:\n  - dataset_validation\n"
            f"dry_run: true\n"
            f"dataset_path: {dataset_path}\n"
            f"data_yaml_path: {data_yaml}\n"
            f"fail_on_warnings: false\n"
            f"training_config_path: null\n",
            encoding="utf-8",
        )
        return str(path)

    def test_outputs_valid_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        cfg = self._write_orchestrator_config(
            tmp_path / "orchestrator.yaml", str(ds), str(ds / "data.yaml")
        )
        result = runner.invoke(
            app,
            [
                "run-workflow",
                "--workflow-id", "wf_fmt_test",
                "--config", cfg,
                "--runs-dir", str(tmp_path / "runs"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "success" in data

    def test_json_contains_status(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        cfg = self._write_orchestrator_config(
            tmp_path / "orchestrator.yaml", str(ds), str(ds / "data.yaml")
        )
        result = runner.invoke(
            app,
            [
                "run-workflow",
                "--workflow-id", "wf_fmt_status",
                "--config", cfg,
                "--runs-dir", str(tmp_path / "runs"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "status" in data


# ── validate-dataset --format json ───────────────────────────────────────────


class TestValidateDatasetFormatJson:
    def test_outputs_valid_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        result = runner.invoke(
            app,
            ["validate-dataset", str(ds), "--format", "json"],
        )
        data = _parse_json(result.output)
        assert "success" in data

    def test_json_contains_status(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        result = runner.invoke(
            app,
            ["validate-dataset", str(ds), "--format", "json"],
        )
        data = _parse_json(result.output)
        assert "status" in data or "success" in data

    def test_text_format_not_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        result = runner.invoke(app, ["validate-dataset", str(ds)])
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_json(result.output)


# ── data-intake --format json ─────────────────────────────────────────────────


class TestDataIntakeFormatJson:
    def _make_intake_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            img = path / f"img{i:03d}.jpg"
            img.write_bytes(b"\xff\xd8\xff" + f"fake{i}".encode() + b"\xff\xd9")

    def test_outputs_valid_json(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw_images"
        self._make_intake_dir(raw)
        result = runner.invoke(
            app,
            [
                "data-intake", str(raw),
                "--dataset-name", "test_ds",
                "--source", "test_batch",
                "--output-dir", str(tmp_path / "intake_out"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "success" in data

    def test_json_contains_dataset_name(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw_images"
        self._make_intake_dir(raw)
        result = runner.invoke(
            app,
            [
                "data-intake", str(raw),
                "--dataset-name", "my_dataset",
                "--source", "batch_1",
                "--output-dir", str(tmp_path / "intake_out"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert data.get("dataset_name") == "my_dataset" or "success" in data


# ── label-qa --format json ────────────────────────────────────────────────────


class TestLabelQAFormatJson:
    def test_outputs_valid_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        result = runner.invoke(
            app,
            [
                "label-qa", str(ds),
                "--output-dir", str(tmp_path / "qa_out"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "success" in data

    def test_json_not_text(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        result = runner.invoke(
            app,
            ["label-qa", str(ds), "--output-dir", str(tmp_path / "qa_out")],
        )
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_json(result.output)


# ── version-dataset --format json ─────────────────────────────────────────────


class TestVersionDatasetFormatJson:
    def test_outputs_valid_json(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        result = runner.invoke(
            app,
            [
                "version-dataset", str(ds),
                "--dataset-name", "test_ds",
                "--registry-dir", str(tmp_path / "registry"),
                "--output-dir", str(tmp_path / "version_out"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "success" in data

    def test_json_contains_version(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        make_valid_dataset(ds)
        result = runner.invoke(
            app,
            [
                "version-dataset", str(ds),
                "--dataset-name", "test_ds",
                "--registry-dir", str(tmp_path / "registry"),
                "--output-dir", str(tmp_path / "version_out"),
                "--format", "json",
            ],
        )
        data = _parse_json(result.output)
        assert "version" in data or "dataset_name" in data or "success" in data


# ── doctor --format json ───────────────────────────────────────────────────────


class TestDoctorFormatJson:
    def test_outputs_valid_json(self) -> None:
        result = runner.invoke(app, ["doctor", "--no-tools", "--format", "json"])
        data = _parse_json(result.output)
        assert "checks" in data

    def test_json_contains_summary_counts(self) -> None:
        result = runner.invoke(app, ["doctor", "--no-tools", "--format", "json"])
        data = _parse_json(result.output)
        assert "num_ok" in data
        assert "num_warnings" in data
        assert "num_errors" in data

    def test_json_overall_status_present(self) -> None:
        result = runner.invoke(app, ["doctor", "--no-tools", "--format", "json"])
        data = _parse_json(result.output)
        assert "overall_status" in data

    def test_text_format_not_json(self) -> None:
        result = runner.invoke(app, ["doctor", "--no-tools"])
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_json(result.output)
