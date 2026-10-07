"""CLI robustness tests — non-happy-path coverage for every command group.

Exercises missing required files, conflicting flags, unknown values,
and edge cases that should produce clean exit-code-1 errors rather than
unhandled Python exceptions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_mlops.cli.main import app

runner = CliRunner()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _minimal_training_config(tmp_path: Path) -> str:
    p = tmp_path / "training.yaml"
    p.write_text("mode: local_dry_run\nepochs: 1\n", encoding="utf-8")
    return str(p)


def _minimal_data_yaml(tmp_path: Path) -> str:
    p = tmp_path / "data.yaml"
    p.write_text("nc: 1\nnames: [crack]\ntrain: train\nval: val\n", encoding="utf-8")
    return str(p)


def _minimal_orchestrator_config(tmp_path: Path, steps: list[str] | None = None) -> str:
    """Write a minimal orchestrator YAML pointing at nonexistent dataset (for error tests)."""
    steps_yaml = (
        "\n".join(f"  - {s}" for s in (steps or ["dataset_validation"]))
    )
    p = tmp_path / "orchestrator.yaml"
    p.write_text(
        f"dataset_path: {tmp_path / 'dataset'}\n"
        f"data_yaml_path: {tmp_path / 'data.yaml'}\n"
        f"training_config_path: {tmp_path / 'training.yaml'}\n"
        f"steps:\n{steps_yaml}\n",
        encoding="utf-8",
    )
    return str(p)


def _minimal_eval_report(tmp_path: Path) -> str:
    p = tmp_path / "evaluation_report.json"
    report = {
        "success": True,
        "message": "ok",
        "metrics": {"map50": 0.85, "map50_95": 0.60, "precision": 0.80, "recall": 0.75},
        "recommendation": "promote_candidate",
        "passed_checks": [],
        "failed_checks": [],
        "mode": "local_dry_run",
    }
    p.write_text(json.dumps(report), encoding="utf-8")
    return str(p)


def _minimal_approval_decision(tmp_path: Path) -> str:
    p = tmp_path / "approval_decision.json"
    decision = {
        "success": True,
        "status": "approved",
        "action": "approve_model",
        "approver": "test",
    }
    p.write_text(json.dumps(decision), encoding="utf-8")
    return str(p)


# ──────────────────────────────────────────────────────────────────────────────
# train command
# ──────────────────────────────────────────────────────────────────────────────


class TestTrainCommand:
    def test_missing_training_config_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "train",
                "--dataset-path", str(tmp_path),
                "--data-yaml", str(tmp_path / "data.yaml"),
                "--training-config", str(tmp_path / "nonexistent.yaml"),
            ],
        )
        assert result.exit_code != 0

    def test_unknown_runner_exits_1(self, tmp_path: Path):
        cfg = _minimal_training_config(tmp_path)
        dy = _minimal_data_yaml(tmp_path)
        result = runner.invoke(
            app,
            [
                "train",
                "--dataset-path", str(tmp_path),
                "--data-yaml", dy,
                "--training-config", cfg,
                "--runner", "bogus-runner",
            ],
        )
        assert result.exit_code == 1
        assert "bogus-runner" in result.output

    def test_azure_runner_without_config_exits_1(self, tmp_path: Path):
        cfg = _minimal_training_config(tmp_path)
        dy = _minimal_data_yaml(tmp_path)
        result = runner.invoke(
            app,
            [
                "train",
                "--dataset-path", str(tmp_path),
                "--data-yaml", dy,
                "--training-config", cfg,
                "--runner", "azure-ml",
            ],
        )
        assert result.exit_code == 1
        assert "--azure-config" in result.output


# ──────────────────────────────────────────────────────────────────────────────
# evaluate command
# ──────────────────────────────────────────────────────────────────────────────


class TestEvaluateCommand:
    def test_unknown_runner_exits_1(self, tmp_path: Path):
        dy = _minimal_data_yaml(tmp_path)
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--dataset-path", str(tmp_path),
                "--data-yaml", dy,
                "--runner", "not-a-real-runner",
            ],
        )
        assert result.exit_code == 1
        assert "not-a-real-runner" in result.output

    def test_azure_runner_without_config_exits_1(self, tmp_path: Path):
        dy = _minimal_data_yaml(tmp_path)
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--dataset-path", str(tmp_path),
                "--data-yaml", dy,
                "--runner", "azure-ml",
            ],
        )
        assert result.exit_code == 1
        assert "--azure-config" in result.output

    def test_training_output_missing_file_raises(self, tmp_path: Path):
        """--training-output pointing at nonexistent JSON should fail, not silently ignore."""
        dy = _minimal_data_yaml(tmp_path)
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--dataset-path", str(tmp_path),
                "--data-yaml", dy,
                "--training-output", str(tmp_path / "nonexistent_output.json"),
            ],
        )
        # Should exit non-zero — FileNotFoundError must not bubble as traceback
        assert result.exit_code != 0


# ──────────────────────────────────────────────────────────────────────────────
# deploy-model command
# ──────────────────────────────────────────────────────────────────────────────


class TestDeployModelCommand:
    def test_local_backend_missing_model_path_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "deploy-model",
                "--model-name", "my-model",
                "--backend", "local",
            ],
        )
        assert result.exit_code == 1
        assert "model_path" in result.output.lower()

    def test_invalid_target_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "deploy-model",
                str(tmp_path / "best.pt"),
                "--model-name", "m",
                "--target", "not-a-target",
            ],
        )
        assert result.exit_code == 1
        assert "not-a-target" in result.output

    def test_invalid_export_format_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "deploy-model",
                str(tmp_path / "best.pt"),
                "--model-name", "m",
                "--export-format", "invalid_fmt",
            ],
        )
        assert result.exit_code == 1
        assert "invalid_fmt" in result.output

    def test_azure_ml_backend_missing_azure_config_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "deploy-model",
                "--model-name", "m",
                "--backend", "azure_ml",
            ],
        )
        assert result.exit_code == 1
        assert "--azure-config" in result.output

    def test_docker_backend_missing_docker_config_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "deploy-model",
                str(tmp_path / "best.pt"),
                "--model-name", "m",
                "--backend", "docker",
            ],
        )
        assert result.exit_code == 1
        assert "--docker-config" in result.output

    def test_aks_backend_missing_aks_config_exits_1(self, tmp_path: Path):
        # Provide docker-config but not aks-config
        docker_cfg = tmp_path / "docker.yaml"
        docker_cfg.write_text("image_name: test\nregistry: test.azurecr.io\n", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "deploy-model",
                str(tmp_path / "best.pt"),
                "--model-name", "m",
                "--backend", "aks",
                "--docker-config", str(docker_cfg),
            ],
        )
        assert result.exit_code == 1
        assert "--aks-config" in result.output

    def test_invalid_backend_exits_1(self):
        result = runner.invoke(
            app,
            [
                "deploy-model",
                "--model-name", "m",
                "--backend", "nonexistent_backend",
            ],
        )
        assert result.exit_code == 1
        assert "nonexistent_backend" in result.output

    def test_production_target_missing_approval_exits_via_agent(self, tmp_path: Path):
        """production target with no approval file should return blocked status -> exit 1."""
        model = tmp_path / "best.pt"
        model.write_bytes(b"fake")
        result = runner.invoke(
            app,
            [
                "deploy-model",
                str(model),
                "--model-name", "m",
                "--target", "production",
                "--export-format", "pt",
                "--rollback-plan", "revert to v1",
                # no --production-approval
            ],
        )
        assert result.exit_code == 1


# ──────────────────────────────────────────────────────────────────────────────
# run-workflow command
# ──────────────────────────────────────────────────────────────────────────────


class TestRunWorkflowCommand:
    def test_missing_config_file_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "run-workflow",
                "--workflow-id", "wf_test",
                "--config", str(tmp_path / "nonexistent.yaml"),
                "--runs-dir", str(tmp_path / "runs"),
            ],
        )
        assert result.exit_code == 1
        assert "nonexistent.yaml" in result.output or "Cannot load" in result.output

    def test_resume_nonexistent_workflow_exits_1(self, tmp_path: Path):
        cfg = _minimal_orchestrator_config(tmp_path)
        result = runner.invoke(
            app,
            [
                "run-workflow",
                "--workflow-id", "wf_never_existed",
                "--config", cfg,
                "--runs-dir", str(tmp_path / "runs"),
                "--resume",
            ],
        )
        # Resuming a workflow that doesn't exist should fail cleanly
        assert result.exit_code == 1

    def test_duplicate_workflow_id_without_resume_exits_1(self, tmp_path: Path):
        """Running the same workflow ID twice without --resume should fail."""

        runs = tmp_path / "runs"
        wf_dir = runs / "wf_dup"
        wf_dir.mkdir(parents=True)
        state = {
            "workflow_id": "wf_dup",
            "status": "completed",
            "current_state": "COMPLETED",
            "completed_steps": [],
            "step_outputs": {},
            "artifacts": {},
            "started_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }
        (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        (wf_dir / "audit_log.jsonl").write_text("", encoding="utf-8")

        # Write a minimal dataset for the config
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        _minimal_data_yaml(tmp_path)
        _minimal_training_config(tmp_path)
        cfg = _minimal_orchestrator_config(tmp_path)

        result = runner.invoke(
            app,
            [
                "run-workflow",
                "--workflow-id", "wf_dup",
                "--config", cfg,
                "--runs-dir", str(runs),
                # no --resume
            ],
        )
        assert result.exit_code == 1


# ──────────────────────────────────────────────────────────────────────────────
# show-state command
# ──────────────────────────────────────────────────────────────────────────────


class TestShowStateCommand:
    def test_nonexistent_workflow_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "show-state",
                "wf_nonexistent_99999",
                "--runs-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "wf_nonexistent_99999" in result.output

    def test_nonexistent_workflow_json_format_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "show-state",
                "wf_nonexistent_json",
                "--runs-dir", str(tmp_path),
                "--format", "json",
            ],
        )
        assert result.exit_code == 1

    def test_no_workflow_id_empty_dir_exits_0(self, tmp_path: Path):
        """Without a workflow ID and empty runs dir: list mode, graceful empty message."""
        result = runner.invoke(
            app,
            [
                "show-state",
                "--runs-dir", str(tmp_path),
            ],
        )
        # No workflows — should succeed (nothing to fail on)
        assert result.exit_code == 0


# ──────────────────────────────────────────────────────────────────────────────
# cost-report command
# ──────────────────────────────────────────────────────────────────────────────


class TestCostReportCommand:
    def test_nonexistent_workflow_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "cost-report",
                "wf_ghost",
                "--runs-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "state" in result.output.lower()

    def test_invalid_pricing_config_exits_1(self, tmp_path: Path):
        """Providing a corrupt pricing YAML should exit 1."""
        runs = tmp_path / "runs"
        wf_dir = runs / "wf_cost"
        wf_dir.mkdir(parents=True)
        state = {
            "workflow_id": "wf_cost",
            "step_outputs": {},
            "status": "completed",
        }
        (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

        bad_pricing = tmp_path / "bad_pricing.yaml"
        bad_pricing.write_text("rates: not_a_dict\n", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "cost-report",
                "wf_cost",
                "--runs-dir", str(runs),
                "--pricing-config", str(bad_pricing),
            ],
        )
        assert result.exit_code == 1


# ──────────────────────────────────────────────────────────────────────────────
# lint-config command
# ──────────────────────────────────────────────────────────────────────────────


class TestLintConfigCommand:
    def test_nonexistent_config_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "lint-config",
                str(tmp_path / "nonexistent.yaml"),
            ],
        )
        assert result.exit_code == 1

    def test_invalid_yaml_exits_1(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("steps: [\n  unclosed bracket\n", encoding="utf-8")
        result = runner.invoke(
            app,
            ["lint-config", str(bad)],
        )
        assert result.exit_code == 1

    def test_valid_minimal_config_exits_0(self, tmp_path: Path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        dy = _minimal_data_yaml(tmp_path)
        cfg = _minimal_training_config(tmp_path)
        orch_cfg = tmp_path / "orchestrator.yaml"
        orch_cfg.write_text(
            f"dataset_path: {dataset}\n"
            f"data_yaml_path: {dy}\n"
            f"training_config_path: {cfg}\n"
            "steps:\n  - dataset_validation\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            ["lint-config", str(orch_cfg)],
        )
        # May have warnings but should not error
        assert result.exit_code == 0


# ──────────────────────────────────────────────────────────────────────────────
# lint-docs command
# ──────────────────────────────────────────────────────────────────────────────


class TestLintDocsCommand:
    def test_nonexistent_docs_dir_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "lint-docs",
                str(tmp_path / "nonexistent_docs"),
                "--src-dir", "src",
            ],
        )
        # Should fail cleanly with error output, not a traceback
        assert result.exit_code == 1

    def test_empty_docs_dir_reports_no_files(self, tmp_path: Path):
        docs = tmp_path / "docs_root"
        docs.mkdir()
        (docs / "agents").mkdir()
        (docs / "docs").mkdir()
        result = runner.invoke(
            app,
            [
                "lint-docs",
                str(docs),
                "--src-dir", str(tmp_path / "src"),
            ],
        )
        # Empty dirs — might exit 0 (no errors found) or 1 (0 specs = warning)
        # Either way, must not crash with traceback
        assert result.exit_code in (0, 1)
        assert result.exception is None or isinstance(result.exception, SystemExit)


# ──────────────────────────────────────────────────────────────────────────────
# approve command
# ──────────────────────────────────────────────────────────────────────────────


class TestApproveCommand:
    def test_invalid_action_exits_1(self, tmp_path: Path):
        eval_report = _minimal_eval_report(tmp_path)
        result = runner.invoke(
            app,
            [
                "approve",
                "--evaluation-output", eval_report,
                "--no-interactive",
                "--action", "invalid_action",
            ],
        )
        assert result.exit_code == 1
        assert "invalid_action" in result.output

    def test_noninteractive_missing_action_exits_via_agent(self, tmp_path: Path):
        """Non-interactive mode with no --action should return failure (agent handles it)."""
        eval_report = _minimal_eval_report(tmp_path)
        result = runner.invoke(
            app,
            [
                "approve",
                "--evaluation-output", eval_report,
                "--no-interactive",
            ],
        )
        # Agent returns failure when no action given in non-interactive mode
        assert result.exit_code == 1


# ──────────────────────────────────────────────────────────────────────────────
# approve-training command
# ──────────────────────────────────────────────────────────────────────────────


class TestApproveTrainingCommand:
    def test_missing_report_file_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "approve-training",
                str(tmp_path / "nonexistent_report.json"),
                "--no-interactive",
                "--action", "approve_training",
            ],
        )
        assert result.exit_code != 0

    def test_invalid_action_exits_1(self, tmp_path: Path):
        report = tmp_path / "dataset_quality_report.json"
        report.write_text(
            json.dumps({"status": "passed", "success": True}),
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "approve-training",
                str(report),
                "--no-interactive",
                "--action", "not_a_real_action",
            ],
        )
        assert result.exit_code == 1


# ──────────────────────────────────────────────────────────────────────────────
# register-model command
# ──────────────────────────────────────────────────────────────────────────────


class TestRegisterModelCommand:
    def test_missing_training_output_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "register-model",
                "--model-name", "m",
                "--training-output", str(tmp_path / "nonexistent.json"),
                "--evaluation-output", str(tmp_path / "nonexistent_eval.json"),
                "--approval-decision", str(tmp_path / "nonexistent_approval.json"),
                "--registry-dir", str(tmp_path / "registry"),
            ],
        )
        assert result.exit_code != 0


# ──────────────────────────────────────────────────────────────────────────────
# model-decision command
# ──────────────────────────────────────────────────────────────────────────────


class TestModelDecisionCommand:
    def test_nonexistent_eval_report_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "model-decision",
                str(tmp_path / "nonexistent.json"),
            ],
        )
        assert result.exit_code != 0

    def test_valid_eval_report_exits_0(self, tmp_path: Path):
        eval_report = _minimal_eval_report(tmp_path)
        result = runner.invoke(
            app,
            [
                "model-decision",
                eval_report,
                "--output-dir", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0


# ──────────────────────────────────────────────────────────────────────────────
# prune-runs command
# ──────────────────────────────────────────────────────────────────────────────


class TestPruneRunsCommand:
    def test_no_criteria_exits_1(self, tmp_path: Path):
        """prune-runs with neither --keep-last nor --older-than-days must fail."""
        result = runner.invoke(
            app,
            [
                "prune-runs",
                "--runs-dir", str(tmp_path),
                # no --keep-last or --older-than-days
            ],
        )
        assert result.exit_code == 1
        assert "--keep-last" in result.output or "--older-than-days" in result.output

    def test_keep_last_zero_exits_gracefully(self, tmp_path: Path):
        """--keep-last 0 is nonsensical but should not crash."""
        result = runner.invoke(
            app,
            [
                "prune-runs",
                "--runs-dir", str(tmp_path),
                "--keep-last", "0",
                "--dry-run",
            ],
        )
        # Either exit 0 (nothing to prune) or exit 1 (validation error) — no crash
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_empty_runs_dir_nothing_to_prune(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "prune-runs",
                "--runs-dir", str(tmp_path),
                "--keep-last", "5",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Nothing to prune" in result.output


# ──────────────────────────────────────────────────────────────────────────────
# tag-run command
# ──────────────────────────────────────────────────────────────────────────────


class TestTagRunCommand:
    def test_nonexistent_workflow_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "tag-run",
                "wf_ghost",
                "--runs-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 1

    def test_malformed_tag_exits_1(self, tmp_path: Path):
        runs = tmp_path / "runs"
        wf = runs / "wf_tag"
        wf.mkdir(parents=True)
        (wf / "state.json").write_text(json.dumps({"tags": {}}), encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "tag-run",
                "wf_tag",
                "bad_tag_no_equals",
                "--runs-dir", str(runs),
            ],
        )
        assert result.exit_code == 1
        assert "key=value" in result.output


# ──────────────────────────────────────────────────────────────────────────────
# validate-dataset command
# ──────────────────────────────────────────────────────────────────────────────


class TestValidateDatasetCommand:
    def test_nonexistent_path_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "validate-dataset",
                str(tmp_path / "nonexistent_dataset"),
            ],
        )
        assert result.exit_code != 0

    def test_empty_dataset_path_exits_nonzero(self, tmp_path: Path):
        empty_ds = tmp_path / "empty_ds"
        empty_ds.mkdir()
        result = runner.invoke(
            app,
            [
                "validate-dataset",
                str(empty_ds),
            ],
        )
        # Missing data.yaml / split dirs → validation failure → exit 1
        assert result.exit_code != 0


# ──────────────────────────────────────────────────────────────────────────────
# doctor command
# ──────────────────────────────────────────────────────────────────────────────


class TestDoctorCommand:
    def test_no_args_exits_0_or_1(self):
        """Basic invocation should not raise an unhandled exception."""
        result = runner.invoke(app, ["doctor", "--no-tools"])
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_nonexistent_config_reported_as_error(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "doctor",
                "--config", str(tmp_path / "nonexistent.yaml"),
                "--no-tools",
            ],
        )
        assert result.exit_code == 1

    def test_json_format_always_valid_json(self):
        result = runner.invoke(app, ["doctor", "--no-tools", "--format", "json"])
        # May exit 0 or 1, but stdout must be valid JSON
        try:
            data = json.loads(result.output)
            assert "checks" in data
        except json.JSONDecodeError:
            pytest.fail("doctor --format json produced invalid JSON")


# ──────────────────────────────────────────────────────────────────────────────
# monitor command
# ──────────────────────────────────────────────────────────────────────────────


class TestMonitorCommand:
    def test_missing_predictions_log_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "monitor",
                str(tmp_path / "nonexistent.jsonl"),
                "--endpoint-name", "test-endpoint",
            ],
        )
        assert result.exit_code != 0

    def test_azure_monitor_source_missing_workspace_id_exits_1(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "monitor",
                "--endpoint-name", "ep",
                "--source", "azure-monitor",
                # no --app-insights-workspace-id
            ],
        )
        assert result.exit_code == 1
