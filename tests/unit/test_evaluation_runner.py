"""Unit tests for evaluation runners.

Coverage matrix:
    1.  FakeEvaluationRunner returns deterministic metrics (same as dry-run default)
    2.  FakeEvaluationRunner with override_metrics uses injected values
    3.  LocalYOLOEvaluationRunner calls YOLO(weights_path).val() with correct args
    4.  LocalYOLOEvaluationRunner extracts map50/map50_95/precision/recall from mock results
    5.  LocalYOLOEvaluationRunner extracts per-class metrics correctly
    6.  LocalYOLOEvaluationRunner collects confusion_matrix.png when present
    7.  LocalYOLOEvaluationRunner collects PR_curve.png when present
    8.  LocalYOLOEvaluationRunner handles missing artifacts gracefully
    9.  LocalYOLOEvaluationRunner writes evaluation_output.json
   10.  LocalYOLOEvaluationRunner evaluation_output.json has correct fields
   11.  LocalYOLOEvaluationRunner produces PROMOTE_CANDIDATE for passing metrics
   12.  LocalYOLOEvaluationRunner produces RETRAIN for failing metrics
   13.  LocalYOLOEvaluationRunner raises RuntimeError when ultralytics not installed
   14.  LocalYOLOEvaluationRunner returns FAILED if model path doesn't exist
   15.  AzureMLEvaluationRunner: job construction, success/failure paths, artifact collection
   16.  EvaluationAgent uses FakeEvaluationRunner in LOCAL_DRY_RUN mode (via YoloEvaluator)
   17.  EvaluationAgent uses LocalYOLOEvaluationRunner in LOCAL_EVAL mode (via YoloEvaluator)
   18.  CLI --runner fake produces dry-run evaluation_request.json
   19.  CLI --runner local-yolo calls LocalYOLOEvaluationRunner (mocked)
   20.  CLI --runner unknown exits code 1
   21.  Richer policy: recall-only fail → COLLECT_MORE_DATA
   22.  Richer policy: precision-only fail → REVIEW_LABELS
   23.  EvaluationConfig loads from YAML correctly
   24.  EvaluationConfig threshold mapping via _config_to_policy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_mlops.contracts.azure_ml import AzureMLConfig
from agentic_mlops.contracts.evaluation import (
    EvaluationConfig,
    EvaluationInput,
    EvaluationMetrics,
    EvaluationMode,
    EvaluationOutput,
    EvaluationRecommendation,
)
from agentic_mlops.integrations.azure_ml_client import FakeAzureMLClientFactory, FakeMLClient
from agentic_mlops.tools.evaluation_runner import (
    AzureMLEvaluationRunner,
    FakeEvaluationRunner,
    LocalYOLOEvaluationRunner,
    _config_to_policy,
)
from agentic_mlops.workflows.policies import (
    PolicyThresholds,
    PromotionPolicy,
    evaluate_metrics_against_policy,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_save_dir(tmp_path: Path) -> Path:
    """Fake YOLO val save_dir with ultralytics 8.x artifact names."""
    save_dir = tmp_path / "runs" / "val" / "val_run"
    save_dir.mkdir(parents=True)
    (save_dir / "confusion_matrix.png").write_bytes(b"fake_cm")
    (save_dir / "BoxPR_curve.png").write_bytes(b"fake_pr")
    (save_dir / "BoxF1_curve.png").write_bytes(b"fake_f1")
    (save_dir / "confusion_matrix_normalized.png").write_bytes(b"fake_cm_norm")
    return save_dir


def _make_mock_results(save_dir: Path, map50=0.88, map50_95=0.60, mp=0.85, mr=0.80):
    """Build a mock Ultralytics val() results object."""
    mock_results = MagicMock()
    mock_results.save_dir = str(save_dir)
    mock_results.names = {0: "scratch", 1: "crack"}

    box = MagicMock()
    box.map50 = map50
    box.map = map50_95
    box.mp = mp
    box.mr = mr
    box.ap_class_index = [0, 1]
    box.ap50 = [map50 - 0.02, map50 + 0.01]
    box.ap = [map50_95 - 0.02, map50_95 + 0.01]
    box.p = [mp - 0.01, mp + 0.01]
    box.r = [mr - 0.01, mr + 0.01]

    mock_results.box = box
    return mock_results


@pytest.fixture
def mock_yolo_val(fake_save_dir: Path):
    """Mock YOLO class: .val() returns fake results pointing to fake_save_dir."""
    mock_model = MagicMock()
    mock_results = _make_mock_results(fake_save_dir)
    mock_model.val.return_value = mock_results
    MockYOLO = MagicMock(return_value=mock_model)
    return MockYOLO, mock_model, mock_results


def _make_input(
    tmp_path: Path,
    mode: EvaluationMode = EvaluationMode.LOCAL_EVAL,
    weights_exists: bool = True,
) -> EvaluationInput:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    data_yaml = dataset_path / "data.yaml"
    data_yaml.write_text("names:\n  0: scratch\n  1: crack\n", encoding="utf-8")

    weights = tmp_path / "best.pt"
    if weights_exists:
        weights.write_bytes(b"fake_weights")

    return EvaluationInput(
        dataset_path=str(dataset_path),
        data_yaml_path=str(data_yaml),
        weights_path=str(weights),
        workflow_id="wf_eval_test",
        mode=mode,
    )


# ── 1-2. FakeEvaluationRunner ─────────────────────────────────────────────────


def test_fake_runner_returns_deterministic_metrics(tmp_path):
    inp = _make_input(tmp_path, mode=EvaluationMode.LOCAL_DRY_RUN)
    result = FakeEvaluationRunner().run(inp, tmp_path / "artifacts")

    assert isinstance(result, EvaluationOutput)
    assert result.metrics.map50 == pytest.approx(0.862)
    assert result.metrics.map50_95 == pytest.approx(0.591)
    assert result.metrics.precision == pytest.approx(0.84)
    assert result.metrics.recall == pytest.approx(0.79)


def test_fake_runner_uses_override_metrics(tmp_path):
    override = EvaluationMetrics(map50=0.99, map50_95=0.88, precision=0.95, recall=0.93)
    inp = _make_input(tmp_path, mode=EvaluationMode.LOCAL_DRY_RUN)
    result = FakeEvaluationRunner(override_metrics=override).run(inp, tmp_path / "artifacts")

    assert result.metrics.map50 == pytest.approx(0.99)
    assert result.metrics.precision == pytest.approx(0.95)


# ── 3-5. LocalYOLOEvaluationRunner: val() call and metric extraction ──────────


def test_local_runner_calls_yolo_val_with_correct_args(tmp_path, mock_yolo_val, fake_save_dir):
    MockYOLO, mock_model, _ = mock_yolo_val
    inp = _make_input(tmp_path)

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        LocalYOLOEvaluationRunner().run(inp, tmp_path / "artifacts")

    MockYOLO.assert_called_once_with(inp.weights_path)
    call_kwargs = mock_model.val.call_args.kwargs
    assert call_kwargs["data"] == inp.data_yaml_path
    assert "imgsz" in call_kwargs
    assert "batch" in call_kwargs
    assert "device" in call_kwargs


def test_local_runner_extracts_global_metrics(tmp_path, mock_yolo_val, fake_save_dir):
    MockYOLO, _, mock_results = mock_yolo_val
    # Override map50=0.88, map50_95=0.60, mp=0.85, mr=0.80
    inp = _make_input(tmp_path)

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOEvaluationRunner().run(inp, tmp_path / "artifacts")

    assert result.metrics.map50 == pytest.approx(0.88)
    assert result.metrics.map50_95 == pytest.approx(0.60)
    assert result.metrics.precision == pytest.approx(0.85)
    assert result.metrics.recall == pytest.approx(0.80)


def test_local_runner_extracts_per_class_metrics(tmp_path, mock_yolo_val, fake_save_dir):
    MockYOLO, _, _ = mock_yolo_val
    inp = _make_input(tmp_path)

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOEvaluationRunner().run(inp, tmp_path / "artifacts")

    assert result.metrics.per_class_metrics


# ── 6-8. Artifact collection ──────────────────────────────────────────────────


def test_local_runner_collects_confusion_matrix(tmp_path, mock_yolo_val, fake_save_dir):
    MockYOLO, _, _ = mock_yolo_val
    inp = _make_input(tmp_path)
    artifacts_dir = tmp_path / "artifacts"

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOEvaluationRunner().run(inp, artifacts_dir)

    assert (artifacts_dir / "confusion_matrix.png").exists()
    assert any("confusion_matrix.png" in a for a in result.artifacts)


def test_local_runner_collects_pr_curve(tmp_path, mock_yolo_val, fake_save_dir):
    MockYOLO, _, _ = mock_yolo_val
    inp = _make_input(tmp_path)
    artifacts_dir = tmp_path / "artifacts"

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOEvaluationRunner().run(inp, artifacts_dir)

    assert (artifacts_dir / "BoxPR_curve.png").exists()
    assert any("BoxPR_curve.png" in a for a in result.artifacts)


def test_local_runner_handles_missing_artifacts_gracefully(tmp_path, fake_save_dir):
    (fake_save_dir / "confusion_matrix.png").unlink()
    (fake_save_dir / "BoxPR_curve.png").unlink()

    mock_model = MagicMock()
    mock_model.val.return_value = _make_mock_results(fake_save_dir)
    MockYOLO = MagicMock(return_value=mock_model)

    inp = _make_input(tmp_path)
    artifacts_dir = tmp_path / "artifacts"

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOEvaluationRunner().run(inp, artifacts_dir)

    assert result.success is not None
    assert not (artifacts_dir / "confusion_matrix.png").exists()
    assert not (artifacts_dir / "BoxPR_curve.png").exists()


# ── 9-10. evaluation_output.json ─────────────────────────────────────────────


def test_local_runner_writes_evaluation_output_json(tmp_path, mock_yolo_val, fake_save_dir):
    MockYOLO, _, _ = mock_yolo_val
    artifacts_dir = tmp_path / "artifacts"
    inp = _make_input(tmp_path)

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        LocalYOLOEvaluationRunner().run(inp, artifacts_dir)

    assert (artifacts_dir / "evaluation_output.json").exists()


def test_local_runner_output_json_has_required_fields(tmp_path, mock_yolo_val, fake_save_dir):
    MockYOLO, _, _ = mock_yolo_val
    artifacts_dir = tmp_path / "artifacts"
    inp = _make_input(tmp_path)

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        LocalYOLOEvaluationRunner().run(inp, artifacts_dir)

    data = json.loads((artifacts_dir / "evaluation_output.json").read_text(encoding="utf-8"))
    assert data["mode"] == "local_eval"
    assert data["runner"] == "local-yolo"
    assert data["model_path"] == inp.weights_path
    assert "started_at" in data
    assert "completed_at" in data
    assert "recommendation" in data


# ── 11-12. Promotion policy ───────────────────────────────────────────────────


def test_local_runner_promotes_with_good_metrics(tmp_path, fake_save_dir):
    mock_model = MagicMock()
    # map50=0.88 > 0.75, map50_95=0.60 > 0.50, mp=0.85 > 0.70, mr=0.80 > 0.70
    mock_model.val.return_value = _make_mock_results(fake_save_dir, 0.88, 0.60, 0.85, 0.80)
    MockYOLO = MagicMock(return_value=mock_model)

    inp = _make_input(tmp_path)
    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOEvaluationRunner().run(inp, tmp_path / "artifacts")

    assert result.recommendation == EvaluationRecommendation.PROMOTE_CANDIDATE
    assert result.success is True


def test_local_runner_retrains_with_bad_metrics(tmp_path, fake_save_dir):
    mock_model = MagicMock()
    # all metrics below default thresholds
    mock_model.val.return_value = _make_mock_results(fake_save_dir, 0.40, 0.20, 0.45, 0.40)
    MockYOLO = MagicMock(return_value=mock_model)

    inp = _make_input(tmp_path)
    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        result = LocalYOLOEvaluationRunner().run(inp, tmp_path / "artifacts")

    assert result.recommendation in {
        EvaluationRecommendation.RETRAIN,
        EvaluationRecommendation.COLLECT_MORE_DATA,
        EvaluationRecommendation.REVIEW_LABELS,
    }
    assert result.success is False


# ── 13-14. Error paths ────────────────────────────────────────────────────────


def test_local_runner_raises_when_ultralytics_not_installed(tmp_path):
    inp = _make_input(tmp_path)

    def _raise_import(*_):
        raise RuntimeError("ultralytics is not installed")

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", side_effect=_raise_import):
        result = LocalYOLOEvaluationRunner().run(inp, tmp_path / "artifacts")

    assert result.success is False
    assert any("ultralytics" in e for e in result.errors)


def test_local_runner_fails_when_weights_not_found(tmp_path):
    inp = _make_input(tmp_path, weights_exists=False)
    result = LocalYOLOEvaluationRunner().run(inp, tmp_path / "artifacts")

    assert result.success is False
    assert result.mode == EvaluationMode.LOCAL_EVAL
    assert any("not found" in e for e in result.errors)


# ── 15. AzureMLEvaluationRunner ───────────────────────────────────────────────


def _minimal_azure_config(**overrides) -> AzureMLConfig:
    defaults: dict = {
        "subscription_id": "sub-123",
        "resource_group": "rg-test",
        "workspace_name": "ws-test",
        "compute_name": "gpu-cluster",
        "environment": {
            "mode": "registered",
            "registered_environment": "azureml:yolo-env:1",
        },
    }
    defaults.update(overrides)
    return AzureMLConfig.model_validate(defaults)


class _EvalDownloadFactory:
    """FakeAzureMLClientFactory variant whose download writes metrics.json + a plot."""

    def __init__(
        self,
        job_status: str = "Completed",
        metrics: dict | None = None,
        write_metrics: bool = True,
    ) -> None:
        self._status = job_status
        self._metrics = metrics or {
            "map50": 0.90,
            "map50_95": 0.65,
            "precision": 0.88,
            "recall": 0.83,
            "per_class_metrics": {
                "scratch": {"precision": 0.9, "recall": 0.85, "map50": 0.92, "map50_95": 0.7},
            },
        }
        self._write_metrics = write_metrics
        self._clients: list[FakeMLClient] = []

    def create(self, config: AzureMLConfig) -> FakeMLClient:
        client = FakeMLClient(job_status=self._status)
        metrics, write_metrics = self._metrics, self._write_metrics

        def patched_download(name: str, output_name: str, download_path: str) -> None:
            out_dir = Path(download_path) / output_name
            out_dir.mkdir(parents=True, exist_ok=True)
            if write_metrics:
                (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (out_dir / "confusion_matrix.png").write_bytes(b"fake_cm")

        client.jobs.download = patched_download
        self._clients.append(client)
        return client

    @property
    def last_client(self) -> FakeMLClient | None:
        return self._clients[-1] if self._clients else None


def _make_fake_eval_job_obj(compute: str = "gpu-cluster") -> MagicMock:
    job = MagicMock()
    job.compute = compute
    job.command = (
        "python eval_yolo.py"
        " --weights ${{inputs.weights}}"
        " --dataset-path ${{inputs.dataset}}"
        " --data-yaml ${{inputs.data_yaml}}"
        " --imgsz 640"
        " --batch 8"
        " --device cpu"
        " --output-dir ${{outputs.eval_output}}"
    )
    dataset_input = MagicMock()
    dataset_input.type = "uri_folder"
    datayaml_input = MagicMock()
    datayaml_input.type = "uri_file"
    weights_input = MagicMock()
    weights_input.type = "uri_file"
    job.inputs = {"dataset": dataset_input, "data_yaml": datayaml_input, "weights": weights_input}
    eval_out = MagicMock()
    eval_out.type = "uri_folder"
    job.outputs = {"model_output": eval_out}
    return job


def _run_with_fake_eval_job(
    runner: AzureMLEvaluationRunner,
    inp: EvaluationInput,
    artifacts_dir: Path,
    fake_job: MagicMock | None = None,
):
    if fake_job is None:
        fake_job = _make_fake_eval_job_obj(compute=runner._config.compute_name)
    with patch.object(runner, "_build_job", return_value=fake_job):
        return runner.run(inp, artifacts_dir)


class TestAzureMLEvaluationRunner:
    def test_success_path_returns_metrics(self, tmp_path):
        cfg = _minimal_azure_config()
        factory = _EvalDownloadFactory(job_status="Completed")
        runner = AzureMLEvaluationRunner(cfg, client_factory=factory)
        inp = _make_input(tmp_path, mode=EvaluationMode.AZURE_EVAL)

        output = _run_with_fake_eval_job(runner, inp, tmp_path / "artifacts")

        assert output.metrics.map50 == pytest.approx(0.90)
        assert output.metrics.per_class_metrics["scratch"].recall == pytest.approx(0.85)
        assert output.runner == "azure-ml"
        assert output.mode == EvaluationMode.AZURE_EVAL

    def test_promote_candidate_recommendation_on_passing_metrics(self, tmp_path):
        cfg = _minimal_azure_config()
        factory = _EvalDownloadFactory(job_status="Completed")
        runner = AzureMLEvaluationRunner(cfg, client_factory=factory)
        inp = _make_input(tmp_path, mode=EvaluationMode.AZURE_EVAL)

        output = _run_with_fake_eval_job(runner, inp, tmp_path / "artifacts")

        assert output.recommendation == EvaluationRecommendation.PROMOTE_CANDIDATE
        assert output.success is True

    def test_failed_job_status_returns_failure(self, tmp_path):
        cfg = _minimal_azure_config()
        factory = _EvalDownloadFactory(job_status="Failed")
        runner = AzureMLEvaluationRunner(cfg, client_factory=factory)
        inp = _make_input(tmp_path, mode=EvaluationMode.AZURE_EVAL)

        output = _run_with_fake_eval_job(runner, inp, tmp_path / "artifacts")

        assert output.success is False

    def test_missing_metrics_json_returns_failure(self, tmp_path):
        cfg = _minimal_azure_config()
        factory = _EvalDownloadFactory(job_status="Completed", write_metrics=False)
        runner = AzureMLEvaluationRunner(cfg, client_factory=factory)
        inp = _make_input(tmp_path, mode=EvaluationMode.AZURE_EVAL)

        output = _run_with_fake_eval_job(runner, inp, tmp_path / "artifacts")

        assert output.success is False
        assert "metrics.json" in output.message

    def test_confusion_matrix_collected_as_artifact(self, tmp_path):
        cfg = _minimal_azure_config()
        factory = _EvalDownloadFactory(job_status="Completed")
        runner = AzureMLEvaluationRunner(cfg, client_factory=factory)
        inp = _make_input(tmp_path, mode=EvaluationMode.AZURE_EVAL)

        output = _run_with_fake_eval_job(runner, inp, tmp_path / "artifacts")

        assert any("confusion_matrix.png" in a for a in output.artifacts)

    def test_evaluation_output_json_written(self, tmp_path):
        cfg = _minimal_azure_config()
        factory = _EvalDownloadFactory(job_status="Completed")
        runner = AzureMLEvaluationRunner(cfg, client_factory=factory)
        inp = _make_input(tmp_path, mode=EvaluationMode.AZURE_EVAL)
        artifacts_dir = tmp_path / "artifacts"

        _run_with_fake_eval_job(runner, inp, artifacts_dir)

        assert (artifacts_dir / "evaluation_output.json").exists()

    def test_stream_logs_true_calls_jobs_stream(self, tmp_path):
        cfg = _minimal_azure_config()
        factory = _EvalDownloadFactory(job_status="Completed")
        runner = AzureMLEvaluationRunner(cfg, client_factory=factory)
        inp = _make_input(tmp_path, mode=EvaluationMode.AZURE_EVAL)

        _run_with_fake_eval_job(runner, inp, tmp_path / "artifacts")

        assert len(factory.last_client.jobs.streamed) == 1

    def test_job_command_and_inputs(self, tmp_path):
        cfg = _minimal_azure_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLEvaluationRunner(cfg, client_factory=factory)
        inp = _make_input(tmp_path, mode=EvaluationMode.AZURE_EVAL)

        mock_AssetTypes = MagicMock(URI_FOLDER="uri_folder", URI_FILE="uri_file")
        mock_InputOutputModes = MagicMock(RO_MOUNT="ro_mount", DOWNLOAD="download")

        def fake_Input(**kwargs):
            m = MagicMock()
            m.type = kwargs.get("type")
            m.path = kwargs.get("path")
            m.mode = kwargs.get("mode")
            return m

        def fake_Output(**kwargs):
            m = MagicMock()
            m.type = kwargs.get("type")
            return m

        def fake_command(**kwargs):
            j = MagicMock()
            j.command = kwargs.get("command", "")
            j.compute = kwargs.get("compute", "")
            j.inputs = kwargs.get("inputs", {})
            j.outputs = kwargs.get("outputs", {})
            return j

        mock_azure_ml = MagicMock(Input=fake_Input, Output=fake_Output, command=fake_command)
        mock_constants = MagicMock(
            AssetTypes=mock_AssetTypes, InputOutputModes=mock_InputOutputModes
        )
        mock_entities = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "azure.ai.ml": mock_azure_ml,
                "azure.ai.ml.constants": mock_constants,
                "azure.ai.ml.entities": mock_entities,
            },
        ):
            job = runner._build_job(inp, None)

        assert "${{inputs.weights}}" in job.command
        assert "eval_yolo.py" in job.command
        assert job.inputs["weights"].type == "uri_file"
        assert job.inputs["dataset"].type == "uri_folder"
        assert job.compute == "gpu-cluster"


# ── 16-17. YoloEvaluator dispatch ────────────────────────────────────────────


def test_yolo_evaluator_azure_eval_requires_injected_runner(tmp_path):
    from agentic_mlops.tools.yolo_evaluator import YoloEvaluator

    inp = _make_input(tmp_path, mode=EvaluationMode.AZURE_EVAL)
    with pytest.raises(RuntimeError, match="Azure ML evaluation requires"):
        YoloEvaluator().run(inp, tmp_path / "artifacts")


def test_yolo_evaluator_uses_injected_azure_runner(tmp_path):
    from agentic_mlops.tools.yolo_evaluator import YoloEvaluator

    cfg = _minimal_azure_config()
    factory = _EvalDownloadFactory(job_status="Completed")
    azure_runner = AzureMLEvaluationRunner(cfg, client_factory=factory)
    inp = _make_input(tmp_path, mode=EvaluationMode.AZURE_EVAL)

    fake_job = _make_fake_eval_job_obj(compute=cfg.compute_name)
    with patch.object(azure_runner, "_build_job", return_value=fake_job):
        result = YoloEvaluator(azure_runner=azure_runner).run(inp, tmp_path / "artifacts")

    assert result.runner == "azure-ml"
    assert result.mode == EvaluationMode.AZURE_EVAL


def test_yolo_evaluator_uses_fake_for_dry_run(tmp_path):
    from agentic_mlops.tools.yolo_evaluator import YoloEvaluator

    inp = _make_input(tmp_path, mode=EvaluationMode.LOCAL_DRY_RUN)
    result = YoloEvaluator().run(inp, tmp_path / "artifacts")

    assert result.mode == EvaluationMode.LOCAL_DRY_RUN
    assert result.runner == "fake"
    assert (tmp_path / "artifacts" / "evaluation_request.json").exists()


def test_yolo_evaluator_uses_local_runner_for_local_eval(tmp_path, mock_yolo_val, fake_save_dir):
    from agentic_mlops.tools.yolo_evaluator import YoloEvaluator

    MockYOLO, _, _ = mock_yolo_val
    inp = _make_input(tmp_path, mode=EvaluationMode.LOCAL_EVAL)

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        result = YoloEvaluator().run(inp, tmp_path / "artifacts")

    assert result.mode == EvaluationMode.LOCAL_EVAL
    assert result.runner == "local-yolo"


# ── 18-20. CLI --runner flag ──────────────────────────────────────────────────


@pytest.fixture
def cli_eval_dataset(tmp_path: Path):
    """Minimal dataset + weights for CLI evaluation tests."""
    ds = tmp_path / "ds"
    data_yaml = ds / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("names:\n  0: scratch\n", encoding="utf-8")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    return ds, data_yaml, weights


def test_cli_evaluate_runner_fake_produces_dry_run(tmp_path, cli_eval_dataset):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    ds, data_yaml, weights = cli_eval_dataset
    out_dir = tmp_path / "out"
    r = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--dataset-path",
            str(ds),
            "--data-yaml",
            str(data_yaml),
            "--weights-path",
            str(weights),
            "--output-dir",
            str(out_dir),
            "--runner",
            "fake",
        ],
    )
    assert r.exit_code == 0, r.output
    assert (out_dir / "evaluation_request.json").exists()


def test_cli_evaluate_runner_local_yolo_calls_local_runner(
    tmp_path, cli_eval_dataset, mock_yolo_val, fake_save_dir
):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    MockYOLO, _, _ = mock_yolo_val
    ds, data_yaml, weights = cli_eval_dataset
    out_dir = tmp_path / "out"

    with patch("agentic_mlops.tools.evaluation_runner._import_yolo", return_value=MockYOLO):
        r = CliRunner().invoke(
            app,
            [
                "evaluate",
                "--dataset-path",
                str(ds),
                "--data-yaml",
                str(data_yaml),
                "--weights-path",
                str(weights),
                "--output-dir",
                str(out_dir),
                "--runner",
                "local-yolo",
            ],
        )

    assert r.exit_code == 0, r.output
    assert (out_dir / "evaluation_output.json").exists()
    data = json.loads((out_dir / "evaluation_output.json").read_text())
    assert data["mode"] == "local_eval"


def test_cli_evaluate_runner_unknown_exits_nonzero(tmp_path, cli_eval_dataset):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    ds, data_yaml, weights = cli_eval_dataset
    r = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--dataset-path",
            str(ds),
            "--data-yaml",
            str(data_yaml),
            "--weights-path",
            str(weights),
            "--runner",
            "azure",
        ],
    )
    assert r.exit_code != 0


def test_cli_evaluate_azure_ml_without_azure_config_exits_1(tmp_path, cli_eval_dataset):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    ds, data_yaml, weights = cli_eval_dataset
    r = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--dataset-path",
            str(ds),
            "--data-yaml",
            str(data_yaml),
            "--weights-path",
            str(weights),
            "--runner",
            "azure-ml",
        ],
    )
    assert r.exit_code == 1
    assert "azure-config" in r.output


def test_cli_evaluate_azure_ml_with_config_succeeds(tmp_path, cli_eval_dataset):
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    ds, data_yaml, weights = cli_eval_dataset
    out_dir = tmp_path / "out"
    azure_cfg_path = tmp_path / "azure_ml.yaml"
    azure_cfg_path.write_text(
        "subscription_id: sub\nresource_group: rg\nworkspace_name: ws\n"
        "compute_name: c\nenvironment:\n  mode: registered\n"
        "  registered_environment: azureml:e:1\n",
        encoding="utf-8",
    )

    factory = _EvalDownloadFactory(job_status="Completed")
    fake_job = _make_fake_eval_job_obj(compute="c")
    original_run = AzureMLEvaluationRunner.run

    def fake_runner_run(self_runner, inp, artifacts_dir):
        self_runner._factory = factory
        with patch.object(self_runner, "_build_job", return_value=fake_job):
            return original_run(self_runner, inp, artifacts_dir)

    with patch.object(AzureMLEvaluationRunner, "run", new=fake_runner_run):
        r = CliRunner().invoke(
            app,
            [
                "evaluate",
                "--dataset-path",
                str(ds),
                "--data-yaml",
                str(data_yaml),
                "--weights-path",
                str(weights),
                "--output-dir",
                str(out_dir),
                "--runner",
                "azure-ml",
                "--azure-config",
                str(azure_cfg_path),
            ],
        )

    assert r.exit_code == 0, r.output
    assert (out_dir / "evaluation_output.json").exists()


# ── 21-22. Richer promotion policy ───────────────────────────────────────────


def test_recall_only_fail_produces_collect_more_data(tmp_path):
    """Only recall below threshold → COLLECT_MORE_DATA."""
    metrics = EvaluationMetrics(
        map50=0.80,
        map50_95=0.55,
        precision=0.75,
        recall=0.60,  # below 0.70 threshold
    )
    policy = PromotionPolicy(
        thresholds=PolicyThresholds(
            map50_min=0.75,
            map50_95_min=0.50,
            precision_min=0.70,
            recall_min=0.70,
        )
    )
    rec, passed, failed = evaluate_metrics_against_policy(metrics, policy)
    assert rec == EvaluationRecommendation.COLLECT_MORE_DATA


def test_precision_only_fail_produces_review_labels(tmp_path):
    """Only precision below threshold → REVIEW_LABELS."""
    metrics = EvaluationMetrics(
        map50=0.80,
        map50_95=0.55,
        precision=0.60,  # below 0.70 threshold
        recall=0.75,
    )
    policy = PromotionPolicy(
        thresholds=PolicyThresholds(
            map50_min=0.75,
            map50_95_min=0.50,
            precision_min=0.70,
            recall_min=0.70,
        )
    )
    rec, passed, failed = evaluate_metrics_against_policy(metrics, policy)
    assert rec == EvaluationRecommendation.REVIEW_LABELS


# ── 23-24. EvaluationConfig ───────────────────────────────────────────────────


def test_evaluation_config_loads_from_yaml(tmp_path):
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(
        "imgsz: 320\nbatch: 4\ndevice: cpu\n"
        "metrics:\n  min_map50: 0.85\n  min_recall: 0.75\n"
        "critical_classes:\n  crack:\n    min_recall: 0.80\n",
        encoding="utf-8",
    )
    cfg = EvaluationConfig.from_yaml(cfg_path)

    assert cfg.imgsz == 320
    assert cfg.batch == 4
    assert cfg.metrics.min_map50 == pytest.approx(0.85)
    assert cfg.metrics.min_recall == pytest.approx(0.75)
    assert "crack" in cfg.critical_classes
    assert cfg.critical_classes["crack"].min_recall == pytest.approx(0.80)


def test_config_to_policy_maps_thresholds(tmp_path):
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(
        "metrics:\n  min_map50: 0.90\n  min_map50_95: 0.60\n"
        "  min_precision: 0.85\n  min_recall: 0.80\n"
        "critical_classes:\n  crack:\n    min_recall: 0.75\n",
        encoding="utf-8",
    )
    cfg = EvaluationConfig.from_yaml(cfg_path)
    policy = _config_to_policy(cfg)

    assert policy.thresholds.map50_min == pytest.approx(0.90)
    assert policy.thresholds.map50_95_min == pytest.approx(0.60)
    assert policy.thresholds.precision_min == pytest.approx(0.85)
    assert policy.thresholds.recall_min == pytest.approx(0.80)
    assert "crack" in policy.critical_classes
    assert policy.per_class_recall_min["crack"] == pytest.approx(0.75)
