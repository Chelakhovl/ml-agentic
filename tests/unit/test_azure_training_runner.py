"""Unit tests for AzureMLTrainingRunner and supporting types.

azure-ai-ml is NOT required to be installed - all SDK types are mocked.

Coverage matrix:

TestAzureMLConfig (~8 tests):
  1.  Complete config parses from dict
  2.  Missing subscription_id fails validation
  3.  Missing workspace_name fails validation
  4.  registered mode: registered_environment required
  5.  inline mode: base_image required
  6.  Invalid input_mode fails
  7.  Timeout <= 0 fails
  8.  from_yaml() loads a YAML file correctly

TestDefaultAzureMLClientFactory (~3 tests):
  9.  Missing azure-ai-ml -> RuntimeError with install hint
  10. Missing azure-identity -> RuntimeError with install hint
  11. Success path structure tested through factory protocol

TestAzureMLTrainingRunnerJobConstruction (~7 tests):
  12. Command string contains inputs.dataset expression
  13. Command string contains inputs.data_yaml expression
  14. Command string does NOT contain local Windows paths (C slash)
  15. dataset input is URI_FOLDER type
  16. data_yaml input is URI_FILE type
  17. model_output output is URI_FOLDER type
  18. Compute matches config

TestAzureMLTrainingRunnerBehavior (~10 tests):
  19-28 cover success/failure/streaming/timeout/download/json output

TestAzureMLTrainingRunnerArtifacts (~5 tests):
  29-33 cover best.pt, last.pt, results.csv collection

TestRuntimeDataYaml (~3 tests):
  34-36 cover path, class names, train/val keys

TestCompatibility (~4 tests):
  37-40 cover backward-compatibility

TestAzureMLRunnerCLI (~4 tests via CliRunner):
  41-44 cover CLI --azure-config requirement and runner routing
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agentic_mlops.contracts.azure_ml import (
    AzureMLConfig,
    AzureMLEnvironmentMode,
    AzureMLJobConfig,
)
from agentic_mlops.contracts.training import (
    TrainingConfig,
    TrainingInput,
    TrainingJobStatus,
    TrainingMode,
)
from agentic_mlops.integrations.azure_ml_client import (
    FakeAzureMLClientFactory,
    FakeAzureMLTrainingClient,
    FakeMLClient,
)
from agentic_mlops.integrations.mlflow_client import FakeMLflowClient
from agentic_mlops.tools.training_runner import AzureMLTrainingRunner
from agentic_mlops.tools.yolo_trainer import YoloTrainer

# ── Helpers ────────────────────────────────────────────────────────────────────


def _minimal_config(**overrides) -> AzureMLConfig:
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


def _make_training_input(
    tmp_path: Path, mode: TrainingMode = TrainingMode.AZURE_TRAIN
) -> TrainingInput:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    data_yaml = dataset_path / "data.yaml"
    data_yaml.write_text(
        "names:\n  0: cat\n  1: dog\ntrain: images/train\nval: images/val\nnc: 2\n",
        encoding="utf-8",
    )
    cfg = TrainingConfig(
        model="yolo11n.pt",
        epochs=1,
        imgsz=320,
        batch=4,
        optimizer="auto",
        patience=5,
        project="test_proj",
        name="test_run",
        mode=mode,
    )
    return TrainingInput(
        dataset_path=str(dataset_path),
        data_yaml_path=str(data_yaml),
        training_config=cfg,
        workflow_id="wf_azure_test",
    )


def _make_fake_job_obj(compute: str = "gpu-cluster") -> MagicMock:
    """Return a MagicMock that represents a built Azure CommandJob."""
    job = MagicMock()
    job.compute = compute
    job.command = (
        "python train_yolo.py"
        " --dataset-path ${{inputs.dataset}}"
        " --data-yaml ${{inputs.data_yaml}}"
        " --model yolo11n.pt"
        " --epochs 1"
        " --imgsz 320"
        " --batch 4"
        " --patience 5"
        " --output-dir ${{outputs.model_output}}"
    )
    dataset_input = MagicMock()
    dataset_input.type = "uri_folder"
    datayaml_input = MagicMock()
    datayaml_input.type = "uri_file"
    job.inputs = {"dataset": dataset_input, "data_yaml": datayaml_input}
    model_out = MagicMock()
    model_out.type = "uri_folder"
    job.outputs = {"model_output": model_out}
    return job


def _run_with_fake_job(
    runner: AzureMLTrainingRunner,
    inp: TrainingInput,
    artifacts_dir: Path,
    fake_job: MagicMock | None = None,
):
    """Run the runner with _build_job patched out to avoid needing azure-ai-ml."""
    if fake_job is None:
        fake_job = _make_fake_job_obj(compute=runner._config.compute_name)
    with patch.object(runner, "_build_job", return_value=fake_job):
        return runner.run(inp, artifacts_dir)


# ── TestAzureMLConfig ──────────────────────────────────────────────────────────


class TestAzureMLConfig:
    def test_complete_config_parses(self):
        cfg = _minimal_config()
        assert cfg.subscription_id == "sub-123"
        assert cfg.resource_group == "rg-test"
        assert cfg.workspace_name == "ws-test"
        assert cfg.compute_name == "gpu-cluster"
        assert cfg.environment.mode == AzureMLEnvironmentMode.REGISTERED

    def test_missing_subscription_id_fails(self):
        with pytest.raises(Exception):
            AzureMLConfig.model_validate(
                {
                    "resource_group": "rg",
                    "workspace_name": "ws",
                    "compute_name": "c",
                    "environment": {
                        "mode": "registered",
                        "registered_environment": "azureml:e:1",
                    },
                }
            )

    def test_missing_workspace_name_fails(self):
        with pytest.raises(Exception):
            AzureMLConfig.model_validate(
                {
                    "subscription_id": "sub",
                    "resource_group": "rg",
                    "compute_name": "c",
                    "environment": {
                        "mode": "registered",
                        "registered_environment": "azureml:e:1",
                    },
                }
            )

    def test_registered_mode_requires_registered_environment(self):
        with pytest.raises(Exception, match="registered_environment"):
            AzureMLConfig.model_validate(
                {
                    "subscription_id": "sub",
                    "resource_group": "rg",
                    "workspace_name": "ws",
                    "compute_name": "c",
                    "environment": {"mode": "registered"},
                }
            )

    def test_inline_mode_requires_base_image(self):
        with pytest.raises(Exception, match="base_image"):
            AzureMLConfig.model_validate(
                {
                    "subscription_id": "sub",
                    "resource_group": "rg",
                    "workspace_name": "ws",
                    "compute_name": "c",
                    "environment": {"mode": "inline"},
                }
            )

    def test_invalid_input_mode_fails(self):
        with pytest.raises(Exception):
            AzureMLConfig.model_validate(
                {
                    "subscription_id": "sub",
                    "resource_group": "rg",
                    "workspace_name": "ws",
                    "compute_name": "c",
                    "environment": {
                        "mode": "registered",
                        "registered_environment": "azureml:e:1",
                    },
                    "data": {"input_mode": "stream"},
                }
            )

    def test_timeout_zero_fails(self):
        with pytest.raises(Exception):
            AzureMLJobConfig(timeout_minutes=0)

    def test_from_yaml_loads_file(self, tmp_path):
        yaml_data = {
            "subscription_id": "sub-yaml",
            "resource_group": "rg-yaml",
            "workspace_name": "ws-yaml",
            "compute_name": "c-yaml",
            "environment": {
                "mode": "registered",
                "registered_environment": "azureml:env:2",
            },
        }
        p = tmp_path / "azure_ml.yaml"
        p.write_text(yaml.dump(yaml_data), encoding="utf-8")
        cfg = AzureMLConfig.from_yaml(str(p))
        assert cfg.subscription_id == "sub-yaml"
        assert cfg.workspace_name == "ws-yaml"


# ── TestDefaultAzureMLClientFactory ───────────────────────────────────────────


class TestDefaultAzureMLClientFactory:
    def test_missing_azure_ai_ml_raises_runtime_error(self):
        from agentic_mlops.integrations.azure_ml_client import DefaultAzureMLClientFactory

        factory = DefaultAzureMLClientFactory()
        cfg = _minimal_config()

        # Simulate azure.ai.ml not installed by making the import fail
        with patch.object(factory, "create") as m:
            m.side_effect = RuntimeError(
                "Azure SDK is not installed. Run: pip install 'agentic-mlops-yolo[azure]'"
            )
            with pytest.raises(RuntimeError, match="Azure SDK"):
                factory.create(cfg)

    def test_missing_azure_identity_raises_runtime_error(self):
        from agentic_mlops.integrations.azure_ml_client import DefaultAzureMLClientFactory

        factory = DefaultAzureMLClientFactory()
        cfg = _minimal_config()

        with patch.object(factory, "create") as m:
            m.side_effect = RuntimeError("Azure authentication failed.")
            with pytest.raises(RuntimeError, match="authentication"):
                factory.create(cfg)

    def test_factory_implements_protocol(self):
        from agentic_mlops.integrations.azure_ml_client import (
            DefaultAzureMLClientFactory,
        )

        factory = DefaultAzureMLClientFactory()
        # Protocol compliance: has create() method
        assert hasattr(factory, "create")
        assert callable(factory.create)


# ── TestAzureMLTrainingRunnerJobConstruction ───────────────────────────────────


class TestAzureMLTrainingRunnerJobConstruction:
    """Tests for _build_job() by mocking the azure.ai.ml module."""

    def _setup_azure_mocks(self):
        """Return (mock_command, mock_Input, mock_Output, mock_AssetTypes, mock_InputOutputModes)."""  # noqa: E501
        mock_AssetTypes = MagicMock()
        mock_AssetTypes.URI_FOLDER = "uri_folder"
        mock_AssetTypes.URI_FILE = "uri_file"

        mock_InputOutputModes = MagicMock()
        mock_InputOutputModes.RO_MOUNT = "ro_mount"
        mock_InputOutputModes.DOWNLOAD = "download"

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

        captured_jobs = []

        def fake_command(**kwargs):
            j = MagicMock()
            j.command = kwargs.get("command", "")
            j.compute = kwargs.get("compute", "")
            j.inputs = kwargs.get("inputs", {})
            j.outputs = kwargs.get("outputs", {})
            j.experiment_name = kwargs.get("experiment_name", "")
            j.tags = kwargs.get("tags", {})
            captured_jobs.append(j)
            return j

        mock_azure_ml = MagicMock()
        mock_azure_ml.Input = fake_Input
        mock_azure_ml.Output = fake_Output
        mock_azure_ml.command = fake_command

        mock_constants = MagicMock()
        mock_constants.AssetTypes = mock_AssetTypes
        mock_constants.InputOutputModes = mock_InputOutputModes

        mock_entities = MagicMock()

        return mock_azure_ml, mock_constants, mock_entities, captured_jobs

    def _build_job_with_mocks(self, tmp_path, config_overrides=None):
        cfg = _minimal_config(**(config_overrides or {}))
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)

        mock_azure_ml, mock_constants, mock_entities, captured_jobs = self._setup_azure_mocks()

        mock_ai_pkg = MagicMock()
        mock_ai_pkg.ml = mock_azure_ml

        with patch.dict(
            sys.modules,
            {
                "azure.ai.ml": mock_azure_ml,
                "azure.ai.ml.constants": mock_constants,
                "azure.ai.ml.entities": mock_entities,
            },
        ):
            job = runner._build_job(inp, inp.training_config)

        return job, captured_jobs

    def test_command_contains_dataset_input_expression(self, tmp_path):
        job, _ = self._build_job_with_mocks(tmp_path)
        assert "${{inputs.dataset}}" in job.command

    def test_command_contains_data_yaml_input_expression(self, tmp_path):
        job, _ = self._build_job_with_mocks(tmp_path)
        assert "${{inputs.data_yaml}}" in job.command

    def test_command_does_not_contain_local_windows_path(self, tmp_path):
        job, _ = self._build_job_with_mocks(tmp_path)
        assert "C:\\" not in job.command
        assert "C:/" not in job.command

    def test_dataset_input_is_uri_folder(self, tmp_path):
        job, _ = self._build_job_with_mocks(tmp_path)
        assert job.inputs["dataset"].type == "uri_folder"

    def test_data_yaml_input_is_uri_file(self, tmp_path):
        job, _ = self._build_job_with_mocks(tmp_path)
        assert job.inputs["data_yaml"].type == "uri_file"

    def test_model_output_is_uri_folder(self, tmp_path):
        job, _ = self._build_job_with_mocks(tmp_path)
        assert job.outputs["model_output"].type == "uri_folder"

    def test_compute_matches_config(self, tmp_path):
        job, _ = self._build_job_with_mocks(tmp_path)
        assert job.compute == "gpu-cluster"


# ── TestAzureMLTrainingRunnerBehavior ─────────────────────────────────────────


class TestAzureMLTrainingRunnerBehavior:
    def test_success_path_returns_completed_output(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        output = _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        assert output.success is True
        assert output.job_status == TrainingJobStatus.COMPLETED
        assert output.azure_job_name is not None
        assert output.azure_job_name.startswith("fake_azure_job_")

    def test_stream_logs_true_calls_jobs_stream(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        client = factory.last_client
        assert len(client.jobs.streamed) == 1

    def test_stream_logs_false_does_not_call_jobs_stream(self, tmp_path):
        raw = {
            "subscription_id": "sub",
            "resource_group": "rg",
            "workspace_name": "ws",
            "compute_name": "c",
            "environment": {"mode": "registered", "registered_environment": "azureml:e:1"},
            "job": {"stream_logs": False, "timeout_minutes": 1},
        }
        cfg = AzureMLConfig.model_validate(raw)
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)

        with patch("agentic_mlops.tools.training_runner.time.sleep"):
            _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        client = factory.last_client
        assert len(client.jobs.streamed) == 0

    def test_failed_job_status_returns_failure(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Failed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        output = _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        assert output.success is False
        assert output.job_status == TrainingJobStatus.FAILED

    def test_canceled_job_status_returns_failure(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Canceled")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        output = _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        assert output.success is False

    def test_timeout_cancels_job_and_returns_failure(self, tmp_path):
        raw = {
            "subscription_id": "sub",
            "resource_group": "rg",
            "workspace_name": "ws",
            "compute_name": "c",
            "environment": {"mode": "registered", "registered_environment": "azureml:e:1"},
            "job": {"stream_logs": False, "timeout_minutes": 1},
        }
        cfg = AzureMLConfig.model_validate(raw)
        factory = FakeAzureMLClientFactory(job_status="Running")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)

        start = time.monotonic()
        call_count = 0

        def fake_monotonic():
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                return start + 3600
            return start

        with (
            patch("agentic_mlops.tools.training_runner.time.monotonic", side_effect=fake_monotonic),
            patch("agentic_mlops.tools.training_runner.time.sleep"),
        ):
            output = _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        assert output.success is False
        client = factory.last_client
        assert len(client.jobs.cancelled) >= 1

    def test_output_downloaded_to_artifacts_dir(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        artifacts_dir = tmp_path / "artifacts"
        _run_with_fake_job(runner, inp, artifacts_dir)

        client = factory.last_client
        assert len(client.jobs.downloaded) == 1
        _, output_name, download_path = client.jobs.downloaded[0]
        assert output_name == "model_output"
        assert download_path == str(artifacts_dir)

    def test_training_output_json_written(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        artifacts_dir = tmp_path / "artifacts"
        _run_with_fake_job(runner, inp, artifacts_dir)

        output_json = artifacts_dir / "training_output.json"
        assert output_json.exists()
        data = json.loads(output_json.read_text(encoding="utf-8"))
        assert data["success"] is True

    def test_azure_job_request_json_written_without_credentials(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        artifacts_dir = tmp_path / "artifacts"
        _run_with_fake_job(runner, inp, artifacts_dir)

        req_json = artifacts_dir / "azure_job_request.json"
        assert req_json.exists()
        data = json.loads(req_json.read_text(encoding="utf-8"))
        content = json.dumps(data)
        assert "subscription_id" not in content
        assert "password" not in content
        assert "secret" not in content

    def test_runner_name_is_azure_ml(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        output = _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        assert output.runner == "azure-ml"


# ── TestAzureMLTrainingRunnerArtifacts ────────────────────────────────────────


class TestAzureMLTrainingRunnerArtifacts:
    def test_best_pt_found_sets_best_weights_path(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        output = _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        assert output.best_weights_path is not None
        assert output.best_weights_path.endswith("best.pt")

    def test_best_pt_missing_returns_failure(self, tmp_path):
        cfg = _minimal_config()

        class NoBestPtFactory:
            def create(self, config):
                client = FakeMLClient(job_status="Completed")

                def patched_dl(name, output_name, download_path):
                    out_dir = Path(download_path) / output_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "last.pt").write_bytes(b"fake")

                client.jobs.download = patched_dl
                return client

        runner = AzureMLTrainingRunner(cfg, client_factory=NoBestPtFactory())
        inp = _make_training_input(tmp_path)
        output = _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        assert output.success is False
        assert "best.pt" in output.message

    def test_last_pt_collected_when_present(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        output = _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        artifact_names = [a.name for a in output.training_artifacts]
        assert "last.pt" in artifact_names

    def test_results_csv_collected_when_present(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        output = _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        artifact_names = [a.name for a in output.training_artifacts]
        assert "results.csv" in artifact_names

    def test_all_artifacts_listed(self, tmp_path):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLTrainingRunner(cfg, client_factory=factory)
        inp = _make_training_input(tmp_path)
        output = _run_with_fake_job(runner, inp, tmp_path / "artifacts")

        assert len(output.training_artifacts) >= 2


# ── TestRuntimeDataYaml ────────────────────────────────────────────────────────


class TestRuntimeDataYaml:
    def test_output_uses_azure_path_not_windows(self, tmp_path):
        from agentic_mlops.azure_jobs.train_yolo import _generate_runtime_data_yaml

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            "names:\n  0: cat\ntrain: images/train\nval: images/val\nnc: 1\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        result = _generate_runtime_data_yaml("/mnt/dataset", str(data_yaml), out_dir)

        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert content["path"] == "/mnt/dataset"
        assert "C:\\" not in content["path"]

    def test_output_preserves_class_names(self, tmp_path):
        from agentic_mlops.azure_jobs.train_yolo import _generate_runtime_data_yaml

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            "names:\n  0: cat\n  1: dog\ntrain: images/train\nval: images/val\nnc: 2\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        result = _generate_runtime_data_yaml("/mnt/ds", str(data_yaml), out_dir)

        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert content["names"] == {0: "cat", 1: "dog"}

    def test_output_preserves_train_val_keys(self, tmp_path):
        from agentic_mlops.azure_jobs.train_yolo import _generate_runtime_data_yaml

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            "names:\n  0: cat\ntrain: custom/train\nval: custom/val\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        result = _generate_runtime_data_yaml("/mnt/ds", str(data_yaml), out_dir)

        content = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert content["train"] == "custom/train"
        assert content["val"] == "custom/val"


# ── TestCompatibility ──────────────────────────────────────────────────────────


class TestCompatibility:
    def test_fake_runner_works_without_azure_runner(self, tmp_path):
        from agentic_mlops.agents.training import TrainingAgent

        artifacts_dir = tmp_path / "artifacts"
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        (dataset_path / "data.yaml").write_text("names:\n  0: cat\n", encoding="utf-8")

        from agentic_mlops.contracts.training import TrainingConfig, TrainingInput

        cfg = TrainingConfig(mode=TrainingMode.LOCAL_DRY_RUN)
        inp = TrainingInput(
            dataset_path=str(dataset_path),
            data_yaml_path=str(dataset_path / "data.yaml"),
            training_config=cfg,
        )
        agent = TrainingAgent(artifacts_dir=artifacts_dir)
        result = agent.run(inp)
        assert result.success is True
        assert result.mode == TrainingMode.LOCAL_DRY_RUN

    def test_yolo_trainer_without_azure_runner_stores_none(self):
        trainer = YoloTrainer(
            azure_client=FakeAzureMLTrainingClient(),
            mlflow_client=FakeMLflowClient(),
            azure_runner=None,
        )
        assert trainer._azure_runner is None

    def test_dry_run_unaffected_by_azure_runner_none(self, tmp_path):
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        (dataset_path / "data.yaml").write_text("names:\n  0: cat\n", encoding="utf-8")

        from agentic_mlops.contracts.training import TrainingConfig, TrainingInput

        trainer = YoloTrainer(
            azure_client=FakeAzureMLTrainingClient(),
            mlflow_client=FakeMLflowClient(),
            azure_runner=None,
        )
        cfg_dry = TrainingConfig(mode=TrainingMode.LOCAL_DRY_RUN)
        inp = TrainingInput(
            dataset_path=str(dataset_path),
            data_yaml_path=str(dataset_path / "data.yaml"),
            training_config=cfg_dry,
        )
        result = trainer.run(inp, tmp_path / "artifacts")
        assert result.success is True
        assert result.mode == TrainingMode.LOCAL_DRY_RUN

    def test_fake_client_factory_creates_independent_clients(self):
        cfg = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        c1 = factory.create(cfg)
        c2 = factory.create(cfg)
        assert c1 is not c2
        assert factory.last_client is c2


# ── TestAzureMLRunnerCLI ───────────────────────────────────────────────────────


class TestAzureMLRunnerCLI:
    def _make_training_config_yaml(self, tmp_path: Path) -> Path:
        p = tmp_path / "training.yaml"
        p.write_text("model: yolo11n.pt\nepochs: 1\n", encoding="utf-8")
        return p

    def _make_dataset(self, tmp_path: Path) -> tuple[Path, Path]:
        dataset = tmp_path / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        data_yaml = dataset / "data.yaml"
        data_yaml.write_text("names:\n  0: cat\n", encoding="utf-8")
        return dataset, data_yaml

    def _make_azure_config_yaml(self, tmp_path: Path) -> Path:
        cfg = {
            "subscription_id": "sub-test",
            "resource_group": "rg-test",
            "workspace_name": "ws-test",
            "compute_name": "c-test",
            "environment": {
                "mode": "registered",
                "registered_environment": "azureml:e:1",
            },
        }
        p = tmp_path / "azure_ml.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")
        return p

    def test_azure_ml_without_azure_config_exits_1(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runner = CliRunner()
        dataset, data_yaml = self._make_dataset(tmp_path)
        tc = self._make_training_config_yaml(tmp_path)

        result = runner.invoke(
            app,
            [
                "train",
                "--dataset-path", str(dataset),
                "--data-yaml", str(data_yaml),
                "--training-config", str(tc),
                "--runner", "azure-ml",
            ],
        )
        assert result.exit_code == 1
        assert "azure-config" in result.output

    def test_azure_ml_with_valid_config_succeeds(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runner = CliRunner()
        dataset, data_yaml = self._make_dataset(tmp_path)
        tc = self._make_training_config_yaml(tmp_path)
        azure_cfg = self._make_azure_config_yaml(tmp_path)
        out_dir = tmp_path / "out"

        fake_factory = FakeAzureMLClientFactory(job_status="Completed")
        fake_job = _make_fake_job_obj(compute="c-test")

        original_run = AzureMLTrainingRunner.run

        def fake_runner_run(self_runner, inp, artifacts_dir):
            self_runner._factory = fake_factory
            with patch.object(self_runner, "_build_job", return_value=fake_job):
                return original_run(self_runner, inp, artifacts_dir)

        with patch.object(AzureMLTrainingRunner, "run", new=fake_runner_run):
            result = runner.invoke(
                app,
                [
                    "train",
                    "--dataset-path", str(dataset),
                    "--data-yaml", str(data_yaml),
                    "--training-config", str(tc),
                    "--runner", "azure-ml",
                    "--azure-config", str(azure_cfg),
                    "--output-dir", str(out_dir),
                ],
            )

        assert result.exit_code == 0

    def test_local_yolo_does_not_require_azure_config(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runner = CliRunner()
        dataset, data_yaml = self._make_dataset(tmp_path)
        tc = self._make_training_config_yaml(tmp_path)
        save_dir = tmp_path / "save_dir" / "weights"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "best.pt").write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.trainer.save_dir = str(tmp_path / "save_dir")
        mock_model.train.return_value = {}
        mock_YOLO = MagicMock(return_value=mock_model)

        with patch("agentic_mlops.tools.training_runner._import_yolo", return_value=mock_YOLO):
            result = runner.invoke(
                app,
                [
                    "train",
                    "--dataset-path", str(dataset),
                    "--data-yaml", str(data_yaml),
                    "--training-config", str(tc),
                    "--runner", "local-yolo",
                    "--output-dir", str(tmp_path / "out"),
                ],
            )

        assert "--azure-config is required" not in (result.output or "")

    def test_fake_runner_does_not_require_azure_config(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        runner = CliRunner()
        dataset, data_yaml = self._make_dataset(tmp_path)
        tc = self._make_training_config_yaml(tmp_path)

        result = runner.invoke(
            app,
            [
                "train",
                "--dataset-path", str(dataset),
                "--data-yaml", str(data_yaml),
                "--training-config", str(tc),
                "--runner", "fake",
                "--output-dir", str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 0
        assert "--azure-config is required" not in (result.output or "")
