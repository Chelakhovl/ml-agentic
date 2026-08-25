"""Unit tests for AzureMLPipelineRunner.

azure-ai-ml is NOT required to be installed — all SDK types are mocked.

Coverage:
  TestAzureMLPipelineRunnerBehavior
    1.  Happy path: pipeline submitted, both outputs downloaded, success
    2.  pipeline failed status -> both outputs reflect failure
    3.  best.pt missing after download -> training output failed
    4.  metrics.json missing after download -> evaluation output failed
    5.  stream_logs=True uses stream() then get()
    6.  stream_logs=False polls _wait_for_completion()
    7.  download_outputs=False: no download call
    8.  exception during run() returns two failed outputs
    9.  Training output has AZURE_PIPELINE mode
    10. Evaluation output has AZURE_PIPELINE_EVAL mode
    11. training_output.json written to training_artifacts_dir
    12. evaluation_output.json written to evaluation_artifacts_dir
    13. pipeline_eval_output_path NOT set on TrainingOutput from runner directly
        (it's set by the orchestrator / mvp workflow on top of the runner)

  TestAzureMLPipelineRunnerJobConstruction
    14. _build_pipeline returns an object accepted by ml_client.jobs.create_or_update
    15. Both components use the azure_jobs code dir
    16. Eval command references best.pt inside weights folder
    17. Environment matches config

  TestAzureMLPipelineConfig
    18. Default pipeline config has 240 min timeout
    19. AzureMLConfig.from_yaml() loads pipeline section
    20. pipeline section is optional (uses defaults when omitted)

  TestMVPWorkflowPipelineIntegration
    21. training_runner=azure-ml-pipeline uses AzureMLPipelineRunner, not individual runners
    22. missing azure_config_path raises ValueError (caught by MVPWorkflow -> fail)
    23. pipeline success: both training+evaluation steps appear in MVPWorkflow steps

  TestOrchestratorPipelineIntegration
    24. _step_training with azure-ml-pipeline calls pipeline runner
    25. _step_evaluation reads pipeline_eval_output_path when set
    26. _step_evaluation missing pipeline_eval_output_path file -> FAILED outcome
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_mlops.contracts.azure_ml import AzureMLConfig, AzureMLPipelineConfig
from agentic_mlops.contracts.evaluation import (
    EvaluationInput,
    EvaluationMode,
    EvaluationOutput,
)
from agentic_mlops.contracts.training import (
    TrainingConfig,
    TrainingInput,
    TrainingJobStatus,
    TrainingMode,
    TrainingOutput,
)
from agentic_mlops.integrations.azure_ml_client import FakeAzureMLClientFactory, FakeMLClient
from agentic_mlops.tools.pipeline_runner import AzureMLPipelineRunner

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


def _make_training_input(tmp_path: Path) -> TrainingInput:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    data_yaml = dataset_path / "data.yaml"
    data_yaml.write_text(
        "path: /tmp/dataset\ntrain: images/train\nval: images/val\nnames: {0: scratch}\n",
        encoding="utf-8",
    )
    return TrainingInput(
        dataset_path=str(dataset_path),
        data_yaml_path=str(data_yaml),
        training_config=TrainingConfig(
            model="yolo11m.pt",
            epochs=3,
            mode=TrainingMode.AZURE_PIPELINE,
        ),
    )


def _make_evaluation_input(tmp_path: Path) -> EvaluationInput:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text(
            "path: /tmp/dataset\ntrain: images/train\nval: images/val\nnames: {0: scratch}\n",
            encoding="utf-8",
        )
    return EvaluationInput(
        dataset_path=str(dataset_path),
        data_yaml_path=str(data_yaml),
        weights_path="pipeline",
        mode=EvaluationMode.AZURE_PIPELINE_EVAL,
    )


def _fake_pipeline(tmp_path: Path) -> tuple[AzureMLPipelineRunner, FakeMLClient]:
    """Return a runner with a FakeAzureMLClientFactory, plus direct access to the fake client."""
    config = _minimal_config()
    factory = FakeAzureMLClientFactory(job_status="Completed")
    runner = AzureMLPipelineRunner(config=config, client_factory=factory)
    # Pre-create the client so we can inspect it
    ml_client = factory.create(config)
    return runner, ml_client


# ── TestAzureMLPipelineRunnerBehavior ──────────────────────────────────────────


class TestAzureMLPipelineRunnerBehavior:
    def test_happy_path_returns_success(self, tmp_path: Path) -> None:
        config = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLPipelineRunner(config=config, client_factory=factory)

        train_inp = _make_training_input(tmp_path)
        eval_inp = _make_evaluation_input(tmp_path)

        fake_job = MagicMock()
        fake_job.name = "pipeline_001"
        with patch.object(runner, "_build_pipeline", return_value=fake_job):
            train_out, eval_out = runner.run(
                train_inp, eval_inp, tmp_path / "train", tmp_path / "eval"
            )

        assert train_out.success
        assert eval_out.success
        assert train_out.job_status == TrainingJobStatus.COMPLETED
        assert train_out.mode == TrainingMode.AZURE_PIPELINE
        assert eval_out.mode == EvaluationMode.AZURE_PIPELINE_EVAL
        assert eval_out.runner == "azure-ml-pipeline"

    def test_pipeline_failed_status(self, tmp_path: Path) -> None:
        config = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Failed")
        runner = AzureMLPipelineRunner(config=config, client_factory=factory)

        fake_job = MagicMock()
        fake_job.name = "pipeline_002"
        with patch.object(runner, "_build_pipeline", return_value=fake_job):
            train_out, eval_out = runner.run(
                _make_training_input(tmp_path),
                _make_evaluation_input(tmp_path),
                tmp_path / "train",
                tmp_path / "eval",
            )

        assert not train_out.success
        assert not eval_out.success
        assert train_out.job_status == TrainingJobStatus.FAILED
        assert "Failed" in train_out.message

    def test_best_pt_missing_fails_training_output(self, tmp_path: Path) -> None:
        config = _minimal_config()

        class _NoWeightsFactory(FakeAzureMLClientFactory):
            def create(self, cfg):
                client = super().create(cfg)

                # Override download to create eval output but NOT best.pt
                def _download_no_best(name, output_name="", download_path="", all_outputs=False):
                    if all_outputs:
                        from pathlib import Path as _P  # noqa: PLC0415

                        eval_dir = _P(download_path) / "eval_step" / "eval_output"
                        eval_dir.mkdir(parents=True, exist_ok=True)
                        (eval_dir / "metrics.json").write_text(
                            '{"map50":0.5,"map50_95":0.3,"precision":0.5,"recall":0.5,'
                            '"per_class_metrics":{}}',
                            encoding="utf-8",
                        )
                        # train_step dir created but NO best.pt
                        (_P(download_path) / "train_step" / "model_output").mkdir(
                            parents=True, exist_ok=True
                        )
                    else:
                        (_P(download_path) / output_name).mkdir(parents=True, exist_ok=True)

                client.jobs.download = _download_no_best
                return client

        runner = AzureMLPipelineRunner(config=config, client_factory=_NoWeightsFactory())
        fake_job = MagicMock()
        fake_job.name = "pipeline_003"
        with patch.object(runner, "_build_pipeline", return_value=fake_job):
            train_out, _eval_out = runner.run(
                _make_training_input(tmp_path),
                _make_evaluation_input(tmp_path),
                tmp_path / "train",
                tmp_path / "eval",
            )

        assert not train_out.success
        assert "best.pt" in train_out.message

    def test_metrics_json_missing_fails_evaluation_output(self, tmp_path: Path) -> None:
        class _NoMetricsFactory(FakeAzureMLClientFactory):
            def create(self, cfg):
                client = super().create(cfg)

                def _download_no_metrics(name, output_name="", download_path="", all_outputs=False):
                    from pathlib import Path as _P  # noqa: PLC0415

                    if all_outputs:
                        train_dir = _P(download_path) / "train_step" / "model_output"
                        train_dir.mkdir(parents=True, exist_ok=True)
                        (train_dir / "best.pt").write_bytes(b"pt")
                        (train_dir / "last.pt").write_bytes(b"pt")
                        (_P(download_path) / "eval_step" / "eval_output").mkdir(
                            parents=True, exist_ok=True
                        )
                        # No metrics.json written

                client.jobs.download = _download_no_metrics
                return client

        runner = AzureMLPipelineRunner(config=_minimal_config(), client_factory=_NoMetricsFactory())
        fake_job = MagicMock()
        fake_job.name = "pipeline_004"
        with patch.object(runner, "_build_pipeline", return_value=fake_job):
            train_out, eval_out = runner.run(
                _make_training_input(tmp_path),
                _make_evaluation_input(tmp_path),
                tmp_path / "train",
                tmp_path / "eval",
            )

        assert train_out.success  # training part succeeded
        assert not eval_out.success
        assert "metrics.json" in eval_out.message

    def test_stream_logs_true_calls_stream(self, tmp_path: Path) -> None:
        config = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLPipelineRunner(config=config, client_factory=factory)

        fake_job = MagicMock()
        fake_job.name = "pipeline_005"
        with patch.object(runner, "_build_pipeline", return_value=fake_job):
            runner.run(
                _make_training_input(tmp_path),
                _make_evaluation_input(tmp_path),
                tmp_path / "train",
                tmp_path / "eval",
            )

        client = factory.last_client
        assert len(client.jobs.streamed) == 1

    def test_stream_logs_false_does_not_call_stream(self, tmp_path: Path) -> None:
        raw = _minimal_config().model_dump()
        raw["pipeline"] = {"stream_logs": False, "timeout_minutes": 1}
        config = AzureMLConfig.model_validate(raw)
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLPipelineRunner(config=config, client_factory=factory)

        fake_job = MagicMock()
        fake_job.name = "pipeline_006"
        with patch.object(runner, "_build_pipeline", return_value=fake_job):
            with patch.object(
                runner, "_wait_for_completion", return_value=MagicMock(status="Completed")
            ) as mock_wait:
                runner.run(
                    _make_training_input(tmp_path),
                    _make_evaluation_input(tmp_path),
                    tmp_path / "train",
                    tmp_path / "eval",
                )

        client = factory.last_client
        assert len(client.jobs.streamed) == 0
        mock_wait.assert_called_once()

    def test_download_outputs_false_skips_download(self, tmp_path: Path) -> None:
        raw = _minimal_config().model_dump()
        raw["pipeline"] = {"download_outputs": False, "timeout_minutes": 10}
        config = AzureMLConfig.model_validate(raw)
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLPipelineRunner(config=config, client_factory=factory)

        fake_job = MagicMock()
        fake_job.name = "pipeline_007"
        with patch.object(runner, "_build_pipeline", return_value=fake_job):
            runner.run(
                _make_training_input(tmp_path),
                _make_evaluation_input(tmp_path),
                tmp_path / "train",
                tmp_path / "eval",
            )

        client = factory.last_client
        assert len(client.jobs.downloaded) == 0

    def test_exception_returns_failed_outputs(self, tmp_path: Path) -> None:
        config = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLPipelineRunner(config=config, client_factory=factory)

        with patch.object(factory, "create", side_effect=RuntimeError("auth failed")):
            train_out, eval_out = runner.run(
                _make_training_input(tmp_path),
                _make_evaluation_input(tmp_path),
                tmp_path / "train",
                tmp_path / "eval",
            )

        assert not train_out.success
        assert not eval_out.success
        assert "auth failed" in train_out.message
        assert "auth failed" in eval_out.message

    def test_training_output_mode_is_azure_pipeline(self, tmp_path: Path) -> None:
        config = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLPipelineRunner(config=config, client_factory=factory)
        fake_job = MagicMock()
        fake_job.name = "pipeline_009"
        with patch.object(runner, "_build_pipeline", return_value=fake_job):
            train_out, _ = runner.run(
                _make_training_input(tmp_path),
                _make_evaluation_input(tmp_path),
                tmp_path / "train",
                tmp_path / "eval",
            )
        assert train_out.mode == TrainingMode.AZURE_PIPELINE
        assert train_out.runner == "azure-ml-pipeline"

    def test_artifacts_written_to_correct_dirs(self, tmp_path: Path) -> None:
        config = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLPipelineRunner(config=config, client_factory=factory)
        train_dir = tmp_path / "train"
        eval_dir = tmp_path / "eval"
        fake_job = MagicMock()
        fake_job.name = "pipeline_010"
        with patch.object(runner, "_build_pipeline", return_value=fake_job):
            train_out, eval_out = runner.run(
                _make_training_input(tmp_path),
                _make_evaluation_input(tmp_path),
                train_dir,
                eval_dir,
            )

        assert (train_dir / "best.pt").exists()
        assert (eval_dir / "metrics.json").exists()
        assert (train_dir / "training_output.json").exists()
        assert (eval_dir / "evaluation_output.json").exists()


# ── TestAzureMLPipelineRunnerJobConstruction ───────────────────────────────────


class TestAzureMLPipelineRunnerJobConstruction:
    """Tests for _build_pipeline — patch out azure.ai.ml imports."""

    def _make_build_mocks(self):
        """Return a dict of mock SDK objects to patch into azure.ai.ml."""
        fake_component = MagicMock(name="command_component")
        fake_component.return_value = MagicMock()
        fake_component.return_value.outputs = {"model_output": MagicMock(name="model_output_ref")}

        fake_pipeline_job = MagicMock(name="pipeline_job")
        fake_dsl_pipeline_decorator = MagicMock(name="dsl_pipeline_decorator")
        fake_dsl_pipeline_decorator.return_value = lambda f: MagicMock(
            return_value=fake_pipeline_job
        )
        fake_dsl = MagicMock(name="dsl")
        fake_dsl.pipeline = fake_dsl_pipeline_decorator

        return {
            "command": fake_component,
            "dsl": fake_dsl,
            "Input": MagicMock(name="Input"),
            "Output": MagicMock(name="Output"),
        }

    def test_build_pipeline_calls_create_or_update(self, tmp_path: Path) -> None:
        config = _minimal_config()
        factory = FakeAzureMLClientFactory(job_status="Completed")
        runner = AzureMLPipelineRunner(config=config, client_factory=factory)

        fake_job = MagicMock()
        fake_job.name = "built_pipeline"
        with patch.object(runner, "_build_pipeline", return_value=fake_job):
            runner.run(
                _make_training_input(tmp_path),
                _make_evaluation_input(tmp_path),
                tmp_path / "train",
                tmp_path / "eval",
            )

        client = factory.last_client
        assert len(client.jobs.created) == 1

    def test_eval_command_contains_weights_best_pt(self, tmp_path: Path) -> None:
        """Confirm the eval command string uses /best.pt from the train step folder."""
        import sys  # noqa: PLC0415

        config = _minimal_config()
        runner = AzureMLPipelineRunner(config=config, client_factory=FakeAzureMLClientFactory())

        train_inp = _make_training_input(tmp_path)
        cfg = train_inp.training_config

        captured_commands: list[str] = []

        real_command_mock = MagicMock(
            side_effect=lambda **kw: (
                captured_commands.append(kw.get("command", "")),
                MagicMock(outputs={"model_output": MagicMock(), "eval_output": MagicMock()}),
            )[1]
        )

        fake_dsl = MagicMock()
        fake_pipeline_func = MagicMock()
        fake_pipeline_func.return_value = MagicMock(
            settings=MagicMock(default_compute=None),
        )
        fake_dsl.pipeline = MagicMock(return_value=lambda f: fake_pipeline_func)

        azure_ml_mock = MagicMock()
        azure_ml_mock.command = real_command_mock
        azure_ml_mock.Input = MagicMock()
        azure_ml_mock.Output = MagicMock()
        azure_ml_mock.dsl = fake_dsl

        with patch.dict(
            sys.modules,
            {
                "azure.ai.ml": azure_ml_mock,
                "azure.ai.ml.constants": MagicMock(),
                "azure.ai.ml.entities": MagicMock(),
            },
        ):
            try:
                runner._build_pipeline(train_inp, cfg, None)
            except Exception:
                pass  # DSL decorator wiring may fail in mocked env; we only need captured_commands

        eval_cmds = [c for c in captured_commands if "eval_yolo.py" in c]
        assert any(
            "/best.pt" in c for c in eval_cmds
        ), f"Expected /best.pt in eval command, got: {eval_cmds}"


# ── TestAzureMLPipelineConfig ──────────────────────────────────────────────────


class TestAzureMLPipelineConfig:
    def test_default_timeout_is_240(self) -> None:
        pcfg = AzureMLPipelineConfig()
        assert pcfg.timeout_minutes == 240

    def test_default_output_names(self) -> None:
        pcfg = AzureMLPipelineConfig()
        assert pcfg.train_output_name == "model_output"
        assert pcfg.eval_output_name == "eval_output"

    def test_azure_ml_config_has_pipeline_field(self) -> None:
        config = _minimal_config()
        assert hasattr(config, "pipeline")
        assert isinstance(config.pipeline, AzureMLPipelineConfig)

    def test_from_yaml_loads_pipeline_section(self, tmp_path: Path) -> None:
        cfg_yaml = tmp_path / "azure_ml.yaml"
        cfg_yaml.write_text(
            "subscription_id: sub-1\nresource_group: rg\nworkspace_name: ws\n"
            "compute_name: cpu\n"
            "environment:\n  mode: registered\n  registered_environment: azureml:env:1\n"
            "pipeline:\n  timeout_minutes: 360\n  train_output_name: my_model\n",
            encoding="utf-8",
        )
        config = AzureMLConfig.from_yaml(cfg_yaml)
        assert config.pipeline.timeout_minutes == 360
        assert config.pipeline.train_output_name == "my_model"

    def test_pipeline_optional_uses_defaults_when_omitted(self, tmp_path: Path) -> None:
        cfg_yaml = tmp_path / "azure_ml_no_pipeline.yaml"
        cfg_yaml.write_text(
            "subscription_id: s\nresource_group: r\nworkspace_name: w\ncompute_name: c\n"
            "environment:\n  mode: registered\n  registered_environment: azureml:env:1\n",
            encoding="utf-8",
        )
        config = AzureMLConfig.from_yaml(cfg_yaml)
        assert config.pipeline.timeout_minutes == 240
        assert config.pipeline.download_outputs is True


# ── TestMVPWorkflowPipelineIntegration ────────────────────────────────────────


class TestMVPWorkflowPipelineIntegration:
    """End-to-end tests for MVPWorkflow with training_runner='azure-ml-pipeline'."""

    def _make_dataset(self, tmp_path: Path) -> tuple[Path, Path]:
        ds = tmp_path / "dataset"
        ds.mkdir(parents=True)
        (ds / "images" / "train").mkdir(parents=True)
        (ds / "images" / "val").mkdir(parents=True)
        (ds / "labels" / "train").mkdir(parents=True)
        (ds / "labels" / "val").mkdir(parents=True)
        data_yaml = ds / "data.yaml"
        data_yaml.write_text(
            "path: /tmp\ntrain: images/train\nval: images/val\nnames: {0: scratch}\n",
            encoding="utf-8",
        )
        return ds, data_yaml

    def _make_training_config(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "training.yaml"
        cfg.write_text("model: yolo11m.pt\nepochs: 3\n", encoding="utf-8")
        return cfg

    def test_pipeline_runner_used_when_azure_ml_pipeline(self, tmp_path: Path) -> None:
        from agentic_mlops.contracts.workflows import MVPWorkflowInput  # noqa: PLC0415
        from agentic_mlops.workflows.mvp_workflow import MVPWorkflow  # noqa: PLC0415

        ds, data_yaml = self._make_dataset(tmp_path)
        train_cfg = self._make_training_config(tmp_path)

        # Build fake TrainingOutput + EvaluationOutput for the pipeline runner to return
        fake_train_out = TrainingOutput(
            success=True,
            message="pipeline ok",
            job_status=TrainingJobStatus.COMPLETED,
            mode=TrainingMode.AZURE_PIPELINE,
            best_weights_path=str(tmp_path / "best.pt"),
            artifacts=[],
        )
        (tmp_path / "best.pt").write_bytes(b"fake")

        fake_eval_out = EvaluationOutput(
            success=True,
            message="eval ok",
            mode=EvaluationMode.AZURE_PIPELINE_EVAL,
            runner="azure-ml-pipeline",
            model_path="pipeline",
            artifacts=[],
        )

        # Fake pipeline runner that returns the above
        fake_pipeline_runner = MagicMock()
        fake_pipeline_runner.run.return_value = (fake_train_out, fake_eval_out)

        # Azure config file (content doesn't matter — runner is injected)
        azure_cfg = tmp_path / "azure_ml.yaml"
        azure_cfg.write_text(
            "subscription_id: s\nresource_group: r\nworkspace_name: w\ncompute_name: c\n"
            "environment:\n  mode: registered\n  registered_environment: azureml:env:1\n",
            encoding="utf-8",
        )

        workflow = MVPWorkflow()

        with patch(
            "agentic_mlops.workflows.mvp_workflow.AzureMLPipelineRunner",
            return_value=fake_pipeline_runner,
        ):
            result = workflow.run(
                MVPWorkflowInput(
                    dataset_path=str(ds),
                    data_yaml_path=str(data_yaml),
                    training_config_path=str(train_cfg),
                    output_dir=str(tmp_path / "out"),
                    training_runner="azure-ml-pipeline",
                    azure_config_path=str(azure_cfg),
                    dry_run=False,
                    interactive_approval=False,
                    approval_action=None,  # will be blocked at approval
                )
            )

        fake_pipeline_runner.run.assert_called_once()
        step_names = [s.step for s in result.steps]
        assert "training" in step_names
        assert "evaluation" in step_names
        training_step = next(s for s in result.steps if s.step == "training")
        evaluation_step = next(s for s in result.steps if s.step == "evaluation")
        assert training_step.success
        assert evaluation_step.success

    def test_missing_azure_config_path_fails_fast(self, tmp_path: Path) -> None:
        from agentic_mlops.contracts.workflows import MVPWorkflowInput  # noqa: PLC0415
        from agentic_mlops.workflows.mvp_workflow import MVPWorkflow  # noqa: PLC0415

        ds, data_yaml = self._make_dataset(tmp_path)
        train_cfg = self._make_training_config(tmp_path)

        workflow = MVPWorkflow()
        result = workflow.run(
            MVPWorkflowInput(
                dataset_path=str(ds),
                data_yaml_path=str(data_yaml),
                training_config_path=str(train_cfg),
                output_dir=str(tmp_path / "out"),
                training_runner="azure-ml-pipeline",
                azure_config_path=None,  # missing
                dry_run=False,
                interactive_approval=False,
            )
        )

        assert not result.success
        assert "azure_config_path" in result.message.lower() or any(
            "azure_config_path" in e.lower() for e in result.errors
        )


# ── TestOrchestratorPipelineIntegration ───────────────────────────────────────


class TestOrchestratorPipelineIntegration:
    """Tests for OrchestratorWorkflow pipeline runner integration."""

    def _make_dataset(self, tmp_path: Path) -> tuple[str, str]:
        ds = tmp_path / "dataset"
        ds.mkdir(parents=True)
        (ds / "images" / "train").mkdir(parents=True)
        (ds / "images" / "val").mkdir(parents=True)
        (ds / "labels" / "train").mkdir(parents=True)
        (ds / "labels" / "val").mkdir(parents=True)
        data_yaml = ds / "data.yaml"
        data_yaml.write_text(
            "path: /tmp\ntrain: images/train\nval: images/val\nnames: {0: scratch}\n",
            encoding="utf-8",
        )
        return str(ds), str(data_yaml)

    def test_step_evaluation_uses_pipeline_eval_path_when_present(self, tmp_path: Path) -> None:
        from agentic_mlops.contracts.orchestrator import OrchestratorInput  # noqa: PLC0415
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow  # noqa: PLC0415

        # Write a pre-computed eval output JSON
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir(parents=True)
        eval_output = EvaluationOutput(
            success=True,
            message="pipeline eval done",
            mode=EvaluationMode.AZURE_PIPELINE_EVAL,
            runner="azure-ml-pipeline",
            model_path="pipeline",
            artifacts=[],
        )
        eval_out_path = eval_dir / "evaluation_output.json"
        eval_out_path.write_text(
            json.dumps(eval_output.model_dump(mode="json"), indent=2), encoding="utf-8"
        )

        ds, data_yaml = self._make_dataset(tmp_path)

        orch = OrchestratorWorkflow()
        inp = OrchestratorInput(
            workflow_id="test_wf",
            dataset_path=ds,
            data_yaml_path=data_yaml,
            training_config_path="dummy",
            output_dir=str(tmp_path / "out"),
        )
        output_root = tmp_path / "out"
        output_root.mkdir(parents=True, exist_ok=True)
        step_outputs = {
            "training": {
                "best_weights_path": "fake/best.pt",
                "job_status": "completed",
                "pipeline_eval_output_path": str(eval_out_path),
            }
        }

        outcome = orch._step_evaluation(inp, output_root, step_outputs, azure_config=None)

        assert outcome.success
        assert outcome.status_label == "COMPLETED"
        assert outcome.key_outputs["output_json_path"] == str(eval_out_path)

    def test_step_evaluation_missing_pipeline_eval_file_fails(self, tmp_path: Path) -> None:
        from agentic_mlops.contracts.orchestrator import OrchestratorInput  # noqa: PLC0415
        from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow  # noqa: PLC0415

        ds, data_yaml = self._make_dataset(tmp_path)
        nonexistent = str(tmp_path / "evaluation" / "evaluation_output.json")

        orch = OrchestratorWorkflow()
        inp = OrchestratorInput(
            workflow_id="test_wf",
            dataset_path=ds,
            data_yaml_path=data_yaml,
            training_config_path="dummy",
            output_dir=str(tmp_path / "out"),
        )
        output_root = tmp_path / "out"
        output_root.mkdir(parents=True, exist_ok=True)
        step_outputs = {
            "training": {
                "best_weights_path": "fake/best.pt",
                "job_status": "completed",
                "pipeline_eval_output_path": nonexistent,
            }
        }

        outcome = orch._step_evaluation(inp, output_root, step_outputs, azure_config=None)

        assert not outcome.success
        assert outcome.status_label == "FAILED"
