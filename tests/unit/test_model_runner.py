"""Unit tests for tools/model_runner.py — ModelRunner protocol and YoloModelRunner."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_mlops.contracts.annotation import AnnotationInput
from agentic_mlops.contracts.common import ModelFramework
from agentic_mlops.contracts.evaluation import EvaluationInput, EvaluationMode
from agentic_mlops.contracts.training import TrainingInput, TrainingMode
from agentic_mlops.tools.model_runner import (
    ModelRunner,
    OnnxOnlyModelRunner,
    TorchvisionModelRunner,
    YoloModelRunner,
    create_model_runner,
)

# ---------------------------------------------------------------------------
# ModelFramework enum
# ---------------------------------------------------------------------------


class TestModelFramework:
    def test_yolo_value(self):
        assert ModelFramework.YOLO == "yolo"

    def test_torchvision_value(self):
        assert ModelFramework.TORCHVISION == "torchvision"

    def test_onnx_only_value(self):
        assert ModelFramework.ONNX_ONLY == "onnx_only"

    def test_all_values(self):
        values = {f.value for f in ModelFramework}
        assert values == {"yolo", "torchvision", "onnx_only"}


# ---------------------------------------------------------------------------
# framework field defaults on contracts
# ---------------------------------------------------------------------------


class TestFrameworkFieldDefaults:
    def _training_config(self):
        from agentic_mlops.contracts.training import TrainingConfig  # noqa: PLC0415

        return TrainingConfig(mode=TrainingMode.LOCAL_DRY_RUN)

    def test_training_input_default_framework(self):
        inp = TrainingInput(
            dataset_path="/d",
            data_yaml_path="/d/data.yaml",
            training_config=self._training_config(),
        )
        assert inp.framework == ModelFramework.YOLO

    def test_training_input_custom_framework(self):
        inp = TrainingInput(
            dataset_path="/d",
            data_yaml_path="/d/data.yaml",
            training_config=self._training_config(),
            framework=ModelFramework.TORCHVISION,
        )
        assert inp.framework == ModelFramework.TORCHVISION

    def test_evaluation_input_default_framework(self):
        inp = EvaluationInput(
            dataset_path="/d",
            data_yaml_path="/d/data.yaml",
            weights_path="/m/best.pt",
        )
        assert inp.framework == ModelFramework.YOLO

    def test_evaluation_input_custom_framework(self):
        inp = EvaluationInput(
            dataset_path="/d",
            data_yaml_path="/d/data.yaml",
            weights_path="/m/best.pt",
            framework=ModelFramework.ONNX_ONLY,
        )
        assert inp.framework == ModelFramework.ONNX_ONLY

    def test_annotation_input_default_framework(self):
        inp = AnnotationInput(images_path="/img", model_path="/m/best.pt")
        assert inp.framework == ModelFramework.YOLO

    def test_annotation_input_custom_framework(self):
        inp = AnnotationInput(
            images_path="/img",
            model_path="/m/best.pt",
            framework=ModelFramework.TORCHVISION,
        )
        assert inp.framework == ModelFramework.TORCHVISION

    def test_framework_serialises_as_string(self):
        inp = AnnotationInput(images_path="/img", model_path="/m/best.pt")
        data = inp.model_dump()
        assert data["framework"] == "yolo"

    def test_framework_accepts_string_value(self):
        inp = AnnotationInput(images_path="/img", model_path="/m/best.pt", framework="yolo")
        assert inp.framework == ModelFramework.YOLO


# ---------------------------------------------------------------------------
# ModelRunner protocol structural check
# ---------------------------------------------------------------------------


class TestModelRunnerProtocol:
    def test_yolo_runner_satisfies_protocol(self):
        runner = YoloModelRunner()
        assert isinstance(runner, ModelRunner)

    def test_custom_class_satisfies_protocol(self):
        class FakeRunner:
            def train(self, inp, artifacts_dir):
                ...

            def evaluate(self, inp, artifacts_dir):
                ...

            def predict(self, inp, artifacts_dir):
                ...

        assert isinstance(FakeRunner(), ModelRunner)

    def test_incomplete_class_does_not_satisfy_protocol(self):
        class IncompleteRunner:
            def train(self, inp, artifacts_dir):
                ...

        assert not isinstance(IncompleteRunner(), ModelRunner)


# ---------------------------------------------------------------------------
# YoloModelRunner — delegate smoke test (dry-run mode, no ultralytics required)
# ---------------------------------------------------------------------------


class TestYoloModelRunnerDryRun:
    def _training_input(self, tmp_path: Path) -> TrainingInput:
        from agentic_mlops.contracts.training import TrainingConfig  # noqa: PLC0415

        cfg = TrainingConfig(mode=TrainingMode.LOCAL_DRY_RUN)
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "data.yaml").write_text("nc: 1\nnames: [crack]\n")
        return TrainingInput(
            dataset_path=str(dataset),
            data_yaml_path=str(dataset / "data.yaml"),
            training_config=cfg,
        )

    def test_train_dry_run_returns_success(self, tmp_path: Path):
        runner = YoloModelRunner()
        artifacts = tmp_path / "artifacts"
        inp = self._training_input(tmp_path)
        out = runner.train(inp, artifacts)
        assert out.success

    def test_evaluate_dry_run_returns_success(self, tmp_path: Path):
        runner = YoloModelRunner()
        artifacts = tmp_path / "artifacts"
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\nnames: [crack]\ntrain: train\nval: val\n")
        inp = EvaluationInput(
            dataset_path=str(tmp_path),
            data_yaml_path=str(data_yaml),
            weights_path=str(tmp_path / "best.pt"),
            mode=EvaluationMode.LOCAL_DRY_RUN,
        )
        out = runner.evaluate(inp, artifacts)
        assert out.success

    def test_predict_missing_dir_returns_failure(self, tmp_path: Path):
        runner = YoloModelRunner()
        artifacts = tmp_path / "artifacts"
        inp = AnnotationInput(
            images_path=str(tmp_path / "nonexistent"),
            model_path=str(tmp_path / "best.pt"),
        )
        out = runner.predict(inp, artifacts)
        assert not out.success


# ---------------------------------------------------------------------------
# create_model_runner factory
# ---------------------------------------------------------------------------


class TestCreateModelRunnerFactory:
    def test_yolo_framework_returns_yolo_runner(self):
        runner = create_model_runner("yolo")
        assert isinstance(runner, YoloModelRunner)

    def test_default_framework_returns_yolo_runner(self):
        runner = create_model_runner()
        assert isinstance(runner, YoloModelRunner)

    def test_unknown_framework_raises(self):
        with pytest.raises(ValueError, match="unknown_fw"):
            create_model_runner("unknown_fw")

    def test_onnx_only_returns_onnx_runner(self):
        runner = create_model_runner("onnx_only")
        assert isinstance(runner, OnnxOnlyModelRunner)

    def test_torchvision_returns_torchvision_runner(self):
        runner = create_model_runner("torchvision")
        assert isinstance(runner, TorchvisionModelRunner)


# ---------------------------------------------------------------------------
# OnnxOnlyModelRunner
# ---------------------------------------------------------------------------


class TestOnnxOnlyModelRunner:
    def test_satisfies_model_runner_protocol(self):
        assert isinstance(OnnxOnlyModelRunner(), ModelRunner)

    def test_train_raises_not_implemented(self):
        runner = OnnxOnlyModelRunner()
        with pytest.raises(NotImplementedError, match="pre-trained"):
            runner.train(None, Path("/tmp"))  # type: ignore[arg-type]

    def test_evaluate_raises_not_implemented(self):
        runner = OnnxOnlyModelRunner()
        with pytest.raises(NotImplementedError, match="OnnxOnlyModelRunner"):
            runner.evaluate(None, Path("/tmp"))  # type: ignore[arg-type]

    def test_predict_without_onnxruntime_returns_failure(self, monkeypatch, tmp_path: Path):
        import sys  # noqa: PLC0415

        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        runner = OnnxOnlyModelRunner()
        inp = AnnotationInput(images_path=str(tmp_path), model_path=str(tmp_path / "m.onnx"))
        out = runner.predict(inp, tmp_path / "artifacts")
        assert not out.success
        assert "onnxruntime" in out.message

    def test_predict_missing_images_dir_returns_failure(self, monkeypatch, tmp_path: Path):
        # Simulate onnxruntime and PIL installed so we reach the dir-check code path.
        import types  # noqa: PLC0415

        fake_ort = types.ModuleType("onnxruntime")
        fake_pil_image = types.ModuleType("PIL.Image")
        fake_pil = types.ModuleType("PIL")
        monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
        monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)
        monkeypatch.setitem(__import__("sys").modules, "PIL.Image", fake_pil_image)

        runner = OnnxOnlyModelRunner()
        inp = AnnotationInput(
            images_path=str(tmp_path / "nonexistent"),
            model_path=str(tmp_path / "m.onnx"),
        )
        out = runner.predict(inp, tmp_path / "artifacts")
        assert not out.success
        assert "not found" in out.message.lower() or "nonexistent" in out.message


# ---------------------------------------------------------------------------
# TorchvisionModelRunner
# ---------------------------------------------------------------------------


class TestTorchvisionModelRunner:
    def test_satisfies_model_runner_protocol(self):
        assert isinstance(TorchvisionModelRunner(), ModelRunner)

    def test_all_methods_raise_not_implemented(self, tmp_path: Path):
        runner = TorchvisionModelRunner()
        for method_name in ("train", "evaluate", "predict"):
            with pytest.raises(NotImplementedError, match="TorchvisionModelRunner"):
                getattr(runner, method_name)(None, tmp_path)  # type: ignore[arg-type]
