"""Dataset Versioning Agent — registers a clean dataset as a new version + lineage.

Runs after Dataset Validation (and, ideally, Label QA). Reads their reports from
disk when given and blocks registration if either says the dataset/labels failed
— "registers clean dataset" is the whole point (spec title). Delegates the actual
hashing/copying/lineage write to a DatasetVersionRegistryClientBase.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agentic_mlops.agents.base import BaseAgent
from agentic_mlops.contracts.dataset_versioning import (
    DatasetVersioningInput,
    DatasetVersioningOutput,
    DatasetVersionStatus,
)
from agentic_mlops.integrations.dataset_registry import (
    DatasetVersionRegistryClientBase,
    create_dataset_registry_client,
)
from agentic_mlops.integrations.mlflow_client import MLflowTrackingClientBase
from agentic_mlops.tools.report_writer import ReportWriter


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_classes(data_yaml_path: Path) -> list[str]:
    with open(data_yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def _blocked(msg: str, dataset_name: str = "") -> DatasetVersioningOutput:
    return DatasetVersioningOutput(
        success=False,
        message=msg,
        status=DatasetVersionStatus.BLOCKED,
        dataset_name=dataset_name,
        block_reason=msg,
        errors=[msg],
    )


def _failed(msg: str, dataset_name: str = "") -> DatasetVersioningOutput:
    return DatasetVersioningOutput(
        success=False,
        message=msg,
        status=DatasetVersionStatus.FAILED,
        dataset_name=dataset_name,
        errors=[msg],
    )


class DatasetVersioningAgent(BaseAgent):
    """Registers a structured YOLO dataset as a new version with lineage.

    Output artifacts (in artifacts_dir):
        dataset_version_report.json / .md
    Registry artifacts (in registry_dir/<dataset_name>/versions/<N>/):
        dataset/ (full copy), lineage.json, dataset_version_output.json
    """

    def __init__(
        self,
        artifacts_dir: Path,
        registry_client: DatasetVersionRegistryClientBase | None = None,
        mlflow_client: MLflowTrackingClientBase | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__(artifacts_dir)
        self._registry_client = registry_client
        self._report_writer = ReportWriter()
        self._mlflow = mlflow_client
        self._mlflow_run_id = mlflow_run_id

    def run(self, input: DatasetVersioningInput) -> DatasetVersioningOutput:
        self._log_start(dataset_path=input.dataset_path, dataset_name=input.dataset_name)

        dataset_path = Path(input.dataset_path).resolve()
        if not dataset_path.is_dir():
            return _failed(
                f"dataset_path not found or not a directory: {dataset_path}", input.dataset_name
            )

        data_yaml_path = dataset_path / "data.yaml"
        if not data_yaml_path.exists():
            return _failed(
                f"data.yaml not found under {dataset_path} — not a structured YOLO dataset.",
                input.dataset_name,
            )

        validation_status: str | None = None
        if input.validation_report_path:
            try:
                vdata = _load_json(input.validation_report_path)
            except (OSError, json.JSONDecodeError) as exc:
                return _failed(f"Cannot read validation_report_path: {exc}", input.dataset_name)
            validation_status = vdata.get("status")
            if validation_status == "failed":
                return _blocked(
                    "Dataset validation status is 'failed' — cannot version an invalid dataset.",
                    input.dataset_name,
                )

        label_qa_status: str | None = None
        if input.label_quality_report_path:
            try:
                ldata = _load_json(input.label_quality_report_path)
            except (OSError, json.JSONDecodeError) as exc:
                return _failed(
                    f"Cannot read label_quality_report_path: {exc}", input.dataset_name
                )
            label_qa_status = ldata.get("status")
            if label_qa_status == "failed":
                return _blocked(
                    "Label QA status is 'failed' — cannot version a dataset with "
                    "unresolved label issues.",
                    input.dataset_name,
                )

        classes = _extract_classes(data_yaml_path)

        try:
            client = self._registry_client or create_dataset_registry_client(input.backend)
        except ValueError as exc:
            return _failed(str(exc), input.dataset_name)

        output = client.register(
            input, classes, validation_status, label_qa_status, self.artifacts_dir
        )

        json_path, md_path = self._report_writer.write_dataset_versioning_report(
            output, self.artifacts_dir
        )
        for p in (str(json_path), str(md_path)):
            if p not in output.artifacts:
                output.artifacts.append(p)

        if self._mlflow and self._mlflow_run_id and output.success:
            self._log_to_mlflow(input, output)

        self._log_done(
            status=str(output.status),
            dataset_name=input.dataset_name,
            version=str(output.version),
        )
        return output

    def _log_to_mlflow(
        self, inp: DatasetVersioningInput, output: DatasetVersioningOutput
    ) -> None:
        rid = self._mlflow_run_id
        client = self._mlflow
        assert rid is not None and client is not None

        params: dict[str, str] = {
            "dataset_versioning.dataset_name": inp.dataset_name,
            "dataset_versioning.version": str(output.version),
        }
        if output.hash:
            params["dataset_versioning.hash"] = output.hash[:16]
        client.log_params(rid, params)

        client.log_tags(rid, {
            "workflow_step": "dataset_versioning",
            "dataset_versioning_status": str(output.status),
            "dataset_name": inp.dataset_name,
            "dataset_version": str(output.version),
        })

        for artifact_path in output.artifacts:
            client.log_artifact(rid, artifact_path)
