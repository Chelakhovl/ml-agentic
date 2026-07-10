"""Unit tests for the Dataset Versioning Agent, local registry, and CLI command.

Coverage matrix:
    1.  dataset_path not found -> failed
    2.  No data.yaml -> failed
    3.  validation_report status='failed' -> blocked
    4.  label_quality_report status='failed' -> blocked
    5.  validation_report status='passed' -> proceeds normally
    6.  First registration creates version 1
    7.  Unchanged content on second registration -> deduplicated, version unchanged
    8.  Changed content -> new version created
    9.  Dataset files are actually copied into the registry
   10.  lineage.json contains classes/hash/approved_by/workflow_id/source_batches/parent_version
   11.  latest.json updated only after a successful new registration
   12.  FakeDatasetVersionRegistry records calls and increments version
   13.  DatasetVersioningAgent writes dataset_version_report.json / .md
   14.  DatasetVersioningAgent logs to MLflow only on success
   15.  CLI: version-dataset command exists and succeeds (exit 0)
   16.  CLI: version-dataset exits 1 on missing dataset_path
   17.  CLI: version-dataset exits 1 when validation report says failed
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from agentic_mlops.agents.dataset_versioning import DatasetVersioningAgent
from agentic_mlops.contracts.dataset_versioning import (
    DatasetVersioningInput,
    DatasetVersionStatus,
)
from agentic_mlops.integrations.dataset_registry import FakeDatasetVersionRegistry, hash_dataset
from agentic_mlops.integrations.mlflow_client import FakeMLflowClient

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_structured_dataset(root: Path, label_line: str = "0 0.5 0.5 0.2 0.2\n") -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(root / "images" / "train" / "a.jpg")
    (root / "labels" / "train" / "a.txt").write_text(label_line, encoding="utf-8")
    (root / "data.yaml").write_text("names:\n  0: scratch\n  1: dent\n", encoding="utf-8")


def _write_report(path: Path, status: str) -> str:
    path.write_text(json.dumps({"status": status}), encoding="utf-8")
    return str(path)


# ── 1-2. Structural failures ──────────────────────────────────────────────────


def test_dataset_path_not_found_fails(tmp_path: Path) -> None:
    agent = DatasetVersioningAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        DatasetVersioningInput(
            dataset_path=str(tmp_path / "nope"),
            dataset_name="ds",
            registry_dir=str(tmp_path / "registry"),
        )
    )
    assert result.success is False
    assert result.status == DatasetVersionStatus.FAILED


def test_missing_data_yaml_fails(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()

    agent = DatasetVersioningAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        DatasetVersioningInput(
            dataset_path=str(ds), dataset_name="ds", registry_dir=str(tmp_path / "registry")
        )
    )
    assert result.success is False
    assert result.status == DatasetVersionStatus.FAILED
    assert "data.yaml" in result.message


# ── 3-5. Gates ─────────────────────────────────────────────────────────────────


def test_blocked_when_validation_failed(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    report = _write_report(tmp_path / "validation.json", "failed")

    agent = DatasetVersioningAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        DatasetVersioningInput(
            dataset_path=str(ds),
            dataset_name="ds",
            registry_dir=str(tmp_path / "registry"),
            validation_report_path=report,
        )
    )
    assert result.success is False
    assert result.status == DatasetVersionStatus.BLOCKED
    assert "validation" in result.block_reason.lower()


def test_blocked_when_label_qa_failed(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    report = _write_report(tmp_path / "label_qa.json", "failed")

    agent = DatasetVersioningAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        DatasetVersioningInput(
            dataset_path=str(ds),
            dataset_name="ds",
            registry_dir=str(tmp_path / "registry"),
            label_quality_report_path=report,
        )
    )
    assert result.success is False
    assert result.status == DatasetVersionStatus.BLOCKED
    assert "label qa" in result.block_reason.lower()


def test_proceeds_when_validation_passed(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    report = _write_report(tmp_path / "validation.json", "passed")

    agent = DatasetVersioningAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        DatasetVersioningInput(
            dataset_path=str(ds),
            dataset_name="ds",
            registry_dir=str(tmp_path / "registry"),
            validation_report_path=report,
        )
    )
    assert result.success is True
    assert result.status == DatasetVersionStatus.REGISTERED
    assert result.lineage.validation_status == "passed"


# ── 6-11. LocalDatasetVersionRegistry ──────────────────────────────────────────


def test_first_registration_creates_version_1(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)

    agent = DatasetVersioningAgent(artifacts_dir=tmp_path / "artifacts")
    result = agent.run(
        DatasetVersioningInput(
            dataset_path=str(ds), dataset_name="ds", registry_dir=str(tmp_path / "registry")
        )
    )
    assert result.success is True
    assert result.status == DatasetVersionStatus.REGISTERED
    assert result.version == 1


def test_unchanged_content_deduplicates(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    registry_dir = tmp_path / "registry"

    r1 = DatasetVersioningAgent(artifacts_dir=tmp_path / "a1").run(
        DatasetVersioningInput(
            dataset_path=str(ds), dataset_name="ds", registry_dir=str(registry_dir)
        )
    )
    r2 = DatasetVersioningAgent(artifacts_dir=tmp_path / "a2").run(
        DatasetVersioningInput(
            dataset_path=str(ds), dataset_name="ds", registry_dir=str(registry_dir)
        )
    )

    assert r1.version == 1
    assert r2.status == DatasetVersionStatus.DEDUPLICATED
    assert r2.version == 1
    assert r1.hash == r2.hash


def test_changed_content_creates_new_version(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    registry_dir = tmp_path / "registry"

    r1 = DatasetVersioningAgent(artifacts_dir=tmp_path / "a1").run(
        DatasetVersioningInput(
            dataset_path=str(ds), dataset_name="ds", registry_dir=str(registry_dir)
        )
    )
    _make_structured_dataset(ds, label_line="0 0.3 0.3 0.1 0.1\n")
    r2 = DatasetVersioningAgent(artifacts_dir=tmp_path / "a2").run(
        DatasetVersioningInput(
            dataset_path=str(ds),
            dataset_name="ds",
            registry_dir=str(registry_dir),
            parent_version=1,
        )
    )

    assert r1.version == 1
    assert r2.status == DatasetVersionStatus.REGISTERED
    assert r2.version == 2
    assert r1.hash != r2.hash
    assert r2.lineage.parent_version == 1


def test_dataset_files_copied_into_registry(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    registry_dir = tmp_path / "registry"

    result = DatasetVersioningAgent(artifacts_dir=tmp_path / "artifacts").run(
        DatasetVersioningInput(
            dataset_path=str(ds), dataset_name="ds", registry_dir=str(registry_dir)
        )
    )

    copy_dir = Path(result.dataset_version_path)
    assert (copy_dir / "images" / "train" / "a.jpg").exists()
    assert (copy_dir / "labels" / "train" / "a.txt").exists()
    assert (copy_dir / "data.yaml").exists()


def test_lineage_fields_populated(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)

    result = DatasetVersioningAgent(artifacts_dir=tmp_path / "artifacts").run(
        DatasetVersioningInput(
            dataset_path=str(ds),
            dataset_name="ds",
            registry_dir=str(tmp_path / "registry"),
            workflow_id="wf_001",
            approved_by="ml_engineer",
            source_batches=["batch_2026_06_01"],
        )
    )

    lineage = result.lineage
    assert lineage.classes == ["scratch", "dent"]
    assert lineage.hash == result.hash
    assert lineage.workflow_id == "wf_001"
    assert lineage.approved_by == "ml_engineer"
    assert lineage.source_batches == ["batch_2026_06_01"]

    lineage_path = Path(result.dataset_version_artifact)
    assert lineage_path.exists()
    on_disk = json.loads(lineage_path.read_text(encoding="utf-8"))
    assert on_disk["hash"] == result.hash


def test_latest_json_updated_on_new_version(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    registry_dir = tmp_path / "registry"

    DatasetVersioningAgent(artifacts_dir=tmp_path / "artifacts").run(
        DatasetVersioningInput(
            dataset_path=str(ds), dataset_name="ds", registry_dir=str(registry_dir)
        )
    )

    latest = json.loads((registry_dir / "ds" / "latest.json").read_text(encoding="utf-8"))
    assert latest["version"] == 1
    assert latest["dataset_name"] == "ds"


def test_hash_dataset_is_deterministic(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    assert hash_dataset(ds) == hash_dataset(ds)


# ── 12. FakeDatasetVersionRegistry ────────────────────────────────────────────


def test_fake_registry_records_calls_and_increments_version(tmp_path: Path) -> None:
    client = FakeDatasetVersionRegistry()
    inp = DatasetVersioningInput(
        dataset_path="/a", dataset_name="ds", registry_dir=str(tmp_path)
    )
    r1 = client.register(inp, ["a"], None, None, tmp_path)
    r2 = client.register(inp, ["a"], None, None, tmp_path)

    assert len(client.calls) == 2
    assert r1.version == 1
    assert r2.version == 2


# ── 13-14. DatasetVersioningAgent ─────────────────────────────────────────────


def test_agent_writes_report_files(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    artifacts_dir = tmp_path / "artifacts"

    agent = DatasetVersioningAgent(artifacts_dir=artifacts_dir)
    result = agent.run(
        DatasetVersioningInput(
            dataset_path=str(ds), dataset_name="ds", registry_dir=str(tmp_path / "registry")
        )
    )

    assert (artifacts_dir / "dataset_version_report.json").exists()
    assert (artifacts_dir / "dataset_version_report.md").exists()

    data = json.loads(
        (artifacts_dir / "dataset_version_report.json").read_text(encoding="utf-8")
    )
    assert data["status"] == "registered"
    assert result.success is True


def test_agent_logs_to_mlflow_on_success(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    client = FakeMLflowClient()
    run_id = client.start_run("exp", "run")

    agent = DatasetVersioningAgent(
        artifacts_dir=tmp_path / "artifacts", mlflow_client=client, mlflow_run_id=run_id
    )
    agent.run(
        DatasetVersioningInput(
            dataset_path=str(ds), dataset_name="ds", registry_dir=str(tmp_path / "registry")
        )
    )

    tags = client.runs[run_id]["tags"]
    assert tags.get("workflow_step") == "dataset_versioning"
    assert len(client.runs[run_id]["artifacts"]) > 0


def test_agent_does_not_log_to_mlflow_when_blocked(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    report = _write_report(tmp_path / "validation.json", "failed")
    client = FakeMLflowClient()
    run_id = client.start_run("exp", "run")

    agent = DatasetVersioningAgent(
        artifacts_dir=tmp_path / "artifacts", mlflow_client=client, mlflow_run_id=run_id
    )
    agent.run(
        DatasetVersioningInput(
            dataset_path=str(ds),
            dataset_name="ds",
            registry_dir=str(tmp_path / "registry"),
            validation_report_path=report,
        )
    )

    assert client.runs[run_id]["tags"] == {}


# ── 15-17. CLI ─────────────────────────────────────────────────────────────────


def test_cli_version_dataset_succeeds(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    ds = tmp_path / "ds"
    _make_structured_dataset(ds)

    result = CliRunner().invoke(
        app,
        [
            "version-dataset",
            str(ds),
            "--dataset-name", "ds",
            "--registry-dir", str(tmp_path / "registry"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "REGISTERED" in result.output


def test_cli_version_dataset_exits_1_on_missing_path(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "version-dataset",
            str(tmp_path / "nope"),
            "--dataset-name", "ds",
            "--registry-dir", str(tmp_path / "registry"),
        ],
    )
    assert result.exit_code == 1


def test_cli_version_dataset_exits_1_when_validation_failed(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentic_mlops.cli.main import app

    ds = tmp_path / "ds"
    _make_structured_dataset(ds)
    report = _write_report(tmp_path / "validation.json", "failed")

    result = CliRunner().invoke(
        app,
        [
            "version-dataset",
            str(ds),
            "--dataset-name", "ds",
            "--registry-dir", str(tmp_path / "registry"),
            "--validation-report", report,
        ],
    )
    assert result.exit_code == 1
    assert "BLOCKED" in result.output
