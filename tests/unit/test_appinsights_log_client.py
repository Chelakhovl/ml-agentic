"""Unit tests for appinsights_log_client.py and its integration with MonitoringAgent/CLI.

Coverage matrix:
    LocalFileLogClient
     1.  Reads valid JSONL records
     2.  Skips malformed JSON lines with warning
     3.  Returns warning (not error) for missing file
     4.  Skips non-object lines with warning

    FakeInferenceLogClient
     5.  Returns preset records unchanged
     6.  Returns preset warnings unchanged

    _rows_to_records / _normalise_detection
     7.  Converts class_id key to class
     8.  Handles invalid detections_json gracefully
     9.  Unknown column order handled by column name lookup

    ApplicationInsightsLogClient
    10.  Raises RuntimeError with install hint when azure-monitor-query absent
    11.  SUCCESS status: records returned, no warnings
    12.  KQL contains endpoint_name and window in seconds
    13.  PARTIAL status: records returned + warning added
    14.  FAILURE status: empty records + warning added
    15.  Query exception: empty records + warning added

    ModelMonitor with injected client
    16.  Injected FakeInferenceLogClient used regardless of source field
    17.  source="azure_monitor" without workspace_id returns failed output
    18.  source="azure_monitor" with injected client runs analysis normally
    19.  source="local" with injected client runs analysis normally

    MonitoringAgent with injected client
    20.  log_client passed to MonitoringAgent propagates to ModelMonitor
    21.  Agent with FakeInferenceLogClient writes report files

    CLI monitor extensions
    22.  --source local without predictions_log arg exits 1
    23.  --source azure-monitor without --app-insights-workspace-id exits 1
    24.  Invalid --source value exits 1
    25.  --source local with valid log succeeds (backward-compat)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_mlops.integrations.appinsights_log_client import (
    FakeInferenceLogClient,
    LocalFileLogClient,
    _normalise_detection,
    _rows_to_records,
)

# ── Helpers ─────────────────────────────────────────────────────────────────────────


class _FakeLogStatus:
    """Stand-in for azure.monitor.query.LogsQueryStatus."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILURE = "FAILURE"


def _make_record(
    image_id: str = "img.jpg",
    ts: str = "2026-07-10T12:00:00+00:00",
    latency_ms: float = 50.0,
    error: bool = False,
    confidences: list[float] | None = None,
    cls: str = "dent",
) -> dict:
    dets = [{"class": cls, "confidence": c} for c in (confidences or [0.9])]
    return {
        "timestamp": ts,
        "image_id": image_id,
        "latency_ms": latency_ms,
        "error": error,
        "detections": dets,
    }


def _write_log(tmp_path: Path, records: list[dict], name: str = "preds.jsonl") -> Path:
    p = tmp_path / name
    p.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return p


def _build_fake_azure_modules(
    status: str = "SUCCESS",
    table_data: list[tuple] | None = None,
    col_names: list[str] | None = None,
    raise_on_query: Exception | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (fake_query_module, fake_identity_module, client_mock)."""
    if table_data is None:
        table_data = [
            (
                "2026-07-10T12:00:00+00:00",
                "img.jpg",
                50.0,
                False,
                '[{"class_id": 0, "confidence": 0.9}]',
            )
        ]
    if col_names is None:
        col_names = ["log_timestamp", "image_id", "latency_ms", "is_error", "detections_json"]

    # Build fake LogsTable
    col_mocks = []
    for name in col_names:
        c = MagicMock()
        c.name = name
        col_mocks.append(c)
    table_mock = MagicMock()
    table_mock.columns = col_mocks
    table_mock.rows = table_data

    # Build fake response
    response_mock = MagicMock()
    response_mock.status = status
    if status == _FakeLogStatus.PARTIAL:
        response_mock.partial_data = [table_mock]
        response_mock.tables = []
        response_mock.partial_error = "partial_error_msg"
    elif status == _FakeLogStatus.FAILURE:
        response_mock.partial_data = []
        response_mock.tables = []
        response_mock.partial_error = "failure_error_msg"
    else:
        response_mock.partial_data = []
        response_mock.tables = [table_mock]

    # Build fake LogsQueryClient
    client_mock = MagicMock()
    if raise_on_query is not None:
        client_mock.query_workspace.side_effect = raise_on_query
    else:
        client_mock.query_workspace.return_value = response_mock

    fake_query_module = MagicMock()
    fake_query_module.LogsQueryClient = MagicMock(return_value=client_mock)
    fake_query_module.LogsQueryStatus = _FakeLogStatus

    fake_identity_module = MagicMock()
    fake_identity_module.DefaultAzureCredential = MagicMock(return_value=MagicMock())

    return fake_query_module, fake_identity_module, client_mock


# ── 1-4. LocalFileLogClient ───────────────────────────────────────────────────────


class TestLocalFileLogClient:
    def test_reads_valid_jsonl(self, tmp_path: Path) -> None:
        records = [_make_record("a"), _make_record("b")]
        log = _write_log(tmp_path, records)
        client = LocalFileLogClient(log)
        result, warnings = client.fetch_records()
        assert len(result) == 2
        assert result[0]["image_id"] == "a"
        assert result[1]["image_id"] == "b"
        assert warnings == []

    def test_skips_malformed_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text('{"ok": true}\nnot_json\n{"ok": true}\n', encoding="utf-8")
        result, warnings = LocalFileLogClient(p).fetch_records()
        assert len(result) == 2
        assert len(warnings) == 1
        assert "malformed JSON" in warnings[0]

    def test_skips_non_object_line(self, tmp_path: Path) -> None:
        p = tmp_path / "arr.jsonl"
        p.write_text('{"ok": true}\n[1, 2]\n', encoding="utf-8")
        result, warnings = LocalFileLogClient(p).fetch_records()
        assert len(result) == 1
        assert len(warnings) == 1
        assert "non-object" in warnings[0]

    def test_missing_file_returns_warning_not_exception(self, tmp_path: Path) -> None:
        client = LocalFileLogClient(tmp_path / "missing.jsonl")
        result, warnings = client.fetch_records()
        assert result == []
        assert len(warnings) == 1
        assert "Cannot read" in warnings[0]


# ── 5-6. FakeInferenceLogClient ───────────────────────────────────────────────────


class TestFakeInferenceLogClient:
    def test_returns_preset_records(self) -> None:
        records = [_make_record("x"), _make_record("y")]
        client = FakeInferenceLogClient(records)
        result, warnings = client.fetch_records()
        assert result == records
        assert warnings == []

    def test_returns_preset_warnings(self) -> None:
        client = FakeInferenceLogClient([], warnings=["w1", "w2"])
        _, warnings = client.fetch_records()
        assert warnings == ["w1", "w2"]


# ── 7-9. _rows_to_records / _normalise_detection ─────────────────────────────────


class TestRowsToRecords:
    def _make_table(self, rows: list[tuple], col_names: list[str]) -> MagicMock:
        col_mocks = []
        for name in col_names:
            c = MagicMock()
            c.name = name
            col_mocks.append(c)
        t = MagicMock()
        t.columns = col_mocks
        t.rows = rows
        return t

    def test_class_id_normalised_to_class(self) -> None:
        table = self._make_table(
            rows=[
                ("2026-07-10T00:00:00Z", "img.jpg", 10.0, False,
                 '[{"class_id": 2, "confidence": 0.8}]'),
            ],
            col_names=["log_timestamp", "image_id", "latency_ms", "is_error", "detections_json"],
        )
        records = _rows_to_records(table)
        assert records[0]["detections"][0]["class"] == "2"
        assert records[0]["detections"][0]["confidence"] == pytest.approx(0.8)

    def test_invalid_detections_json_becomes_empty_list(self) -> None:
        table = self._make_table(
            rows=[("ts", "img.jpg", 5.0, False, "not_json")],
            col_names=["log_timestamp", "image_id", "latency_ms", "is_error", "detections_json"],
        )
        records = _rows_to_records(table)
        assert records[0]["detections"] == []

    def test_normalise_prefers_class_over_class_id(self) -> None:
        d = {"class": "scratch", "class_id": 0, "confidence": 0.7}
        result = _normalise_detection(d)
        assert result["class"] == "scratch"


# ── 10-15. ApplicationInsightsLogClient ──────────────────────────────────────────


class TestApplicationInsightsLogClient:
    def _build_client(
        self,
        status: str = "SUCCESS",
        table_data: list[tuple] | None = None,
        raise_on_query: Exception | None = None,
        workspace_id: str = "ws-id-123",
        endpoint_name: str = "my-ep",
        window_seconds: int = 86400,
    ):
        from agentic_mlops.integrations.appinsights_log_client import (
            ApplicationInsightsLogClient,
        )

        fq, fi, client_mock = _build_fake_azure_modules(
            status=status, table_data=table_data, raise_on_query=raise_on_query
        )
        with patch.dict(sys.modules, {"azure.monitor.query": fq, "azure.identity": fi}):
            ai_client = ApplicationInsightsLogClient(
                workspace_id=workspace_id,
                endpoint_name=endpoint_name,
                window_seconds=window_seconds,
            )
        # keep reference so fetch_records can import LogsQueryStatus
        ai_client._test_query_module = fq
        ai_client._test_client_mock = client_mock
        return ai_client, fq, client_mock

    def test_raises_runtime_error_when_sdk_missing(self) -> None:
        from agentic_mlops.integrations.appinsights_log_client import (
            ApplicationInsightsLogClient,
        )

        with patch.dict(sys.modules, {"azure.monitor.query": None}):  # type: ignore[dict-item]
            with pytest.raises(RuntimeError, match="azure-monitor-query"):
                ApplicationInsightsLogClient(
                    workspace_id="ws", endpoint_name="ep", window_seconds=3600
                )

    def test_success_status_returns_records_no_warnings(self) -> None:
        ai_client, fq, _ = self._build_client()
        with patch.dict(sys.modules, {"azure.monitor.query": fq}):
            records, warnings = ai_client.fetch_records()
        assert len(records) == 1
        assert records[0]["image_id"] == "img.jpg"
        assert warnings == []

    def test_kql_contains_endpoint_name_and_window(self) -> None:
        ai_client, fq, client_mock = self._build_client(
            workspace_id="ws-xyz",
            endpoint_name="factory-ep",
            window_seconds=7200,
        )
        with patch.dict(sys.modules, {"azure.monitor.query": fq}):
            ai_client.fetch_records()
        call_kwargs = client_mock.query_workspace.call_args
        kql = call_kwargs.kwargs.get("query") or call_kwargs.args[1]
        assert "factory-ep" in kql
        assert "7200s" in kql

    def test_partial_status_returns_partial_data_and_warning(self) -> None:
        ai_client, fq, _ = self._build_client(status=_FakeLogStatus.PARTIAL)
        with patch.dict(sys.modules, {"azure.monitor.query": fq}):
            records, warnings = ai_client.fetch_records()
        assert len(records) == 1
        assert any("PARTIAL" in w for w in warnings)

    def test_failure_status_returns_empty_and_warning(self) -> None:
        ai_client, fq, _ = self._build_client(status=_FakeLogStatus.FAILURE)
        with patch.dict(sys.modules, {"azure.monitor.query": fq}):
            records, warnings = ai_client.fetch_records()
        assert records == []
        assert any("FAILURE" in w for w in warnings)

    def test_query_exception_returns_empty_and_warning(self) -> None:
        ai_client, fq, _ = self._build_client(
            raise_on_query=RuntimeError("network error")
        )
        with patch.dict(sys.modules, {"azure.monitor.query": fq}):
            records, warnings = ai_client.fetch_records()
        assert records == []
        assert any("network error" in w for w in warnings)


# ── 16-19. ModelMonitor with injected client ─────────────────────────────────────


class TestModelMonitorWithInjectedClient:
    def test_injected_client_used_regardless_of_source(self, tmp_path: Path) -> None:
        from agentic_mlops.contracts.monitoring import MonitoringInput
        from agentic_mlops.tools.monitor import ModelMonitor

        records = [_make_record("x", "2026-07-10T12:00:00+00:00")]
        client = FakeInferenceLogClient(records)
        monitor = ModelMonitor(log_client=client)
        result = monitor.run(
            MonitoringInput(endpoint_name="ep", predictions_log_path="irrelevant"),
            tmp_path / "out",
        )
        assert result.success is True
        assert result.total_predictions == 1

    def test_azure_monitor_source_without_workspace_id_fails(
        self, tmp_path: Path
    ) -> None:
        from agentic_mlops.contracts.monitoring import MonitoringInput, MonitoringStatus
        from agentic_mlops.tools.monitor import ModelMonitor

        monitor = ModelMonitor()
        result = monitor.run(
            MonitoringInput(endpoint_name="ep", source="azure_monitor"),
            tmp_path / "out",
        )
        assert result.success is False
        assert result.status == MonitoringStatus.FAILED
        assert "app_insights_workspace_id" in result.message

    def test_azure_monitor_source_with_injected_client_runs_normally(
        self, tmp_path: Path
    ) -> None:
        from agentic_mlops.contracts.monitoring import MonitoringInput
        from agentic_mlops.tools.monitor import ModelMonitor

        records = [_make_record("z", "2026-07-10T12:00:00+00:00")]
        client = FakeInferenceLogClient(records)
        monitor = ModelMonitor(log_client=client)
        result = monitor.run(
            MonitoringInput(
                endpoint_name="ep",
                source="azure_monitor",
                app_insights_workspace_id="ws-id",
            ),
            tmp_path / "out",
        )
        assert result.success is True
        assert result.endpoint_name == "ep"

    def test_injected_client_warnings_propagate_to_output(
        self, tmp_path: Path
    ) -> None:
        from agentic_mlops.contracts.monitoring import MonitoringInput
        from agentic_mlops.tools.monitor import ModelMonitor

        records = [_make_record("a", "2026-07-10T12:00:00+00:00")]
        client = FakeInferenceLogClient(records, warnings=["watch out"])
        monitor = ModelMonitor(log_client=client)
        result = monitor.run(
            MonitoringInput(endpoint_name="ep", predictions_log_path=""),
            tmp_path / "out",
        )
        assert any("watch out" in w for w in result.warnings)


# ── 20-21. MonitoringAgent with injected client ───────────────────────────────────


class TestMonitoringAgentWithInjectedClient:
    def test_agent_passes_log_client_to_monitor(self, tmp_path: Path) -> None:
        from agentic_mlops.agents.monitoring import MonitoringAgent
        from agentic_mlops.contracts.monitoring import MonitoringInput

        records = [_make_record("a", "2026-07-10T12:00:00+00:00")]
        client = FakeInferenceLogClient(records)
        agent = MonitoringAgent(artifacts_dir=tmp_path / "artifacts", log_client=client)
        result = agent.run(MonitoringInput(endpoint_name="ep"))
        assert result.success is True
        assert result.total_predictions == 1

    def test_agent_with_fake_client_writes_report_files(self, tmp_path: Path) -> None:
        from agentic_mlops.agents.monitoring import MonitoringAgent
        from agentic_mlops.contracts.monitoring import MonitoringInput

        records = [_make_record("b", "2026-07-10T12:00:00+00:00", confidences=[0.95])]
        client = FakeInferenceLogClient(records)
        artifacts_dir = tmp_path / "artifacts"
        agent = MonitoringAgent(artifacts_dir=artifacts_dir, log_client=client)
        result = agent.run(MonitoringInput(endpoint_name="ep"))

        assert (artifacts_dir / "monitoring_report.json").exists()
        assert (artifacts_dir / "monitoring_report.md").exists()
        data = json.loads((artifacts_dir / "monitoring_report.json").read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert result.monitoring_report_path == str(artifacts_dir / "monitoring_report.json")


# ── 22-25. CLI monitor extensions ────────────────────────────────────────────────


class TestCLIMonitorExtensions:
    def test_local_source_without_predictions_log_exits_1(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        result = CliRunner().invoke(
            app,
            ["monitor", "--endpoint-name", "ep", "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        assert "predictions_log" in result.output.lower() or "required" in result.output.lower()

    def test_azure_monitor_without_workspace_id_exits_1(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        result = CliRunner().invoke(
            app,
            [
                "monitor",
                "--endpoint-name", "ep",
                "--source", "azure-monitor",
                "--output-dir", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1
        assert "workspace-id" in result.output or "workspace_id" in result.output

    def test_invalid_source_exits_1(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        result = CliRunner().invoke(
            app,
            [
                "monitor",
                "--endpoint-name", "ep",
                "--source", "kafka",
                "--output-dir", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1
        assert "kafka" in result.output

    def test_local_source_with_valid_log_succeeds(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        log = _write_log(tmp_path, [_make_record("a", "2026-07-10T12:00:00+00:00")])
        result = CliRunner().invoke(
            app,
            [
                "monitor",
                str(log),
                "--endpoint-name", "ep",
                "--output-dir", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "COMPLETED" in result.output
