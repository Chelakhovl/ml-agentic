"""Unit tests for the observability tracing abstraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_mlops.observability.tracing import NoOpTracer, _SpanCtx

# ── NoOpTracer ────────────────────────────────────────────────────────────────


class TestNoOpTracer:
    def test_start_span_yields_span_ctx(self) -> None:
        tracer = NoOpTracer()
        with tracer.start_span("test.op") as span:
            assert isinstance(span, _SpanCtx)

    def test_span_set_attribute_does_not_raise(self) -> None:
        tracer = NoOpTracer()
        with tracer.start_span("test.attr") as span:
            span.set_attribute("key", "value")

    def test_span_record_exception_does_not_raise(self) -> None:
        tracer = NoOpTracer()
        with tracer.start_span("test.exc") as span:
            span.record_exception(ValueError("boom"))

    def test_span_set_status_ok_does_not_raise(self) -> None:
        tracer = NoOpTracer()
        with tracer.start_span("test.ok") as span:
            span.set_status_ok()

    def test_span_set_status_error_does_not_raise(self) -> None:
        tracer = NoOpTracer()
        with tracer.start_span("test.err") as span:
            span.set_status_error("something failed")

    def test_nested_spans_work(self) -> None:
        tracer = NoOpTracer()
        with tracer.start_span("outer") as outer:
            with tracer.start_span("inner") as inner:
                inner.set_attribute("level", "inner")
            outer.set_status_ok()


# ── BaseAgent.run_traced ──────────────────────────────────────────────────────


class _RecordingTracer(NoOpTracer):
    """Records span names and whether they finished ok or with an error."""

    def __init__(self) -> None:
        self.spans: list[dict] = []

    class _RecordingSpan(_SpanCtx):
        def __init__(self, record: dict) -> None:
            self._record = record

        def set_attribute(self, key: str, value: Any) -> None:
            self._record.setdefault("attributes", {})[key] = value

        def record_exception(self, exc: BaseException) -> None:
            self._record["exception"] = repr(exc)

        def set_status_ok(self) -> None:
            self._record["status"] = "ok"

        def set_status_error(self, description: str = "") -> None:
            self._record["status"] = "error"
            self._record["error_description"] = description

    def start_span(self, name: str, **attributes: Any):  # type: ignore[override]
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            record: dict = {"name": name, "attributes": dict(attributes)}
            self.spans.append(record)
            yield self._RecordingSpan(record)

        return _ctx()


def _make_agent(tmp_path: Path, tracer=None):
    from agentic_mlops.agents.base import BaseAgent

    class _StubAgent(BaseAgent):
        def run(self, input: Any) -> str:
            return "done"

    return _StubAgent(tmp_path / "artifacts", tracer=tracer)


def _make_failing_agent(tmp_path: Path, tracer=None):
    from agentic_mlops.agents.base import BaseAgent

    class _FailAgent(BaseAgent):
        def run(self, input: Any) -> None:
            raise RuntimeError("intentional failure")

    return _FailAgent(tmp_path / "artifacts", tracer=tracer)


class TestRunTraced:
    def test_run_traced_returns_result(self, tmp_path: Path) -> None:
        tracer = _RecordingTracer()
        agent = _make_agent(tmp_path, tracer=tracer)
        result = agent.run_traced("input")
        assert result == "done"

    def test_run_traced_records_ok_span(self, tmp_path: Path) -> None:
        tracer = _RecordingTracer()
        agent = _make_agent(tmp_path, tracer=tracer)
        agent.run_traced("input")
        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert "agent._StubAgent" in span["name"]
        assert span.get("status") == "ok"

    def test_run_traced_sets_agent_name_attribute(self, tmp_path: Path) -> None:
        tracer = _RecordingTracer()
        agent = _make_agent(tmp_path, tracer=tracer)
        agent.run_traced("input")
        attrs = tracer.spans[0].get("attributes", {})
        assert attrs.get("agent.name") == "_StubAgent"

    def test_run_traced_records_error_span_on_exception(self, tmp_path: Path) -> None:
        tracer = _RecordingTracer()
        agent = _make_failing_agent(tmp_path, tracer=tracer)
        with pytest.raises(RuntimeError, match="intentional failure"):
            agent.run_traced("input")
        span = tracer.spans[0]
        assert span.get("status") == "error"
        assert "exception" in span

    def test_default_tracer_is_noop(self, tmp_path: Path) -> None:
        from agentic_mlops.observability.tracing import NoOpTracer

        agent = _make_agent(tmp_path)  # no tracer kwarg
        assert isinstance(agent._tracer, NoOpTracer)

    def test_run_without_tracer_does_not_raise(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        result = agent.run_traced("anything")
        assert result == "done"
