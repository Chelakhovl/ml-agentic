"""Thin OpenTelemetry tracing abstraction.

Usage
-----
By default agents use a ``NoOpTracer`` (no external dependency).
When the ``[otel]`` extra is installed, pass an ``OtelTracer`` instance to
the workflow/agent constructor to get real OTLP spans:

    from agentic_mlops.observability.tracing import OtelTracer
    tracer = OtelTracer(service_name="agentic-mlops")
    workflow = OrchestratorWorkflow(tracer=tracer)

Span attributes emitted by every agent span
--------------------------------------------
- ``agent.name``  — class name of the agent
- ``agent.status`` — "ok" or "error"
- ``agent.error``  — exception type/message on failure (error span only)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

# ── Abstract base ─────────────────────────────────────────────────────────────


class _SpanCtx:
    """Minimal span handle returned by the abstract tracer."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ARG002
        pass

    def record_exception(self, exc: BaseException) -> None:  # noqa: ARG002
        pass

    def set_status_ok(self) -> None:
        pass

    def set_status_error(self, description: str = "") -> None:  # noqa: ARG002
        pass


class BaseTracer(ABC):
    @abstractmethod
    @contextmanager
    def start_span(self, name: str, **attributes: Any) -> Generator[_SpanCtx, None, None]:
        ...


# ── No-op implementation (zero dependencies) ─────────────────────────────────


class NoOpTracer(BaseTracer):
    """Default tracer — does nothing; requires no packages."""

    @contextmanager
    def start_span(self, name: str, **attributes: Any) -> Generator[_SpanCtx, None, None]:  # noqa: ARG002
        yield _SpanCtx()


# ── Real OTLP implementation (requires [otel] extra) ─────────────────────────


class OtelTracer(BaseTracer):
    """Real OpenTelemetry tracer.

    Requires the ``[otel]`` extra::

        pip install "agentic-mlops[otel]"

    Parameters
    ----------
    service_name:
        Resource service.name attribute for the OTLP exporter.
    tracer_provider:
        Inject a pre-configured ``TracerProvider``; if None, a default
        ``TracerProvider`` backed by a ``SimpleSpanProcessor`` +
        ``ConsoleSpanExporter`` is created automatically (handy for local
        debugging — swap for an OTLP exporter in production).
    """

    def __init__(
        self,
        service_name: str = "agentic-mlops",
        tracer_provider: Any = None,
    ) -> None:
        try:
            from opentelemetry import trace  # type: ignore[import-untyped]
            from opentelemetry.sdk.resources import Resource  # type: ignore[import-untyped]
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-untyped]
            from opentelemetry.sdk.trace.export import (  # type: ignore[import-untyped]
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenTelemetry packages are not installed. "
                "Install the [otel] extra: pip install 'agentic-mlops[otel]'"
            ) from exc

        if tracer_provider is None:
            resource = Resource.create({"service.name": service_name})
            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )

        self._trace = trace
        self._tracer = tracer_provider.get_tracer(service_name)
        self._StatusCode = trace.StatusCode

    @contextmanager
    def start_span(self, name: str, **attributes: Any) -> Generator[_SpanCtx, None, None]:
        with self._tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
            ctx = _OtelSpanCtx(span, self._StatusCode)
            yield ctx


class _OtelSpanCtx(_SpanCtx):
    def __init__(self, span: Any, StatusCode: Any) -> None:
        self._span = span
        self._StatusCode = StatusCode

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def record_exception(self, exc: BaseException) -> None:
        self._span.record_exception(exc)

    def set_status_ok(self) -> None:
        self._span.set_status(self._StatusCode.OK)

    def set_status_error(self, description: str = "") -> None:
        self._span.set_status(self._StatusCode.ERROR, description)
