"""Base class for all agentic MLOps agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agentic_mlops.observability.logging import get_logger
from agentic_mlops.observability.tracing import BaseTracer, NoOpTracer


class BaseAgent(ABC):
    """Every agent in the workflow extends this class.

    Agents are responsible for orchestration and decision routing.
    They delegate computation to tools and never contain heavy ML logic directly.
    """

    def __init__(self, artifacts_dir: Path, tracer: BaseTracer | None = None) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)
        self._tracer: BaseTracer = tracer if tracer is not None else NoOpTracer()

    @abstractmethod
    def run(self, input: Any) -> Any:
        """Execute the agent with the given input contract and return an output contract."""
        ...

    def run_traced(self, input: Any) -> Any:
        """Like ``run()`` but wraps execution in an OpenTelemetry span.

        Each concrete agent calls ``run()`` directly; the orchestrator uses
        ``run_traced()`` so callers that only have a ``BaseAgent`` reference
        still get spans without touching per-agent code.
        """
        span_name = f"agent.{self.__class__.__name__}"
        with self._tracer.start_span(
            span_name, **{"agent.name": self.__class__.__name__}
        ) as span:
            try:
                result = self.run(input)
                span.set_status_ok()
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status_error(str(exc))
                span.set_attribute("agent.error", f"{type(exc).__name__}: {exc}")
                raise

    def _log_start(self, **extra: Any) -> None:
        self.logger.info(f"{self.__class__.__name__} starting", extra=extra)

    def _log_done(self, status: str, **extra: Any) -> None:
        self.logger.info(f"{self.__class__.__name__} finished", extra={"status": status, **extra})
