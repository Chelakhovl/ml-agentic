"""Base class for all agentic MLOps agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agentic_mlops.observability.logging import get_logger


class BaseAgent(ABC):
    """Every agent in the workflow extends this class.

    Agents are responsible for orchestration and decision routing.
    They delegate computation to tools and never contain heavy ML logic directly.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self, input: Any) -> Any:
        """Execute the agent with the given input contract and return an output contract."""
        ...

    def _log_start(self, **extra: Any) -> None:
        self.logger.info(f"{self.__class__.__name__} starting", extra=extra)

    def _log_done(self, status: str, **extra: Any) -> None:
        self.logger.info(f"{self.__class__.__name__} finished", extra={"status": status, **extra})
