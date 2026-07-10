"""MLflow experiment tracking integration.

Four implementations:
- NoOpMLflowTrackingClient  : safe default, no mlflow import required
- LocalMLflowTrackingClient : uses mlflow Python package (file-based or remote)
- FakeMLflowTrackingClient  : in-memory test double
- MLflowClient              : legacy stub (raises NotImplementedError, kept for compat)
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MLflowTrackingClientBase(ABC):
    """Abstract interface for MLflow experiment tracking."""

    @abstractmethod
    def start_run(self, experiment_name: str, run_name: str) -> str:
        """Create a new run and return its run_id."""
        ...

    @abstractmethod
    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters."""
        ...

    @abstractmethod
    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """Log a dictionary of scalar metrics."""
        ...

    @abstractmethod
    def log_artifact(self, run_id: str, local_path: str) -> None:
        """Upload a local file as an artifact."""
        ...

    @abstractmethod
    def log_tags(self, run_id: str, tags: dict[str, str]) -> None:
        """Set key-value tags on a run."""
        ...

    @abstractmethod
    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        """Mark a run as finished."""
        ...


# Backwards-compatibility alias used by existing code
MLflowClientBase = MLflowTrackingClientBase


class NoOpMLflowTrackingClient(MLflowTrackingClientBase):
    """No-op client — safe when MLflow is disabled.

    Does nothing and does not require the mlflow package to be installed.
    """

    def start_run(self, experiment_name: str, run_name: str) -> str:
        return f"noop_{uuid.uuid4().hex[:8]}"

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        pass

    def log_artifact(self, run_id: str, local_path: str) -> None:
        pass

    def log_tags(self, run_id: str, tags: dict[str, str]) -> None:
        pass

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        pass


class LocalMLflowTrackingClient(MLflowTrackingClientBase):
    """Uses the mlflow Python package for local SQLite or remote tracking.

    Args:
        tracking_uri: MLflow tracking URI.
            Recommended: ``sqlite:///outputs/mlruns.db`` (default).
            Legacy file-based URIs (``file:./mlruns``) are also accepted; the
            ``MLFLOW_ALLOW_FILE_STORE`` env-var is set automatically so that
            MLflow ≥ 3.14 does not reject them.

    Raises:
        RuntimeError: if the ``mlflow`` package is not installed.
    """

    def __init__(self, tracking_uri: str) -> None:
        try:
            import os  # noqa: PLC0415

            import mlflow  # noqa: PLC0415

            # MLflow ≥ 3.14 disables the file-store backend by default.
            # Honour it when a file: URI is requested (e.g. legacy configs).
            if tracking_uri.startswith("file:") or tracking_uri.startswith("./"):
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            self._client = mlflow.tracking.MlflowClient()
        except ImportError as exc:
            raise RuntimeError(
                "mlflow is not installed. Install it with: pip install mlflow"
            ) from exc

    def start_run(self, experiment_name: str, run_name: str) -> str:
        experiment = self._mlflow.set_experiment(experiment_name)
        run = self._client.create_run(
            experiment.experiment_id,
            tags={"mlflow.runName": run_name},
        )
        logger.info(
            "MLflow run started",
            extra={"run_id": run.info.run_id, "experiment": experiment_name},
        )
        return run.info.run_id

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        for key, value in params.items():
            try:
                self._client.log_param(run_id, str(key)[:250], str(value)[:250])
            except Exception as exc:
                logger.warning("Failed to log param %s: %s", key, exc)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        import time  # noqa: PLC0415

        timestamp = int(time.time() * 1000)
        for key, value in metrics.items():
            try:
                self._client.log_metric(
                    run_id, str(key)[:250], float(value), timestamp=timestamp
                )
            except Exception as exc:
                logger.warning("Failed to log metric %s: %s", key, exc)

    def log_artifact(self, run_id: str, local_path: str) -> None:
        if not Path(local_path).exists():
            return
        try:
            self._client.log_artifact(run_id, local_path)
        except Exception as exc:
            logger.warning("Failed to log artifact %s: %s", local_path, exc)

    def log_tags(self, run_id: str, tags: dict[str, str]) -> None:
        for key, value in tags.items():
            try:
                self._client.set_tag(run_id, str(key)[:250], str(value)[:250])
            except Exception as exc:
                logger.warning("Failed to set tag %s: %s", key, exc)

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        try:
            self._client.update_run(run_id, status=status)
            logger.info("MLflow run ended", extra={"run_id": run_id, "status": status})
        except Exception as exc:
            logger.warning("Failed to end MLflow run %s: %s", run_id, exc)


class FakeMLflowTrackingClient(MLflowTrackingClientBase):
    """In-memory test double.

    Records every call so tests can assert on interactions without a real MLflow server.
    """

    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}

    def start_run(self, experiment_name: str, run_name: str) -> str:
        run_id = f"fake_{uuid.uuid4().hex[:8]}"
        self.runs[run_id] = {
            "experiment": experiment_name,
            "name": run_name,
            "params": {},
            "metrics": {},
            "artifacts": [],
            "tags": {},
            "status": "running",
        }
        return run_id

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        self.runs[run_id]["params"].update(params)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        self.runs[run_id]["metrics"].update(metrics)

    def log_artifact(self, run_id: str, local_path: str) -> None:
        self.runs[run_id]["artifacts"].append(local_path)

    def log_tags(self, run_id: str, tags: dict[str, str]) -> None:
        self.runs[run_id]["tags"].update(tags)

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        self.runs[run_id]["status"] = status.lower()


# Backwards-compatibility alias used by existing tests
FakeMLflowClient = FakeMLflowTrackingClient


class MLflowClient(MLflowTrackingClientBase):
    """Legacy stub — all methods raise NotImplementedError.

    Retained for backwards compatibility. Use NoOpMLflowTrackingClient or
    LocalMLflowTrackingClient for new code.
    """

    def __init__(self) -> None:
        logger.warning(
            "MLflowClient is a legacy stub. "
            "Use LocalMLflowTrackingClient or NoOpMLflowTrackingClient instead."
        )

    def start_run(self, experiment_name: str, run_name: str) -> str:
        raise NotImplementedError(
            "MLflow tracking is not configured. "
            "Use LocalMLflowTrackingClient with a tracking URI."
        )

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        raise NotImplementedError

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        raise NotImplementedError

    def log_artifact(self, run_id: str, local_path: str) -> None:
        raise NotImplementedError

    def log_tags(self, run_id: str, tags: dict[str, str]) -> None:
        raise NotImplementedError

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        raise NotImplementedError
