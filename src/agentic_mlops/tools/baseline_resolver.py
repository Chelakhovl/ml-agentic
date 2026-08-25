"""Auto-resolve the best registered model as a training baseline.

Scans the local model registry (same on-disk format as LocalModelRegistryClient)
to find the version with the highest mAP50, then writes a synthetic
evaluation_report.json so BaselineComparator can diff against it without needing
the original training run to still be on disk.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)

_METRIC_KEYS = ("map50", "map50_95", "precision", "recall")


class BaselineResolveResult:
    """Metrics from the best registered model version."""

    def __init__(
        self,
        model_name: str,
        version: int,
        map50: float,
        map50_95: float,
        precision: float,
        recall: float,
        registry_dir: str,
    ) -> None:
        self.model_name = model_name
        self.version = version
        self.map50 = map50
        self.map50_95 = map50_95
        self.precision = precision
        self.recall = recall
        self.registry_dir = registry_dir

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BaselineResolveResult({self.model_name}:v{self.version}, " f"mAP50={self.map50:.4f})"
        )


class BaselineResolver:
    """Finds the best registered model version to use as an automatic baseline.

    Usage::

        resolver = BaselineResolver()
        result = resolver.resolve("outputs/model_registry", exclude_model_name="yolo-model")
        if result:
            path = resolver.write_synthetic_report(result, output_dir)
            # pass `path` to BaselineComparator as baseline_path
    """

    def resolve(
        self,
        registry_dir: Path | str,
        *,
        exclude_model_name: str | None = None,
        exclude_version: int | None = None,
    ) -> BaselineResolveResult | None:
        """Return the registered version with the highest mAP50, or ``None``.

        Args:
            registry_dir: Root of the local model registry (contains per-model
                sub-directories, each with ``versions/<N>/lineage.json``).
            exclude_model_name: Skip all versions of this model — useful when
                the model being evaluated is already in the registry and you
                don't want to compare it against itself.
            exclude_version: Skip this specific version of *exclude_model_name*.
                Has no effect when *exclude_model_name* is ``None``.
        """
        registry_dir = Path(registry_dir)
        if not registry_dir.exists():
            logger.warning(
                "BaselineResolver: registry_dir does not exist; no baseline resolved",
                extra={"registry_dir": str(registry_dir)},
            )
            return None

        best: BaselineResolveResult | None = None

        for model_dir in sorted(registry_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            versions_dir = model_dir / "versions"
            if not versions_dir.exists():
                continue
            for version_dir in sorted(versions_dir.iterdir(), key=lambda p: p.name):
                if not version_dir.is_dir():
                    continue
                try:
                    version_num = int(version_dir.name)
                except ValueError:
                    continue

                # Apply exclusion filter
                if exclude_model_name and model_name == exclude_model_name:
                    if exclude_version is None or version_num == exclude_version:
                        continue

                lineage_file = version_dir / "lineage.json"
                if not lineage_file.exists():
                    continue
                try:
                    lineage = json.loads(lineage_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue

                map50 = lineage.get("map50")
                if map50 is None:
                    continue  # no metrics recorded — skip
                map50 = float(map50)

                if best is None or map50 > best.map50:
                    best = BaselineResolveResult(
                        model_name=model_name,
                        version=version_num,
                        map50=map50,
                        map50_95=float(lineage.get("map50_95") or 0.0),
                        precision=float(lineage.get("precision") or 0.0),
                        recall=float(lineage.get("recall") or 0.0),
                        registry_dir=str(registry_dir),
                    )

        if best is not None:
            logger.info(
                "BaselineResolver: resolved baseline",
                extra={
                    "model_name": best.model_name,
                    "version": best.version,
                    "map50": best.map50,
                },
            )
        else:
            logger.info("BaselineResolver: no suitable baseline found in registry")

        return best

    def write_synthetic_report(
        self,
        result: BaselineResolveResult,
        output_dir: Path | str,
        filename: str = "auto_baseline_report.json",
    ) -> Path:
        """Write a synthetic evaluation_report.json that BaselineComparator can parse.

        The file has the same top-level ``metrics`` structure as a real
        ``EvaluationAgent`` output, so ``BaselineComparator._read_metrics()``
        works without modification.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        report = {
            "auto_resolved_baseline": True,
            "baseline_model_name": result.model_name,
            "baseline_version": result.version,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "metrics": {
                "map50": result.map50,
                "map50_95": result.map50_95,
                "precision": result.precision,
                "recall": result.recall,
            },
        }
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(
            "BaselineResolver: synthetic baseline report written",
            extra={"path": str(out_path)},
        )
        return out_path


class FakeBaselineResolver(BaselineResolver):
    """Test double — returns a fixed result without touching the filesystem."""

    def __init__(self, result: BaselineResolveResult | None = None) -> None:
        self._result = result or BaselineResolveResult(
            model_name="fake-model",
            version=1,
            map50=0.85,
            map50_95=0.72,
            precision=0.88,
            recall=0.82,
            registry_dir="fake_registry",
        )
        self.resolve_calls: list[dict] = []

    def resolve(
        self,
        registry_dir: Path | str,
        *,
        exclude_model_name: str | None = None,
        exclude_version: int | None = None,
    ) -> BaselineResolveResult | None:
        self.resolve_calls.append(
            {
                "registry_dir": str(registry_dir),
                "exclude_model_name": exclude_model_name,
                "exclude_version": exclude_version,
            }
        )
        return self._result
