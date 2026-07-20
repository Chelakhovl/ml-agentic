"""Read-only filesystem helpers for the web UI.

All functions scan on-disk JSON files and return plain Python dicts/lists.
No ORM, no database — the on-disk files written by WorkflowStateStore and
LocalModelRegistryClient are the source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Workflow readers ──────────────────────────────────────────────────────────


def is_safe_path_id(name: str) -> bool:
    """Reject path-traversal / separator characters in a user-supplied path segment.

    workflow_id / model_name / dataset_name / step all come straight from URL path
    or query params (see web/routes.py) and are joined onto runs_dir/registry_dir
    below — without this check a value like "../../etc" would escape the intended
    root for both reads and (via WorkflowStateStore, for workflow_id) writes.
    """
    return bool(name) and "/" not in name and "\\" not in name and name not in (".", "..")


def list_workflows(runs_dir: Path) -> list[dict]:
    """Return all workflow state dicts, sorted newest-first by started_at."""
    runs_dir = Path(runs_dir)
    results: list[dict] = []
    if not runs_dir.exists():
        return results
    for state_file in sorted(runs_dir.glob("*/state.json")):
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            results.append(state)
        except (json.JSONDecodeError, OSError):
            continue
    results.sort(key=lambda s: s.get("started_at", ""), reverse=True)
    return results


def get_workflow(runs_dir: Path, workflow_id: str) -> dict | None:
    if not is_safe_path_id(workflow_id):
        return None
    state_file = Path(runs_dir) / workflow_id / "state.json"
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_audit_log(runs_dir: Path, workflow_id: str) -> list[dict]:
    """Return audit log entries newest-first. Skips malformed lines."""
    if not is_safe_path_id(workflow_id):
        return []
    log_file = Path(runs_dir) / workflow_id / "audit_log.jsonl"
    if not log_file.exists():
        return []
    entries: list[dict] = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    entries.reverse()
    return entries


# ── Model registry readers ────────────────────────────────────────────────────


def list_models(registry_dir: Path) -> list[dict]:
    """Return one dict per registered model, sorted by registered_at desc."""
    registry_dir = Path(registry_dir)
    results: list[dict] = []
    if not registry_dir.exists():
        return results
    for latest_file in sorted(registry_dir.glob("*/latest.json")):
        try:
            data = json.loads(latest_file.read_text(encoding="utf-8"))
            data["model_name"] = data.get("model_name") or latest_file.parent.name
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    results.sort(key=lambda m: m.get("registered_at", ""), reverse=True)
    return results


def get_model_versions(registry_dir: Path, model_name: str) -> list[dict]:
    """Return one dict per version, sorted by version number desc."""
    if not is_safe_path_id(model_name):
        return []
    versions_dir = Path(registry_dir) / model_name / "versions"
    if not versions_dir.exists():
        return []
    results: list[dict] = []
    for version_dir in sorted(versions_dir.iterdir(), key=lambda p: p.name):
        if not version_dir.is_dir():
            continue
        try:
            version_num = int(version_dir.name)
        except ValueError:
            continue
        lineage_file = version_dir / "lineage.json"
        reg_file = version_dir / "registration_output.json"
        lineage: dict = {}
        reg_out: dict = {}
        if lineage_file.exists():
            try:
                lineage = json.loads(lineage_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        if reg_file.exists():
            try:
                reg_out = json.loads(reg_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        results.append({
            "version": version_num,
            "lineage": lineage,
            "registration": reg_out,
            "registered_at": reg_out.get("registered_at") or lineage.get("approval_timestamp", ""),
        })
    results.sort(key=lambda v: v["version"], reverse=True)
    return results


# ── Monitoring readers ────────────────────────────────────────────────────────


def list_datasets(registry_dir: Path) -> list[dict]:
    """Return one dict per registered dataset, sorted by registered_at desc."""
    registry_dir = Path(registry_dir)
    results: list[dict] = []
    if not registry_dir.exists():
        return results
    for latest_file in sorted(registry_dir.glob("*/latest.json")):
        try:
            data = json.loads(latest_file.read_text(encoding="utf-8"))
            data["dataset_name"] = data.get("dataset_name") or latest_file.parent.name
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    results.sort(key=lambda d: d.get("registered_at", ""), reverse=True)
    return results


def get_dataset_versions(registry_dir: Path, dataset_name: str) -> list[dict]:
    """Return one dict per version, sorted by version number desc."""
    if not is_safe_path_id(dataset_name):
        return []
    versions_dir = Path(registry_dir) / dataset_name / "versions"
    if not versions_dir.exists():
        return []
    results: list[dict] = []
    for version_dir in sorted(versions_dir.iterdir(), key=lambda p: p.name):
        if not version_dir.is_dir():
            continue
        try:
            version_num = int(version_dir.name)
        except ValueError:
            continue
        lineage_file = version_dir / "lineage.json"
        lineage: dict = {}
        if lineage_file.exists():
            try:
                lineage = json.loads(lineage_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        results.append({"version": version_num, "lineage": lineage})
    results.sort(key=lambda v: v["version"], reverse=True)
    return results


# ── Monitoring readers ────────────────────────────────────────────────────────


def list_monitoring_reports(runs_dir: Path) -> list[dict]:
    """Find all monitoring_report.json files under runs_dir, newest-first."""
    runs_dir = Path(runs_dir)
    results: list[dict] = []
    if not runs_dir.exists():
        return results
    for report_file in sorted(runs_dir.rglob("monitoring_report.json")):
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
            # Annotate with source context derived from the path
            parts = report_file.relative_to(runs_dir).parts
            data["_workflow_id"] = parts[0] if len(parts) > 0 else ""
            data["_step"] = parts[2] if len(parts) > 2 else "monitoring"
            # Ensure num_hard_samples is always present for the list template
            if "num_hard_samples" not in data:
                data["num_hard_samples"] = len(data.get("hard_samples") or [])
            results.append(data)
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    results.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
    return results


def get_monitoring_report(runs_dir: Path, workflow_id: str, step: str) -> dict | None:
    """Return one monitoring report with hard_samples, or None if not found."""
    if not is_safe_path_id(workflow_id) or not is_safe_path_id(step):
        return None
    report_dir = Path(runs_dir) / workflow_id / "artifacts" / step
    report_file = report_dir / "monitoring_report.json"
    if not report_file.exists():
        return None
    try:
        data = json.loads(report_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    data["_workflow_id"] = workflow_id
    data["_step"] = step
    # Hard samples are embedded in the report; fall back to the separate manifest file
    if not data.get("hard_samples"):
        manifest_file = report_dir / "hard_samples_manifest.json"
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                if isinstance(manifest, list):
                    data["hard_samples"] = manifest
            except (json.JSONDecodeError, OSError):
                pass
    return data
