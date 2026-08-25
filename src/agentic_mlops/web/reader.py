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
        results.append(
            {
                "version": version_num,
                "lineage": lineage,
                "registration": reg_out,
                "registered_at": reg_out.get("registered_at")
                or lineage.get("approval_timestamp", ""),
            }
        )
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


def get_dataset_quality_data(registry_dir: Path, dataset_name: str) -> list[dict]:
    """Return quality metrics for every version of a dataset, sorted newest-first.

    Each entry is a plain dict with:
        version, registered_at, validation_status, label_qa_status,
        num_images, num_labels, class_distribution {class: count}, classes [str]

    Versions without a quality_summary (registered before this feature was added)
    will have None for num_images/num_labels and an empty class_distribution.
    """
    versions = get_dataset_versions(registry_dir, dataset_name)
    result: list[dict] = []
    for v in versions:
        lin = v.get("lineage") or {}
        qs = lin.get("quality_summary") or {}
        result.append(
            {
                "version": v.get("version"),
                "registered_at": lin.get("registered_at"),
                "validation_status": lin.get("validation_status"),
                "label_qa_status": lin.get("label_qa_status"),
                "num_images": qs.get("num_images"),
                "num_labels": qs.get("num_labels"),
                "class_distribution": qs.get("class_distribution") or {},
                "blocking_issues_count": qs.get("blocking_issues_count", 0),
                "label_issues_count": qs.get("label_issues_count", 0),
                "classes": lin.get("classes") or [],
            }
        )
    return result


def get_dataset_version_detail(registry_dir: Path, dataset_name: str, version: int) -> dict | None:
    """Return full lineage + quality data for one specific dataset version, or None."""
    if not is_safe_path_id(dataset_name):
        return None
    ver_dir = Path(registry_dir) / dataset_name / "versions" / str(version)
    if not ver_dir.is_dir():
        return None
    lineage_file = ver_dir / "lineage.json"
    lineage: dict = {}
    if lineage_file.exists():
        try:
            lineage = json.loads(lineage_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    qs = lineage.get("quality_summary") or {}
    return {
        "version": version,
        "registered_at": lineage.get("registered_at"),
        "validation_status": lineage.get("validation_status"),
        "label_qa_status": lineage.get("label_qa_status"),
        "classes": lineage.get("classes") or [],
        "content_hash": lineage.get("content_hash"),
        "approved_by": lineage.get("approved_by"),
        "workflow_id": lineage.get("workflow_id"),
        "parent_version": lineage.get("parent_version"),
        "source_batches": lineage.get("source_batches") or [],
        "num_images": qs.get("num_images"),
        "num_labels": qs.get("num_labels"),
        "blocking_issues_count": qs.get("blocking_issues_count", 0),
        "label_issues_count": qs.get("label_issues_count", 0),
        "class_distribution": qs.get("class_distribution") or {},
    }


def get_dataset_diff(registry_dir: Path, dataset_name: str, v1: int, v2: int) -> dict | None:
    """Return a diff dict comparing two dataset versions, or None if either is missing."""
    a = get_dataset_version_detail(registry_dir, dataset_name, v1)
    b = get_dataset_version_detail(registry_dir, dataset_name, v2)
    if a is None or b is None:
        return None

    def _delta(key: str) -> dict:
        va, vb = a.get(key), b.get(key)
        changed = va != vb
        if isinstance(va, int | float) and isinstance(vb, int | float):
            delta = vb - va
        else:
            delta = None
        return {"a": va, "b": vb, "changed": changed, "delta": delta}

    classes_a = set(a["classes"])
    classes_b = set(b["classes"])
    return {
        "dataset_name": dataset_name,
        "v1": v1,
        "v2": v2,
        "fields": {
            "num_images": _delta("num_images"),
            "num_labels": _delta("num_labels"),
            "blocking_issues_count": _delta("blocking_issues_count"),
            "label_issues_count": _delta("label_issues_count"),
            "validation_status": _delta("validation_status"),
            "label_qa_status": _delta("label_qa_status"),
            "approved_by": _delta("approved_by"),
            "registered_at": _delta("registered_at"),
            "content_hash": _delta("content_hash"),
            "parent_version": _delta("parent_version"),
        },
        "classes_added": sorted(classes_b - classes_a),
        "classes_removed": sorted(classes_a - classes_b),
        "classes_common": sorted(classes_a & classes_b),
        "class_dist_a": a["class_distribution"],
        "class_dist_b": b["class_distribution"],
        "v1_source_batches": a["source_batches"],
        "v2_source_batches": b["source_batches"],
    }


def list_all_model_versions(registry_dir: Path) -> list[dict]:
    """Return one dict per registered version across all models, ranked by mAP50 desc.

    Each entry has: model_name, version, lineage, registered_at, map50, map50_95,
    precision, recall, rank (1-based; None when map50 is absent).
    """
    registry_dir = Path(registry_dir)
    results: list[dict] = []
    if not registry_dir.exists():
        return results
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
            lineage: dict = {}
            reg_out: dict = {}
            lineage_file = version_dir / "lineage.json"
            reg_file = version_dir / "registration_output.json"
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
            results.append(
                {
                    "model_name": model_name,
                    "version": version_num,
                    "lineage": lineage,
                    "registered_at": (
                        reg_out.get("registered_at") or lineage.get("approval_timestamp", "")
                    ),
                    "map50": lineage.get("map50"),
                    "map50_95": lineage.get("map50_95"),
                    "precision": lineage.get("precision"),
                    "recall": lineage.get("recall"),
                }
            )
    # Primary sort: map50 desc (None last); secondary: map50_95 desc
    results.sort(
        key=lambda v: (
            v["map50"] is None,
            -(v["map50"] or 0.0),
            v["map50_95"] is None,
            -(v["map50_95"] or 0.0),
        )
    )
    rank = 0
    for v in results:
        if v["map50"] is not None:
            rank += 1
            v["rank"] = rank
        else:
            v["rank"] = None
    return results


def get_risk_report(runs_dir: Path, workflow_id: str) -> dict | None:
    """Return the risk_report.json for a workflow's model_registry step, or None."""
    if not is_safe_path_id(workflow_id):
        return None
    path = Path(runs_dir) / workflow_id / "artifacts" / "model_registry" / "risk_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_run_diff(runs_dir: Path, wf_id_a: str, wf_id_b: str) -> dict | None:
    """Return a serialisable diff dict from WorkflowRunDiffer, or None."""
    from agentic_mlops.tools.run_diff import WorkflowRunDiffer  # noqa: PLC0415

    diff = WorkflowRunDiffer().diff(runs_dir, wf_id_a, wf_id_b)
    if diff is None:
        return None
    return {
        "wf_a": diff.wf_a,
        "wf_b": diff.wf_b,
        "status_a": diff.status_a,
        "status_b": diff.status_b,
        "started_at_a": diff.started_at_a,
        "started_at_b": diff.started_at_b,
        "step_diffs": [
            {"name": s.name, "status_a": s.status_a, "status_b": s.status_b, "changed": s.changed}
            for s in diff.step_diffs
        ],
        "metric_diffs": [
            {
                "metric": m.metric,
                "value_a": m.value_a,
                "value_b": m.value_b,
                "delta": m.delta,
                "improved": m.improved,
            }
            for m in diff.metric_diffs
        ],
        "steps_only_in_a": diff.steps_only_in_a,
        "steps_only_in_b": diff.steps_only_in_b,
        "has_changes": diff.has_changes,
    }


def list_cost_reports(runs_dir: Path) -> list[dict]:
    """Return one cost_report.json per workflow, newest-first."""
    runs_dir = Path(runs_dir)
    results: list[dict] = []
    if not runs_dir.exists():
        return results
    for report_file in sorted(runs_dir.glob("*/artifacts/cost_report.json")):
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
            data["workflow_id"] = report_file.parts[-3]
            results.append(data)
        except (json.JSONDecodeError, OSError, IndexError):
            continue
    results.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
    return results


def get_cost_report(runs_dir: Path, workflow_id: str) -> dict | None:
    """Return the cost_report.json written by CostTracker, or None."""
    if not is_safe_path_id(workflow_id):
        return None
    path = Path(runs_dir) / workflow_id / "artifacts" / "cost_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_baseline_comparison(runs_dir: Path, workflow_id: str) -> dict | None:
    """Return the baseline_comparison.json for a workflow's evaluation step, or None."""
    if not is_safe_path_id(workflow_id):
        return None
    path = Path(runs_dir) / workflow_id / "artifacts" / "evaluation" / "baseline_comparison.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


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


def list_hard_samples(runs_dir: Path) -> list[dict]:
    """Aggregate all hard_samples_manifest.json files, one entry per workflow/step pair."""
    runs_dir = Path(runs_dir)
    results: list[dict] = []
    if not runs_dir.exists():
        return results
    for manifest_file in sorted(runs_dir.rglob("hard_samples_manifest.json")):
        try:
            samples = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not isinstance(samples, list):
                continue
            rel_parts = manifest_file.relative_to(runs_dir).parts
            workflow_id = rel_parts[0] if rel_parts else ""
            step = rel_parts[2] if len(rel_parts) > 2 else "monitoring"
            results.append(
                {
                    "workflow_id": workflow_id,
                    "step": step,
                    "samples": samples,
                    "count": len(samples),
                }
            )
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    results.sort(key=lambda r: (r["workflow_id"], r["step"]))
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
