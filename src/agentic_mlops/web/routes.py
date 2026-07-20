"""Route handlers for the MLOps web dashboard."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)

from agentic_mlops.contracts.orchestrator import DEFAULT_STEPS, PIPELINE_STEPS

from . import reader

router = APIRouter()

# ── Helpers ───────────────────────────────────────────────────────────────────

_STEP_ICONS: dict[str, str] = {
    "data_intake": "📥",
    "dataset_structuring": "🗂",
    "dataset_validation": "✅",
    "dataset_versioning": "🏷",
    "training_approval": "👍",
    "training": "🏋",
    "evaluation": "📊",
    "model_decision": "🤔",
    "approval": "🔏",
    "model_registry": "📦",
    "deployment": "🚀",
}

H4_ACTIONS = ["approve_training", "reject_training", "cancel"]
H5_ACTIONS = [
    "approve_model",
    "reject_model",
    "request_retraining",
    "request_more_data",
    "request_label_review",
    "cancel",
]


def _tmpl(request: Request):
    return request.app.state.templates


def _runs(request: Request) -> Path:
    return request.app.state.runs_dir


def _registry(request: Request) -> Path:
    return request.app.state.registry_dir


def _datasets(request: Request) -> Path:
    return request.app.state.datasets_dir


def _status_class(status: str) -> str:
    s = (status or "").lower()
    if s == "completed":
        return "badge-completed"
    if s in ("running",):
        return "badge-running"
    if s == "pending_approval":
        return "badge-pending"
    if s in ("failed",):
        return "badge-failed"
    if s == "blocked":
        return "badge-blocked"
    if s == "skipped":
        return "badge-skipped"
    return "badge-default"


def _fmt_dt(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso[:16]


def _audit_icon(event: str) -> str:
    icons = {
        "workflow_started": "▶",
        "workflow_resumed": "↩",
        "workflow_finished": "■",
        "workflow_already_completed": "✓",
        "step_started": "→",
        "step_finished": "✓",
        "step_exception": "✕",
        "workflow_failed_precheck": "✕",
    }
    return icons.get(event, "·")


def _action_class(action: str) -> str:
    """Badge CSS class for MonitoringOutput.recommended_action values."""
    mapping = {
        "no_action": "badge-completed",
        "notify_ops": "badge-default",
        "need_more_data": "badge-pending",
        "model_review": "badge-failed",
        "create_retraining_request": "badge-blocked",
    }
    return mapping.get(action or "no_action", "badge-default")


def _render(request: Request, template: str, ctx: dict[str, Any]) -> HTMLResponse:
    ctx.setdefault("request", request)
    ctx.setdefault("status_class", _status_class)
    ctx.setdefault("action_class", _action_class)
    ctx.setdefault("fmt_dt", _fmt_dt)
    ctx.setdefault("audit_icon", _audit_icon)
    ctx.setdefault("step_icon", lambda s: _STEP_ICONS.get(s, "⚙"))
    ctx.setdefault("flash", request.query_params.get("flash", ""))
    return _tmpl(request).TemplateResponse(template, ctx)


# ── Dashboard ─────────────────────────────────────────────────────────────────


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    workflows = reader.list_workflows(_runs(request))
    by_status: dict[str, int] = {}
    for wf in workflows:
        s = wf.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
    stats = {
        "total": len(workflows),
        "running": by_status.get("running", 0),
        "pending": by_status.get("pending_approval", 0),
        "failed": by_status.get("failed", 0) + by_status.get("blocked", 0),
        "completed": by_status.get("completed", 0),
    }
    return _render(request, "dashboard.html", {"workflows": workflows, "stats": stats})


# ── New workflow ──────────────────────────────────────────────────────────────


@router.get("/workflows/new", response_class=HTMLResponse)
async def new_workflow_form(request: Request):
    default_id = f"wf_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
    return _render(request, "new_workflow.html", {
        "pipeline_steps": PIPELINE_STEPS,
        "default_steps": DEFAULT_STEPS,
        "default_id": default_id,
    })


@router.post("/workflows/new")
async def new_workflow_submit(
    request: Request,
    workflow_id: Annotated[str, Form()],
    steps: Annotated[list[str] | None, Form()] = None,
    dataset_path: Annotated[str, Form()] = "",
    data_yaml_path: Annotated[str, Form()] = "",
    training_runner: Annotated[str, Form()] = "fake",
    dry_run: Annotated[str, Form()] = "true",
    approver: Annotated[str, Form()] = "",
    approval_action: Annotated[str, Form()] = "",
    model_name: Annotated[str, Form()] = "yolo-model",
    training_config_path: Annotated[str, Form()] = "",
    promotion_policy_path: Annotated[str, Form()] = "",
):
    workflow_id = workflow_id.strip()
    if not reader.is_safe_path_id(workflow_id):
        return _render(request, "new_workflow.html", {
            "pipeline_steps": PIPELINE_STEPS,
            "default_steps": DEFAULT_STEPS,
            "default_id": workflow_id,
            "error": "workflow_id is required and may not contain '/', '\\', '.' or '..'",
        })

    is_dry = dry_run.lower() in ("true", "1", "on", "yes")
    chosen_steps = [s for s in PIPELINE_STEPS if s in (steps or list(DEFAULT_STEPS))]

    from agentic_mlops.contracts.approvals import ApprovalAction
    from agentic_mlops.contracts.orchestrator import OrchestratorInput

    inp_kwargs: dict[str, Any] = {
        "workflow_id": workflow_id,
        "runs_dir": str(_runs(request)),
        "steps": chosen_steps or list(DEFAULT_STEPS),
        "dry_run": is_dry,
        "model_name": model_name or "yolo-model",
        # A web request has no TTY for HumanApprovalAgent's / TrainingApprovalAgent's
        # input() prompt — always run non-interactively so the orchestrator pauses at
        # its own H4/H5 pending_approval gate instead of blocking the background
        # executor thread forever.
        "interactive_approval": False,
        "interactive_training_approval": False,
    }
    if dataset_path.strip():
        inp_kwargs["dataset_path"] = dataset_path.strip()
    if data_yaml_path.strip():
        inp_kwargs["data_yaml_path"] = data_yaml_path.strip()
    if training_runner.strip():
        inp_kwargs["training_runner"] = training_runner.strip()
    if approver.strip():
        inp_kwargs["approver"] = approver.strip()
    if approval_action.strip():
        try:
            inp_kwargs["approval_action"] = ApprovalAction(approval_action.strip())
        except ValueError:
            pass
    if training_config_path.strip():
        inp_kwargs["training_config_path"] = training_config_path.strip()
    if promotion_policy_path.strip():
        inp_kwargs["promotion_policy_path"] = promotion_policy_path.strip()

    if not is_dry:
        # For real training, show CLI command — don't execute from web
        cli_cmd = (
            f"agentic-mlops run-workflow \\\n"
            f"  --workflow-id {workflow_id} \\\n"
            f"  --runs-dir {inp_kwargs['runs_dir']} \\\n"
            f"  --config configs/orchestrator.yaml"
        )
        return _render(request, "new_workflow.html", {
            "pipeline_steps": PIPELINE_STEPS,
            "default_steps": chosen_steps,
            "default_id": workflow_id,
            "cli_command": cli_cmd,
            "info": (
                "Non-dry-run workflows must be started from the CLI to support "
                "long-running training jobs."
            ),
        })

    # dry_run=True: run in background thread
    try:
        inp = OrchestratorInput(**inp_kwargs)
    except Exception as exc:
        return _render(request, "new_workflow.html", {
            "pipeline_steps": PIPELINE_STEPS,
            "default_steps": chosen_steps,
            "default_id": workflow_id,
            "error": f"Invalid input: {exc}",
        })

    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow

    def _run():
        OrchestratorWorkflow().run(inp)

    asyncio.get_running_loop().run_in_executor(None, _run)
    return RedirectResponse(f"/workflows/{workflow_id}?flash=Workflow+started", status_code=303)


# ── Workflow detail ───────────────────────────────────────────────────────────


@router.get("/workflows/{workflow_id}", response_class=HTMLResponse)
async def workflow_detail(request: Request, workflow_id: str):
    state = reader.get_workflow(_runs(request), workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    audit_log = reader.get_audit_log(_runs(request), workflow_id)
    return _render(request, "workflow_detail.html", {
        "state": state,
        "audit_log": audit_log,
        "pipeline_steps": PIPELINE_STEPS,
    })


# ── Approval ──────────────────────────────────────────────────────────────────


def _detect_gate(state: dict) -> str:
    """Returns 'h4', 'h5', or '' (not pending)."""
    cs = (state.get("current_state") or "").upper()
    if "TRAINING_APPROVAL" in cs:
        return "h4"
    if "MODEL_APPROVAL" in cs or state.get("status") == "pending_approval":
        return "h5"
    return ""


@router.get("/approve/{workflow_id}", response_class=HTMLResponse)
async def approve_form(request: Request, workflow_id: str):
    state = reader.get_workflow(_runs(request), workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    gate = _detect_gate(state)
    if not gate:
        return RedirectResponse(
            f"/workflows/{workflow_id}?flash=No+approval+pending", status_code=303
        )
    actions = H4_ACTIONS if gate == "h4" else H5_ACTIONS
    return _render(request, "approve.html", {
        "state": state,
        "gate": gate,
        "actions": actions,
    })


@router.post("/approve/{workflow_id}")
async def approve_submit(
    request: Request,
    workflow_id: str,
    action: Annotated[str, Form()],
    approver: Annotated[str, Form()] = "",
    comment: Annotated[str, Form()] = "",
):
    state = reader.get_workflow(_runs(request), workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    gate = _detect_gate(state)
    if not gate:
        return RedirectResponse(
            f"/workflows/{workflow_id}?flash=No+approval+pending", status_code=303
        )

    allowed = H4_ACTIONS if gate == "h4" else H5_ACTIONS
    if action not in allowed:
        return RedirectResponse(
            f"/workflows/{workflow_id}?flash=Invalid+action+for+{gate}", status_code=303
        )

    # Feed the decision back into the paused orchestrator run as the field the
    # relevant gate step actually reads (training_approval_action / approval_action)
    # — TrainingApprovalAgent/HumanApprovalAgent themselves write
    # approval_decision.json as their *output* once that happens, so nothing else
    # needs to write it here.
    overrides: dict[str, Any] = {"training_approver" if gate == "h4" else "approver": (
        approver.strip() or "web-ui"
    )}
    if gate == "h4":
        overrides["training_approval_action"] = action
    else:
        overrides["approval_action"] = action

    return _resume_workflow(_runs(request), workflow_id, overrides)


# ── Model registry ────────────────────────────────────────────────────────────


@router.get("/registry", response_class=HTMLResponse)
async def registry_list(request: Request):
    models = reader.list_models(_registry(request))
    return _render(request, "registry.html", {"models": models})


@router.get("/registry/{model_name}", response_class=HTMLResponse)
async def model_detail(request: Request, model_name: str):
    versions = reader.get_model_versions(_registry(request), model_name)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return _render(request, "model_detail.html", {
        "model_name": model_name,
        "versions": versions,
    })


# ── Monitoring ────────────────────────────────────────────────────────────────


@router.get("/monitoring", response_class=HTMLResponse)
async def monitoring_list(request: Request):
    reports = reader.list_monitoring_reports(_runs(request))
    return _render(request, "monitoring.html", {"reports": reports})


@router.get("/monitoring/{workflow_id}/{step}", response_class=HTMLResponse)
async def monitoring_detail_view(request: Request, workflow_id: str, step: str):
    report = reader.get_monitoring_report(_runs(request), workflow_id, step)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Monitoring report '{workflow_id}/{step}' not found",
        )
    return _render(request, "monitoring_detail.html", {
        "report": report,
        "workflow_id": workflow_id,
        "step": step,
    })


# ── Workflow resume ───────────────────────────────────────────────────────────


def _resume_workflow(
    runs_dir: Path, workflow_id: str, overrides: dict[str, Any]
) -> RedirectResponse:
    """Re-run the orchestrator for an already-started workflow_id.

    Reads the original run's persisted input.json, applies `overrides` (e.g. a
    just-submitted approval action), forces non-interactive mode (a web request
    has no TTY for HumanApprovalAgent's/TrainingApprovalAgent's input() prompt —
    without this a CLI-started run that persisted interactive=True would block
    the background executor thread forever), and re-runs in the background.
    """
    if not reader.is_safe_path_id(workflow_id):
        raise HTTPException(status_code=400, detail="Invalid workflow_id")
    input_file = runs_dir / workflow_id / "input.json"
    if not input_file.exists():
        return RedirectResponse(
            f"/workflows/{workflow_id}?flash=No+input.json+found+%E2%80%94+resume+from+CLI",
            status_code=303,
        )

    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return RedirectResponse(
            f"/workflows/{workflow_id}?flash=Cannot+read+input.json%3A+{exc}",
            status_code=303,
        )

    data["resume"] = True
    data["interactive_approval"] = False
    data["interactive_training_approval"] = False
    data.update(overrides)

    from agentic_mlops.contracts.orchestrator import OrchestratorInput  # noqa: PLC0415
    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow  # noqa: PLC0415

    try:
        inp = OrchestratorInput(**data)
    except Exception as exc:
        return RedirectResponse(
            f"/workflows/{workflow_id}?flash=Invalid+input%3A+{str(exc)[:80]}",
            status_code=303,
        )

    def _run():
        OrchestratorWorkflow().run(inp)

    asyncio.get_running_loop().run_in_executor(None, _run)
    return RedirectResponse(f"/workflows/{workflow_id}?flash=Workflow+resumed", status_code=303)


@router.post("/workflows/{workflow_id}/resume")
async def workflow_resume(request: Request, workflow_id: str):
    return _resume_workflow(_runs(request), workflow_id, overrides={})


# ── JSON API ──────────────────────────────────────────────────────────────────


@router.get("/api/workflows/{workflow_id}/status")
async def workflow_status_api(workflow_id: str, request: Request):
    state = reader.get_workflow(_runs(request), workflow_id)
    if state is None:
        raise HTTPException(status_code=404)
    return JSONResponse({
        "workflow_id": workflow_id,
        "status": state.get("status", "unknown"),
        "current_state": state.get("current_state", ""),
        "completed_steps": state.get("completed_steps", []),
        "step_status": state.get("step_status", {}),
        "last_agent": state.get("last_agent"),
        "updated_at": state.get("updated_at", ""),
        "pending_approval_id": state.get("pending_approval_id"),
    })


@router.get("/api/workflows/{workflow_id}/stream")
async def workflow_status_stream(workflow_id: str, request: Request):
    """SSE stream — pushes state updates while workflow is running."""
    runs_dir = _runs(request)

    async def generate():
        last_updated_at: str | None = None
        heartbeat = 0
        for _ in range(300):  # max ~10 min at 2 s intervals
            state = reader.get_workflow(runs_dir, workflow_id)
            if state is None:
                break
            updated_at = state.get("updated_at", "")
            if updated_at != last_updated_at:
                last_updated_at = updated_at
                payload = json.dumps({
                    "workflow_id": workflow_id,
                    "status": state.get("status", "unknown"),
                    "current_state": state.get("current_state", ""),
                    "completed_steps": state.get("completed_steps", []),
                    "step_status": state.get("step_status", {}),
                    "last_agent": state.get("last_agent"),
                    "updated_at": updated_at,
                })
                yield f"data: {payload}\n\n"
            if state.get("status", "unknown") != "running":
                break
            heartbeat += 1
            if heartbeat % 15 == 0:
                yield ": heartbeat\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Dataset registry ──────────────────────────────────────────────────────────


@router.get("/datasets", response_class=HTMLResponse)
async def datasets_list(request: Request):
    datasets = reader.list_datasets(_datasets(request))
    return _render(request, "datasets.html", {"datasets": datasets})


@router.get("/datasets/{dataset_name}", response_class=HTMLResponse)
async def dataset_detail_view(request: Request, dataset_name: str):
    versions = reader.get_dataset_versions(_datasets(request), dataset_name)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    return _render(
        request, "dataset_detail.html", {"dataset_name": dataset_name, "versions": versions}
    )


# ── Model comparison ──────────────────────────────────────────────────────────


def _parse_model_ref(ref: str) -> tuple[str, int | None]:
    """Parse 'modelname:version' → (name, version_num | None)."""
    if ":" in ref:
        name, ver_str = ref.rsplit(":", 1)
        try:
            return name.strip(), int(ver_str.strip())
        except ValueError:
            return ref.strip(), None
    return ref.strip(), None


def _resolve_model_ref(registry_dir: Path, ref: str) -> dict | None:
    """Return versioned model dict for 'name:v' ref, or latest if no version."""
    if not ref:
        return None
    name, ver = _parse_model_ref(ref)
    versions = reader.get_model_versions(registry_dir, name)
    if not versions:
        return None
    if ver is not None:
        found = next((v for v in versions if v["version"] == ver), None)
    else:
        found = versions[0]  # sorted desc → newest
    if found:
        found = dict(found)
        found["_name"] = name
    return found


@router.get("/registry/compare", response_class=HTMLResponse)
async def model_compare(request: Request, a: str = "", b: str = ""):
    reg = _registry(request)
    all_models = reader.list_models(reg)
    all_versions: dict[str, list[dict]] = {
        m["model_name"]: reader.get_model_versions(reg, m["model_name"])
        for m in all_models
    }

    model_a = _resolve_model_ref(reg, a) if a else None
    model_b = _resolve_model_ref(reg, b) if b else None

    metrics = ("map50", "map50_95", "precision", "recall")
    deltas: list[dict] = []
    if model_a and model_b:
        lin_a = model_a.get("lineage", {})
        lin_b = model_b.get("lineage", {})
        for m in metrics:
            va = lin_a.get(m)
            vb = lin_b.get(m)
            delta = (vb - va) if (va is not None and vb is not None) else None
            deltas.append({"metric": m, "a": va, "b": vb, "delta": delta})

    return _render(request, "compare.html", {
        "model_a": model_a,
        "model_b": model_b,
        "ref_a": a,
        "ref_b": b,
        "deltas": deltas,
        "all_models": all_models,
        "all_versions": all_versions,
        "metrics": metrics,
    })


# ── Prometheus metrics ────────────────────────────────────────────────────────


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(request: Request):
    """Prometheus text-format metrics endpoint for Grafana scraping."""
    runs_dir = _runs(request)
    registry_dir = _registry(request)
    datasets_dir = _datasets(request)

    workflows = reader.list_workflows(runs_dir)
    models = reader.list_models(registry_dir)
    datasets = reader.list_datasets(datasets_dir)

    status_counts: dict[str, int] = {}
    step_outcomes: dict[tuple[str, str], int] = {}
    for wf in workflows:
        s = wf.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
        for step, step_status in wf.get("step_status", {}).items():
            key = (step, str(step_status))
            step_outcomes[key] = step_outcomes.get(key, 0) + 1

    lines: list[str] = []

    def metric(help_text: str, type_: str, name: str) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {type_}")

    metric("Number of workflows by current status", "gauge", "mlops_workflows_total")
    for s in ("running", "completed", "failed", "blocked", "pending_approval"):
        lines.append(f'mlops_workflows_total{{status="{s}"}} {status_counts.get(s, 0)}')

    if step_outcomes:
        metric("Cumulative step execution outcomes across all workflows", "counter",
               "mlops_workflow_step_outcomes_total")
        for (step, s), count in sorted(step_outcomes.items()):
            lines.append(
                f'mlops_workflow_step_outcomes_total{{step="{step}",status="{s}"}} {count}'
            )

    metric("Number of registered model assets", "gauge", "mlops_models_registered_total")
    lines.append(f"mlops_models_registered_total {len(models)}")

    if models:
        metric("Latest mAP50 for each registered model (latest version)", "gauge",
               "mlops_model_map50")
        for m in models:
            name = (m.get("model_name") or "unknown").replace('"', '\\"')
            ver = m.get("version", "")
            map50 = m.get("map50")
            if map50 is not None:
                lines.append(f'mlops_model_map50{{model_name="{name}",version="{ver}"}} {map50}')

    metric("Number of registered dataset assets", "gauge", "mlops_datasets_registered_total")
    lines.append(f"mlops_datasets_registered_total {len(datasets)}")

    return "\n".join(lines) + "\n"
