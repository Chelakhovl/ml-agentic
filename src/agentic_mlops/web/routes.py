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


@router.get("/health")
async def health():
    """Lightweight liveness probe — returns 200 with no side effects."""
    return {"status": "ok"}


@router.get("/api/me")
async def me(request: Request):
    """Return the authenticated actor identity set by the auth middleware."""
    return {"actor": getattr(request.state, "actor", "anonymous")}


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
    ctx.setdefault("status_class", _status_class)
    ctx.setdefault("action_class", _action_class)
    ctx.setdefault("fmt_dt", _fmt_dt)
    ctx.setdefault("audit_icon", _audit_icon)
    ctx.setdefault("step_icon", lambda s: _STEP_ICONS.get(s, "⚙"))
    ctx.setdefault("flash", request.query_params.get("flash", ""))
    return _tmpl(request).TemplateResponse(request, template, ctx)


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


# ── Run comparison ───────────────────────────────────────────────────────────


@router.get("/workflows/compare", response_class=HTMLResponse)
async def workflow_compare(request: Request, a: str = "", b: str = ""):
    workflows = reader.list_workflows(_runs(request))
    diff: dict | None = None
    error: str | None = None
    if a and b:
        diff = reader.get_run_diff(_runs(request), a, b)
        if diff is None:
            error = f"One or both workflow IDs not found: '{a}', '{b}'"
    return _render(
        request,
        "workflow_compare.html",
        {
            "workflows": workflows,
            "ref_a": a,
            "ref_b": b,
            "diff": diff,
            "error": error,
        },
    )


# ── Cost summary ──────────────────────────────────────────────────────────────


@router.get("/costs", response_class=HTMLResponse)
async def costs_summary(request: Request):
    reports = reader.list_cost_reports(_runs(request))
    grand_total = sum(r.get("total_cost_usd", 0.0) for r in reports)
    top = sorted(reports, key=lambda r: r.get("total_cost_usd", 0.0), reverse=True)
    return _render(
        request,
        "costs.html",
        {
            "reports": reports,
            "grand_total": grand_total,
            "top_run": top[0] if top else None,
        },
    )


# ── Prune runs ────────────────────────────────────────────────────────────────


@router.get("/workflows/prune", response_class=HTMLResponse)
async def prune_preview(
    request: Request,
    keep_last: int | None = None,
    older_than_days: int | None = None,
):
    from agentic_mlops.tools.run_pruner import WorkflowPruner  # noqa: PLC0415

    preview: dict | None = None
    error: str | None = None
    if keep_last is not None or older_than_days is not None:
        try:
            pruner = WorkflowPruner(
                keep_last=keep_last,
                older_than_days=older_than_days,
                dry_run=True,
            )
            result = pruner.prune(_runs(request))
            preview = {
                "to_delete": result.deleted,
                "to_keep": result.kept,
                "skipped": result.skipped,
                "keep_last": keep_last,
                "older_than_days": older_than_days,
            }
        except ValueError as exc:
            error = str(exc)
    return _render(
        request,
        "prune.html",
        {
            "preview": preview,
            "error": error,
            "keep_last": keep_last,
            "older_than_days": older_than_days,
        },
    )


@router.post("/workflows/prune")
async def prune_execute(
    request: Request,
    keep_last: Annotated[int | None, Form()] = None,
    older_than_days: Annotated[int | None, Form()] = None,
):
    from agentic_mlops.tools.run_pruner import WorkflowPruner  # noqa: PLC0415

    try:
        pruner = WorkflowPruner(keep_last=keep_last, older_than_days=older_than_days)
        result = pruner.prune(_runs(request))
    except ValueError as exc:
        return RedirectResponse(f"/?flash={str(exc)[:80]}", status_code=303)
    msg = f"Pruned:+deleted+{result.deleted_count}+run(s),+kept+{len(result.kept)}"
    return RedirectResponse(f"/?flash={msg}", status_code=303)


# ── New workflow ──────────────────────────────────────────────────────────────


@router.get("/workflows/new", response_class=HTMLResponse)
async def new_workflow_form(request: Request):
    default_id = f"wf_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
    return _render(
        request,
        "new_workflow.html",
        {
            "pipeline_steps": PIPELINE_STEPS,
            "default_steps": DEFAULT_STEPS,
            "default_id": default_id,
        },
    )


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
        return _render(
            request,
            "new_workflow.html",
            {
                "pipeline_steps": PIPELINE_STEPS,
                "default_steps": DEFAULT_STEPS,
                "default_id": workflow_id,
                "error": "workflow_id is required and may not contain '/', '\\', '.' or '..'",
            },
        )

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
        return _render(
            request,
            "new_workflow.html",
            {
                "pipeline_steps": PIPELINE_STEPS,
                "default_steps": chosen_steps,
                "default_id": workflow_id,
                "cli_command": cli_cmd,
                "info": (
                    "Non-dry-run workflows must be started from the CLI to support "
                    "long-running training jobs."
                ),
            },
        )

    # dry_run=True: run in background thread
    try:
        inp = OrchestratorInput(**inp_kwargs)
    except Exception as exc:
        return _render(
            request,
            "new_workflow.html",
            {
                "pipeline_steps": PIPELINE_STEPS,
                "default_steps": chosen_steps,
                "default_id": workflow_id,
                "error": f"Invalid input: {exc}",
            },
        )

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
    baseline_comparison = reader.get_baseline_comparison(_runs(request), workflow_id)
    risk_report = reader.get_risk_report(_runs(request), workflow_id)
    cost_report = reader.get_cost_report(_runs(request), workflow_id)
    return _render(
        request,
        "workflow_detail.html",
        {
            "state": state,
            "audit_log": audit_log,
            "pipeline_steps": PIPELINE_STEPS,
            "baseline_comparison": baseline_comparison,
            "risk_report": risk_report,
            "cost_report": cost_report,
        },
    )


@router.post("/workflows/{workflow_id}/tags")
async def workflow_tags_update(
    request: Request,
    workflow_id: str,
    tag_key: Annotated[str, Form()] = "",
    tag_value: Annotated[str, Form()] = "",
    remove_key: Annotated[str, Form()] = "",
):
    if not reader.is_safe_path_id(workflow_id):
        raise HTTPException(status_code=400, detail="Invalid workflow_id")
    from agentic_mlops.tools.run_tagger import WorkflowRunTagger

    tagger = WorkflowRunTagger()
    try:
        if remove_key.strip():
            tagger.remove(_runs(request), workflow_id, [remove_key.strip()])
        elif tag_key.strip():
            tagger.add(_runs(request), workflow_id, {tag_key.strip(): tag_value.strip()})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")  # noqa: B904
    return RedirectResponse(f"/workflows/{workflow_id}?flash=Tags+updated", status_code=303)


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
    return _render(
        request,
        "approve.html",
        {
            "state": state,
            "gate": gate,
            "actions": actions,
        },
    )


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
    # Prefer the explicit form field; fall back to the verified auth identity so
    # audit_log.jsonl always records who took the action (actor field).
    actor = getattr(request.state, "actor", "anonymous")
    effective_approver = approver.strip() or actor or "web-ui"
    overrides: dict[str, Any] = {
        "training_approver" if gate == "h4" else "approver": effective_approver
    }
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


@router.get("/registry/compare", response_class=HTMLResponse)
async def model_compare_early(request: Request, a: str = "", b: str = ""):
    reg = _registry(request)
    all_models = reader.list_models(reg)
    all_versions: dict[str, list[dict]] = {
        m["model_name"]: reader.get_model_versions(reg, m["model_name"]) for m in all_models
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
    return _render(
        request,
        "compare.html",
        {
            "model_a": model_a,
            "model_b": model_b,
            "ref_a": a,
            "ref_b": b,
            "deltas": deltas,
            "all_models": all_models,
            "all_versions": all_versions,
            "metrics": metrics,
        },
    )


@router.get("/registry/rank", response_class=HTMLResponse)
async def model_rank_early(request: Request, model_filter: str = ""):
    reg = _registry(request)
    all_versions = reader.list_all_model_versions(reg)
    model_names = sorted({v["model_name"] for v in all_versions})
    ranked = (
        all_versions
        if not model_filter
        else [v for v in all_versions if v["model_name"] == model_filter]
    )
    winner = ranked[0] if ranked and ranked[0].get("rank") == 1 else None
    return _render(
        request,
        "rank.html",
        {
            "ranked": ranked,
            "winner": winner,
            "model_names": model_names,
            "model_filter": model_filter,
            "metrics": ("map50", "map50_95", "precision", "recall"),
        },
    )


@router.get("/registry/{model_name}", response_class=HTMLResponse)
async def model_detail(request: Request, model_name: str):
    versions = reader.get_model_versions(_registry(request), model_name)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return _render(
        request,
        "model_detail.html",
        {
            "model_name": model_name,
            "versions": versions,
        },
    )


# ── Monitoring ────────────────────────────────────────────────────────────────


@router.get("/monitoring", response_class=HTMLResponse)
async def monitoring_list(request: Request):
    reports = reader.list_monitoring_reports(_runs(request))
    return _render(request, "monitoring.html", {"reports": reports})


@router.get("/monitoring/hard-samples", response_class=HTMLResponse)
async def hard_samples_aggregate(request: Request):
    manifests = reader.list_hard_samples(_runs(request))
    total_count = sum(m["count"] for m in manifests)
    top = max(manifests, key=lambda m: m["count"], default=None)
    return _render(
        request,
        "monitoring_hard_samples.html",
        {
            "manifests": manifests,
            "total_count": total_count,
            "top_workflow": top,
        },
    )


@router.get("/monitoring/{workflow_id}/{step}", response_class=HTMLResponse)
async def monitoring_detail_view(request: Request, workflow_id: str, step: str):
    report = reader.get_monitoring_report(_runs(request), workflow_id, step)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Monitoring report '{workflow_id}/{step}' not found",
        )
    return _render(
        request,
        "monitoring_detail.html",
        {
            "report": report,
            "workflow_id": workflow_id,
            "step": step,
        },
    )


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


# ── Slack interactive approval webhook ────────────────────────────────────────


@router.post("/webhooks/approve/slack")
async def slack_approval_webhook(request: Request):
    """Receive Slack interactive button callbacks for H4/H5 approval gates.

    Requires:
    - Slack app with "Interactivity" enabled.
    - "Interactivity Request URL" set to ``{callback_base_url}/webhooks/approve/slack``.
    - ``SLACK_SIGNING_SECRET`` env var or ``--slack-signing-secret`` flag on
      ``agentic-mlops serve`` (same value as in your Slack app's Basic Information).

    Button values are encoded as ``<workflow_id>:<action>`` by
    ``_format_slack_approval()`` in ``integrations/notification_client.py``.
    """
    import hashlib  # noqa: PLC0415
    import hmac  # noqa: PLC0415
    import time  # noqa: PLC0415
    import urllib.parse  # noqa: PLC0415

    signing_secret: str = getattr(request.app.state, "slack_signing_secret", "")
    body_bytes = await request.body()

    if signing_secret:
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        sig_header = request.headers.get("X-Slack-Signature", "")

        # Reject stale requests (replay protection — 5 minute window).
        try:
            if abs(time.time() - int(timestamp)) > 300:
                raise HTTPException(status_code=403, detail="Slack request timestamp too old")
        except ValueError:
            raise HTTPException(status_code=403, detail="Missing or invalid Slack timestamp")  # noqa: B904

        sig_basestring = f"v0:{timestamp}:{body_bytes.decode('utf-8')}"
        computed_sig = (
            "v0="
            + hmac.new(
                signing_secret.encode("utf-8"),
                sig_basestring.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        )
        if not hmac.compare_digest(computed_sig, sig_header):
            raise HTTPException(status_code=403, detail="Invalid Slack signature")

    # Slack sends block_actions as application/x-www-form-urlencoded
    # with a single "payload" field containing JSON.
    try:
        form_data = urllib.parse.parse_qs(body_bytes.decode("utf-8"))
        payload_json = form_data.get("payload", ["{}"])[0]
        slack_payload = json.loads(payload_json)
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
        raise HTTPException(status_code=400, detail="Malformed Slack payload")  # noqa: B904

    if slack_payload.get("type") != "block_actions":
        return JSONResponse({"ok": True})

    actions = slack_payload.get("actions", [])
    if not actions:
        return JSONResponse({"ok": True})

    # Button value is encoded as "<workflow_id>:<action>"
    action = actions[0]
    raw_value = action.get("value", "")
    if ":" not in raw_value:
        raise HTTPException(status_code=400, detail="Unexpected action value format")

    workflow_id, _, action_name = raw_value.partition(":")
    workflow_id = workflow_id.strip()
    action_name = action_name.strip()

    if not reader.is_safe_path_id(workflow_id):
        raise HTTPException(status_code=400, detail="Invalid workflow_id in action value")

    state = reader.get_workflow(_runs(request), workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    gate = _detect_gate(state)
    allowed = H4_ACTIONS if gate == "h4" else H5_ACTIONS
    if action_name not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Action '{action_name}' not valid for gate '{gate or 'none'}'",
        )

    slack_user = (
        slack_payload.get("user", {}).get("name")
        or slack_payload.get("user", {}).get("id")
        or "slack-user"
    )
    overrides: dict[str, Any] = {
        "training_approver" if gate == "h4" else "approver": slack_user,
    }
    if gate == "h4":
        overrides["training_approval_action"] = action_name
    else:
        overrides["approval_action"] = action_name

    # Run the resume in the background — Slack expects a 200 within 3 seconds.
    runs_dir = _runs(request)
    input_file = runs_dir / workflow_id / "input.json"
    if not input_file.exists():
        return JSONResponse({"ok": True, "warning": "input.json not found — resume from CLI"})

    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": True, "warning": "Could not read input.json"})

    data["resume"] = True
    data["interactive_approval"] = False
    data["interactive_training_approval"] = False
    data.update(overrides)

    from agentic_mlops.contracts.orchestrator import OrchestratorInput  # noqa: PLC0415
    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow  # noqa: PLC0415

    try:
        inp = OrchestratorInput(**data)
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": True, "warning": "Invalid input — resume from CLI"})

    def _run() -> None:
        OrchestratorWorkflow().run(inp)

    asyncio.get_running_loop().run_in_executor(None, _run)
    return JSONResponse({"ok": True, "action": action_name, "workflow_id": workflow_id})


# ── Teams interactive approval webhook ───────────────────────────────────────


@router.post("/webhooks/approve/teams")
async def teams_approval_webhook(request: Request):
    """Receive Teams connector HttpPOST callbacks for H4/H5 approval gates.

    Requires:
    - A Teams connector configured with a security token.
    - The connector's "Action URL" set to ``{callback_base_url}/webhooks/approve/teams``.
    - ``TEAMS_SIGNING_SECRET`` env var or ``--teams-signing-secret`` flag on
      ``agentic-mlops serve`` (the base64-encoded security token from the connector).

    The request body is the JSON we embedded in each HttpPOST action's ``body`` field:
    ``{"workflow_id": "<id>", "action": "<action_name>"}``.

    Teams verifies requests by attaching ``Authorization: HMAC <base64-hmac>`` where
    the HMAC is SHA-256 over the UTF-8 request body using the base64-decoded token key.
    """
    import base64  # noqa: PLC0415
    import hashlib  # noqa: PLC0415
    import hmac  # noqa: PLC0415

    signing_secret: str = getattr(request.app.state, "teams_signing_secret", "")
    body_bytes = await request.body()

    if signing_secret:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.upper().startswith("HMAC "):
            raise HTTPException(status_code=403, detail="Missing Teams HMAC authorization")
        provided_hmac = auth_header[5:].strip()
        try:
            key_bytes = base64.b64decode(signing_secret)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=500, detail="Invalid Teams signing secret encoding")  # noqa: B904
        computed_hmac = base64.b64encode(
            hmac.new(key_bytes, body_bytes, hashlib.sha256).digest()
        ).decode("ascii")
        if not hmac.compare_digest(computed_hmac, provided_hmac):
            raise HTTPException(status_code=403, detail="Invalid Teams HMAC signature")

    try:
        teams_payload = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Malformed Teams payload")  # noqa: B904

    workflow_id = str(teams_payload.get("workflow_id", "")).strip()
    action_name = str(teams_payload.get("action", "")).strip()

    if not workflow_id or not action_name:
        raise HTTPException(
            status_code=400, detail="Missing workflow_id or action in Teams payload"
        )
    if not reader.is_safe_path_id(workflow_id):
        raise HTTPException(status_code=400, detail="Invalid workflow_id")

    state = reader.get_workflow(_runs(request), workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    gate = _detect_gate(state)
    allowed = H4_ACTIONS if gate == "h4" else H5_ACTIONS
    if action_name not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Action '{action_name}' not valid for gate '{gate or 'none'}'",
        )

    overrides: dict[str, Any] = {
        "training_approver" if gate == "h4" else "approver": "teams-user",
    }
    if gate == "h4":
        overrides["training_approval_action"] = action_name
    else:
        overrides["approval_action"] = action_name

    runs_dir = _runs(request)
    input_file = runs_dir / workflow_id / "input.json"
    if not input_file.exists():
        return JSONResponse({"type": "message", "text": "input.json not found — resume from CLI"})

    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"type": "message", "text": "Could not read input.json"})

    data["resume"] = True
    data["interactive_approval"] = False
    data["interactive_training_approval"] = False
    data.update(overrides)

    from agentic_mlops.contracts.orchestrator import OrchestratorInput  # noqa: PLC0415
    from agentic_mlops.workflows.orchestrator import OrchestratorWorkflow  # noqa: PLC0415

    try:
        inp = OrchestratorInput(**data)
    except Exception:  # noqa: BLE001
        return JSONResponse({"type": "message", "text": "Invalid input — resume from CLI"})

    def _run() -> None:
        OrchestratorWorkflow().run(inp)

    asyncio.get_running_loop().run_in_executor(None, _run)
    return JSONResponse(
        {
            "type": "message",
            "text": f"Action '{action_name}' accepted for workflow '{workflow_id}'",
        }
    )


# ── JSON API ──────────────────────────────────────────────────────────────────


@router.get("/api/dashboard/stats")
async def dashboard_stats_api(request: Request):
    """Return live dashboard stats — used for periodic JS polling."""
    workflows = reader.list_workflows(_runs(request))
    by_status: dict[str, int] = {}
    for wf in workflows:
        s = wf.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
    return JSONResponse(
        {
            "total": len(workflows),
            "running": by_status.get("running", 0),
            "pending": by_status.get("pending_approval", 0),
            "failed": by_status.get("failed", 0) + by_status.get("blocked", 0),
            "completed": by_status.get("completed", 0),
            "workflow_ids": [wf.get("workflow_id") for wf in workflows],
        }
    )


@router.get("/api/workflows/{workflow_id}/audit")
async def workflow_audit_api(request: Request, workflow_id: str, since: int = 0):
    """Return audit log entries. `since` is the count the caller already has."""
    entries = reader.get_audit_log(_runs(request), workflow_id)
    new_entries = entries[: max(0, len(entries) - since)] if since > 0 else entries
    return JSONResponse({"entries": new_entries, "total": len(entries)})


@router.get("/api/workflows/{workflow_id}/export")
async def workflow_export(workflow_id: str, request: Request):
    """Download workflow state.json as a JSON attachment."""
    state = reader.get_workflow(_runs(request), workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return JSONResponse(
        state,
        headers={"Content-Disposition": f'attachment; filename="{workflow_id}.json"'},
    )


@router.get("/api/workflows/{workflow_id}/status")
async def workflow_status_api(workflow_id: str, request: Request):
    state = reader.get_workflow(_runs(request), workflow_id)
    if state is None:
        raise HTTPException(status_code=404)
    return JSONResponse(
        {
            "workflow_id": workflow_id,
            "status": state.get("status", "unknown"),
            "current_state": state.get("current_state", ""),
            "completed_steps": state.get("completed_steps", []),
            "step_status": state.get("step_status", {}),
            "last_agent": state.get("last_agent"),
            "updated_at": state.get("updated_at", ""),
            "pending_approval_id": state.get("pending_approval_id"),
        }
    )


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
                payload = json.dumps(
                    {
                        "workflow_id": workflow_id,
                        "status": state.get("status", "unknown"),
                        "current_state": state.get("current_state", ""),
                        "completed_steps": state.get("completed_steps", []),
                        "step_status": state.get("step_status", {}),
                        "last_agent": state.get("last_agent"),
                        "updated_at": updated_at,
                    }
                )
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


@router.get("/datasets/{dataset_name}/quality", response_class=HTMLResponse)
async def dataset_quality_view(request: Request, dataset_name: str):
    quality_data = reader.get_dataset_quality_data(_datasets(request), dataset_name)
    if not quality_data:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    return _render(
        request,
        "dataset_quality.html",
        {"dataset_name": dataset_name, "versions": quality_data},
    )


@router.get("/datasets/{dataset_name}/diff", response_class=HTMLResponse)
async def dataset_diff_view(request: Request, dataset_name: str, v1: int = 0, v2: int = 0):
    if not reader.is_safe_path_id(dataset_name):
        raise HTTPException(status_code=400, detail="Invalid dataset name")

    all_versions = reader.get_dataset_versions(_datasets(request), dataset_name)
    if not all_versions:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

    version_nums = sorted([v["version"] for v in all_versions])

    # Default: compare the two most recent versions
    if v1 == 0 or v2 == 0:
        if len(version_nums) >= 2:
            v1, v2 = version_nums[-2], version_nums[-1]
        elif version_nums:
            v1 = v2 = version_nums[-1]

    diff = reader.get_dataset_diff(_datasets(request), dataset_name, v1, v2)
    if diff is None:
        raise HTTPException(
            status_code=404,
            detail=f"Version v{v1} or v{v2} of dataset '{dataset_name}' not found",
        )
    return _render(
        request,
        "dataset_diff.html",
        {
            "dataset_name": dataset_name,
            "diff": diff,
            "version_nums": version_nums,
            "v1": v1,
            "v2": v2,
        },
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
        m["model_name"]: reader.get_model_versions(reg, m["model_name"]) for m in all_models
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

    return _render(
        request,
        "compare.html",
        {
            "model_a": model_a,
            "model_b": model_b,
            "ref_a": a,
            "ref_b": b,
            "deltas": deltas,
            "all_models": all_models,
            "all_versions": all_versions,
            "metrics": metrics,
        },
    )


# ── Model ranking ─────────────────────────────────────────────────────────────


@router.get("/registry/rank", response_class=HTMLResponse)
async def model_rank(request: Request, model_filter: str = ""):
    reg = _registry(request)
    all_versions = reader.list_all_model_versions(reg)
    model_names = sorted({v["model_name"] for v in all_versions})
    ranked = (
        all_versions
        if not model_filter
        else [v for v in all_versions if v["model_name"] == model_filter]
    )
    winner = ranked[0] if ranked and ranked[0].get("rank") == 1 else None
    return _render(
        request,
        "rank.html",
        {
            "ranked": ranked,
            "winner": winner,
            "model_names": model_names,
            "model_filter": model_filter,
            "metrics": ("map50", "map50_95", "precision", "recall"),
        },
    )


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
        metric(
            "Cumulative step execution outcomes across all workflows",
            "counter",
            "mlops_workflow_step_outcomes_total",
        )
        for (step, s), count in sorted(step_outcomes.items()):
            lines.append(
                f'mlops_workflow_step_outcomes_total{{step="{step}",status="{s}"}} {count}'
            )

    metric("Number of registered model assets", "gauge", "mlops_models_registered_total")
    lines.append(f"mlops_models_registered_total {len(models)}")

    if models:
        metric(
            "Latest mAP50 for each registered model (latest version)", "gauge", "mlops_model_map50"
        )
        for m in models:
            name = (m.get("model_name") or "unknown").replace('"', '\\"')
            ver = m.get("version", "")
            map50 = m.get("map50")
            if map50 is not None:
                lines.append(f'mlops_model_map50{{model_name="{name}",version="{ver}"}} {map50}')

    metric("Number of registered dataset assets", "gauge", "mlops_datasets_registered_total")
    lines.append(f"mlops_datasets_registered_total {len(datasets)}")

    return "\n".join(lines) + "\n"
