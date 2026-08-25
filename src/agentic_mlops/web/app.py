"""FastAPI application factory for the MLOps web dashboard."""

from __future__ import annotations

import base64
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from .routes import router


def create_app(
    runs_dir: str = "runs",
    registry_dir: str = "outputs/model_registry",
    datasets_dir: str = "outputs/dataset_registry",
    dashboard_user: str = "admin",
    dashboard_password: str = "",
    slack_signing_secret: str = "",
    teams_signing_secret: str = "",
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        runs_dir: Path to the directory containing workflow run state files.
        registry_dir: Path to the local model registry root.
        datasets_dir: Path to the local dataset registry root.
        dashboard_user: HTTP Basic Auth username (default "admin").
        dashboard_password: HTTP Basic Auth password.
            When empty (the default) authentication is disabled — safe for
            local-only development.  Set via the ``DASHBOARD_PASSWORD`` env
            var or the ``--password`` CLI flag on ``agentic-mlops serve``.
    """
    app = FastAPI(
        title="MLOps Dashboard",
        description="Web UI for the agentic MLOps pipeline",
        version="1.0.0",
    )

    if dashboard_password:
        _user = dashboard_user
        _pass = dashboard_password

        @app.middleware("http")
        async def _basic_auth(request: Request, call_next):  # type: ignore[misc]
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Basic "):
                try:
                    decoded = base64.b64decode(auth[6:]).decode("utf-8")
                    req_user, _, req_pass = decoded.partition(":")
                    user_ok = secrets.compare_digest(req_user, _user)
                    pass_ok = secrets.compare_digest(req_pass, _pass)
                    if user_ok and pass_ok:
                        return await call_next(request)
                except Exception:  # noqa: BLE001
                    pass
            return Response(
                content="Unauthorized — set DASHBOARD_PASSWORD to configure credentials.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="MLOps Dashboard"'},
            )

    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    app.state.runs_dir = Path(runs_dir).resolve()
    app.state.registry_dir = Path(registry_dir).resolve()
    app.state.datasets_dir = Path(datasets_dir).resolve()
    app.state.templates = templates
    app.state.slack_signing_secret = slack_signing_secret or ""
    app.state.teams_signing_secret = teams_signing_secret or ""

    app.include_router(router)

    return app
