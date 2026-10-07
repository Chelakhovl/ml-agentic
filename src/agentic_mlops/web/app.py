"""FastAPI application factory for the MLOps web dashboard."""

from __future__ import annotations

import base64
import secrets
from pathlib import Path
from typing import Any

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
    api_key: str = "",
    oidc_issuer: str = "",
    oidc_client_id: str = "",
    require_auth: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        runs_dir: Path to the directory containing workflow run state files.
        registry_dir: Path to the local model registry root.
        datasets_dir: Path to the local dataset registry root.
        dashboard_user: HTTP Basic Auth username (default "admin").
        dashboard_password: HTTP Basic Auth password.
            When empty authentication is disabled — safe for local-only
            development.  Set via ``DASHBOARD_PASSWORD`` env var / ``--password``.
        api_key: Bearer / X-API-Key token.  When set, every request must supply
            it in ``Authorization: Bearer <token>`` or ``X-API-Key: <token>``.
            Set via ``DASHBOARD_API_KEY`` env var / ``--api-key``.
        oidc_issuer: OIDC issuer URL (e.g. ``https://login.microsoftonline.com/...``).
            When set together with ``oidc_client_id``, JWT tokens are validated
            on every request (requires the ``auth`` extra: ``python-jose`` +
            ``httpx``).  Set via ``DASHBOARD_OIDC_ISSUER`` env var.
        oidc_client_id: Expected ``aud`` claim in OIDC JWTs.  Set via
            ``DASHBOARD_OIDC_CLIENT_ID`` env var.
        require_auth: When ``True``, the app refuses to start unless at least one
            auth method (Basic password, API key, or OIDC) is configured.  Useful
            for preventing accidental unauthenticated production deploys.
    """
    _any_auth = bool(dashboard_password or api_key or (oidc_issuer and oidc_client_id))
    if require_auth and not _any_auth:
        raise RuntimeError(
            "--require-auth is set but no auth method is configured. "
            "Set at least one of: DASHBOARD_PASSWORD, DASHBOARD_API_KEY, "
            "or both DASHBOARD_OIDC_ISSUER + DASHBOARD_OIDC_CLIENT_ID."
        )

    # Pre-load OIDC JWKS if OIDC is configured.
    _oidc_keys: list[dict[str, Any]] = []
    if oidc_issuer and oidc_client_id:
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "OIDC authentication requires the 'auth' extra: "
                "pip install -e '.[auth]'"
            ) from exc
        try:
            from jose import jwt as _jose_jwt  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError(
                "OIDC authentication requires the 'auth' extra: "
                "pip install -e '.[auth]'"
            ) from exc
        # Fetch JWKS synchronously at startup (runs before first request).
        try:
            _oidc_config_url = oidc_issuer.rstrip("/") + "/.well-known/openid-configuration"
            with httpx.Client(timeout=10) as _hc:
                _oidc_meta = _hc.get(_oidc_config_url).raise_for_status().json()
                _jwks = _hc.get(_oidc_meta["jwks_uri"]).raise_for_status().json()
                _oidc_keys = _jwks.get("keys", [])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to fetch OIDC JWKS from {oidc_issuer}: {exc}"
            ) from exc

    app = FastAPI(
        title="MLOps Dashboard",
        description="Web UI for the agentic MLOps pipeline",
        version="1.0.0",
    )

    # ── Authentication middleware ──────────────────────────────────────────────
    # Priority: OIDC JWT > API key > Basic Auth.
    # Any single matching layer grants access and sets request.state.actor.
    # The first layer that *recognises* the credential (right format) but *rejects*
    # it (wrong token) short-circuits to 401 rather than falling through.

    if api_key or (oidc_issuer and oidc_client_id) or dashboard_password:
        _api_key = api_key
        _basic_user = dashboard_user
        _basic_pass = dashboard_password
        _oidc_issuer = oidc_issuer
        _oidc_client_id = oidc_client_id
        _jwks_keys = _oidc_keys

        @app.middleware("http")
        async def _auth_middleware(request: Request, call_next):  # type: ignore[misc]
            actor: str | None = None

            auth_header = request.headers.get("Authorization", "")
            x_api_key = request.headers.get("X-API-Key", "")

            # 1. OIDC JWT (Bearer token validated against JWKS).
            if _oidc_issuer and _oidc_client_id and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                try:
                    from jose import JWTError, jwt  # noqa: PLC0415
                    from jose.backends import RSAKey  # noqa: F401, PLC0415

                    # Try each key until one succeeds.
                    last_exc: Exception | None = None
                    for jwk in _jwks_keys:
                        try:
                            claims = jwt.decode(
                                token,
                                jwk,
                                algorithms=["RS256", "RS384", "RS512", "ES256"],
                                audience=_oidc_client_id,
                                issuer=_oidc_issuer,
                                options={"verify_exp": True},
                            )
                            actor = claims.get("email") or claims.get("sub") or "oidc-user"
                            break
                        except JWTError as exc:
                            last_exc = exc
                    if actor is None:
                        # Recognised as JWT attempt but failed — reject.
                        return Response(
                            content=f"OIDC token validation failed: {last_exc}",
                            status_code=401,
                            headers={"WWW-Authenticate": "Bearer"},
                        )
                except ImportError:
                    return Response(
                        content="OIDC is configured but 'auth' extra is not installed.",
                        status_code=500,
                    )

            # 2. API key (Bearer or X-API-Key header).
            if actor is None and _api_key:
                candidate: str | None = None
                if auth_header.startswith("Bearer ") and not (_oidc_issuer and _oidc_client_id):
                    candidate = auth_header[7:]
                elif x_api_key:
                    candidate = x_api_key
                if candidate is not None:
                    if secrets.compare_digest(candidate, _api_key):
                        actor = "api-key"
                    else:
                        return Response(
                            content="Invalid API key.",
                            status_code=401,
                            headers={"WWW-Authenticate": 'Bearer realm="MLOps Dashboard"'},
                        )

            # 3. HTTP Basic Auth.
            if actor is None and _basic_pass and auth_header.startswith("Basic "):
                try:
                    decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
                    req_user, _, req_pass = decoded.partition(":")
                    user_ok = secrets.compare_digest(req_user, _basic_user)
                    pass_ok = secrets.compare_digest(req_pass, _basic_pass)
                    if user_ok and pass_ok:
                        actor = req_user
                    else:
                        return Response(
                            content="Invalid credentials.",
                            status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="MLOps Dashboard"'},
                        )
                except Exception:  # noqa: BLE001
                    pass

            # 4. No matching auth layer — require credentials.
            if actor is None:
                # Determine which realm to advertise.
                if _basic_pass:
                    headers = {"WWW-Authenticate": 'Basic realm="MLOps Dashboard"'}
                else:
                    headers = {"WWW-Authenticate": 'Bearer realm="MLOps Dashboard"'}
                return Response(
                    content="Unauthorized — authentication required.",
                    status_code=401,
                    headers=headers,
                )

            request.state.actor = actor
            return await call_next(request)

    else:
        # No auth configured — set a default actor so routes never fail on AttributeError.
        @app.middleware("http")
        async def _noop_auth(request: Request, call_next):  # type: ignore[misc]
            request.state.actor = "anonymous"
            return await call_next(request)

    # ── App state ─────────────────────────────────────────────────────────────

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
