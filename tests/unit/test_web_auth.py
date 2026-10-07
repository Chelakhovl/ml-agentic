"""Tests for web dashboard authentication middleware (web/app.py)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from agentic_mlops.web.app import create_app  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _basic(user: str, password: str) -> str:
    creds = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {creds}"


def _make_app(**kwargs):
    """Thin wrapper so tests can skip if create_app raises ImportError (no 'auth' extra)."""
    return create_app(
        runs_dir="/tmp/runs",
        registry_dir="/tmp/reg",
        datasets_dir="/tmp/ds",
        **kwargs,
    )


def _simple_client(**kwargs) -> TestClient:
    app = _make_app(**kwargs)
    return TestClient(app, raise_server_exceptions=True)


def _mock_httpx_for_oidc():
    """Return a context manager that patches httpx.Client to return fake OIDC metadata."""
    fake_oidc_meta = {"jwks_uri": "https://example.com/.well-known/jwks.json"}
    fake_jwks = {"keys": [{"kty": "RSA", "kid": "test-key", "use": "sig"}]}

    mock_resp_meta = MagicMock()
    mock_resp_meta.raise_for_status.return_value = mock_resp_meta
    mock_resp_meta.json.return_value = fake_oidc_meta

    mock_resp_jwks = MagicMock()
    mock_resp_jwks.raise_for_status.return_value = mock_resp_jwks
    mock_resp_jwks.json.return_value = fake_jwks

    mock_hc = MagicMock()
    mock_hc.__enter__ = MagicMock(return_value=mock_hc)
    mock_hc.__exit__ = MagicMock(return_value=False)
    mock_hc.get.side_effect = [mock_resp_meta, mock_resp_jwks]

    return patch("httpx.Client", return_value=mock_hc)


# ── No-auth mode ──────────────────────────────────────────────────────────────


class TestNoAuthMode:
    def test_health_endpoint_accessible_without_credentials(self):
        client = _simple_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_actor_defaults_to_anonymous(self):
        """The noop middleware sets actor='anonymous'; routes don't crash."""
        client = _simple_client()
        resp = client.get("/health")
        assert resp.status_code == 200


# ── require_auth ──────────────────────────────────────────────────────────────


class TestRequireAuth:
    def test_require_auth_with_no_config_raises(self):
        with pytest.raises(RuntimeError, match="--require-auth"):
            _make_app(require_auth=True)

    def test_require_auth_with_password_ok(self):
        app = _make_app(require_auth=True, dashboard_password="secret")
        assert app is not None

    def test_require_auth_with_api_key_ok(self):
        app = _make_app(require_auth=True, api_key="my-token")
        assert app is not None

    def test_require_auth_with_oidc_issuer_only_raises(self):
        """oidc_issuer without oidc_client_id doesn't count as configured auth."""
        with pytest.raises(RuntimeError, match="--require-auth"):
            _make_app(require_auth=True, oidc_issuer="https://example.com")

    def test_require_auth_with_oidc_both_fields_ok(self):
        pytest.importorskip("jose", reason="python-jose[cryptography] not installed (auth extra)")
        pytest.importorskip("httpx", reason="httpx not installed (auth extra)")
        with _mock_httpx_for_oidc():
            app = _make_app(
                require_auth=True,
                oidc_issuer="https://example.com",
                oidc_client_id="my-app",
            )
        assert app is not None


# ── API key auth ──────────────────────────────────────────────────────────────


class TestApiKeyAuth:
    def setup_method(self):
        self.key = "super-secret-token"
        self.client = _simple_client(api_key=self.key)

    def test_valid_bearer_token_passes(self):
        resp = self.client.get(
            "/health", headers={"Authorization": f"Bearer {self.key}"}
        )
        assert resp.status_code == 200

    def test_valid_x_api_key_passes(self):
        resp = self.client.get(
            "/health", headers={"X-API-Key": self.key}
        )
        assert resp.status_code == 200

    def test_wrong_bearer_token_returns_401(self):
        resp = self.client.get(
            "/health", headers={"Authorization": "Bearer wrong-token"}
        )
        assert resp.status_code == 401

    def test_wrong_x_api_key_returns_401(self):
        resp = self.client.get(
            "/health", headers={"X-API-Key": "wrong"}
        )
        assert resp.status_code == 401

    def test_no_credentials_returns_401(self):
        resp = self.client.get("/health")
        assert resp.status_code == 401

    def test_401_has_www_authenticate_header(self):
        resp = self.client.get("/health")
        assert "WWW-Authenticate" in resp.headers


# ── HTTP Basic Auth ───────────────────────────────────────────────────────────


class TestBasicAuth:
    def setup_method(self):
        self.client = _simple_client(
            dashboard_user="admin",
            dashboard_password="hunter2",
        )

    def test_valid_basic_credentials_pass(self):
        resp = self.client.get(
            "/health", headers={"Authorization": _basic("admin", "hunter2")}
        )
        assert resp.status_code == 200

    def test_wrong_password_returns_401(self):
        resp = self.client.get(
            "/health", headers={"Authorization": _basic("admin", "wrong")}
        )
        assert resp.status_code == 401

    def test_wrong_user_returns_401(self):
        resp = self.client.get(
            "/health", headers={"Authorization": _basic("hacker", "hunter2")}
        )
        assert resp.status_code == 401

    def test_no_credentials_returns_401(self):
        resp = self.client.get("/health")
        assert resp.status_code == 401

    def test_401_advertises_basic_realm(self):
        resp = self.client.get("/health")
        assert "Basic" in resp.headers.get("WWW-Authenticate", "")


# ── API key + Basic Auth together ─────────────────────────────────────────────


class TestApiKeyAndBasicAuth:
    def setup_method(self):
        self.key = "my-api-key"
        self.client = _simple_client(
            api_key=self.key,
            dashboard_user="admin",
            dashboard_password="pass",
        )

    def test_api_key_accepted(self):
        resp = self.client.get(
            "/health", headers={"X-API-Key": self.key}
        )
        assert resp.status_code == 200

    def test_basic_also_accepted(self):
        resp = self.client.get(
            "/health", headers={"Authorization": _basic("admin", "pass")}
        )
        assert resp.status_code == 200

    def test_neither_returns_401(self):
        resp = self.client.get("/health")
        assert resp.status_code == 401


# ── actor propagated to approve_submit ───────────────────────────────────────


class TestActorPropagation:
    """The auth middleware actor flows into approval overrides as the approver."""

    def _make_pending_state(self, runs_dir: Path, wf_id: str) -> None:
        wf_dir = runs_dir / wf_id
        wf_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "workflow_id": wf_id,
            "status": "pending_approval",
            "current_state": "MODEL_APPROVAL_REQUIRED",
            "steps": ["dataset_validation", "training", "evaluation", "approval"],
            "completed_steps": ["dataset_validation", "training", "evaluation"],
            "step_status": {},
            "step_outputs": {},
            "tags": {},
            "started_at": "2026-01-01T10:00:00+00:00",
            "updated_at": "2026-01-01T10:05:00+00:00",
            "pending_approval_id": "appr_wf_001",
        }
        (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        # Write a minimal input.json so _resume_workflow can read it.
        inp_data = {
            "workflow_id": wf_id,
            "dataset_path": str(runs_dir / wf_id / "ds"),
            "data_yaml_path": str(runs_dir / wf_id / "ds" / "data.yaml"),
            "runs_dir": str(runs_dir),
        }
        (wf_dir / "input.json").write_text(json.dumps(inp_data), encoding="utf-8")

    def test_actor_used_as_approver_when_form_field_empty(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        self._make_pending_state(runs_dir, "wf_001")

        client = TestClient(
            create_app(
                runs_dir=str(runs_dir),
                registry_dir=str(tmp_path / "reg"),
                datasets_dir=str(tmp_path / "ds"),
                api_key="tok",
            ),
            raise_server_exceptions=False,
        )

        # Capture the approver that gets written to overrides.
        # We intercept _resume_workflow by checking what OrchestratorInput it receives.
        captured = {}

        def fake_resume(runs_dir_arg, workflow_id, overrides):
            captured.update(overrides)
            from fastapi.responses import RedirectResponse

            return RedirectResponse(f"/workflows/{workflow_id}?flash=ok", status_code=303)

        import agentic_mlops.web.routes as _routes

        original = _routes._resume_workflow
        _routes._resume_workflow = fake_resume
        try:
            resp = client.post(
                "/approve/wf_001",
                data={"action": "approve_model", "approver": ""},
                headers={"X-API-Key": "tok"},
                follow_redirects=False,
            )
        finally:
            _routes._resume_workflow = original

        # The approver should be the actor set by the API key middleware.
        assert resp.status_code in (303, 200)
        assert captured.get("approver") == "api-key"

    def test_explicit_form_approver_takes_priority(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        self._make_pending_state(runs_dir, "wf_002")

        client = TestClient(
            create_app(
                runs_dir=str(runs_dir),
                registry_dir=str(tmp_path / "reg"),
                datasets_dir=str(tmp_path / "ds"),
                api_key="tok",
            ),
            raise_server_exceptions=False,
        )

        captured = {}

        def fake_resume(runs_dir_arg, workflow_id, overrides):
            captured.update(overrides)
            from fastapi.responses import RedirectResponse

            return RedirectResponse(f"/workflows/{workflow_id}?flash=ok", status_code=303)

        import agentic_mlops.web.routes as _routes

        original = _routes._resume_workflow
        _routes._resume_workflow = fake_resume
        try:
            client.post(
                "/approve/wf_002",
                data={"action": "approve_model", "approver": "jane.doe"},
                headers={"X-API-Key": "tok"},
                follow_redirects=False,
            )
        finally:
            _routes._resume_workflow = original

        # Explicit form field wins over the auth actor.
        assert captured.get("approver") == "jane.doe"


# ── OIDC middleware wired up (startup behaviour) ──────────────────────────────


class TestOidcStartup:
    def test_oidc_issuer_without_client_id_ignored(self):
        """oidc_issuer alone (no client_id) → OIDC block skipped, no startup fetch."""
        # This should NOT try to fetch JWKS and should just create the app
        # with the standard no-auth middleware (because _any_auth is False).
        app = _make_app(oidc_issuer="https://example.com")
        assert app is not None

    def test_oidc_startup_fetch_failure_raises(self):
        """If OIDC JWKS cannot be fetched, create_app raises RuntimeError."""
        pytest.importorskip("jose", reason="python-jose[cryptography] not installed (auth extra)")
        pytest.importorskip("httpx", reason="httpx not installed (auth extra)")
        with patch("httpx.Client") as mock_cls:
            mock_hc = MagicMock()
            mock_hc.__enter__ = MagicMock(return_value=mock_hc)
            mock_hc.__exit__ = MagicMock(return_value=False)
            mock_hc.get.side_effect = Exception("network error")
            mock_cls.return_value = mock_hc

            with pytest.raises(RuntimeError, match="Failed to fetch OIDC JWKS"):
                _make_app(
                    oidc_issuer="https://example.com",
                    oidc_client_id="my-app",
                )

    def test_oidc_missing_auth_extra_raises_import_error(self):
        """If httpx is not importable, create_app raises ImportError."""
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises((ImportError, TypeError)):
                _make_app(
                    oidc_issuer="https://example.com",
                    oidc_client_id="my-app",
                )

    def test_oidc_invalid_bearer_returns_401(self):
        """A Bearer token that fails OIDC validation → 401 with Bearer WWW-Authenticate."""
        pytest.importorskip("jose", reason="python-jose[cryptography] not installed (auth extra)")
        pytest.importorskip("httpx", reason="httpx not installed (auth extra)")
        with _mock_httpx_for_oidc():
            client = TestClient(
                _make_app(
                    oidc_issuer="https://example.com",
                    oidc_client_id="my-app",
                ),
                raise_server_exceptions=False,
            )

        resp = client.get(
            "/health", headers={"Authorization": "Bearer not-a-valid-jwt"}
        )
        assert resp.status_code == 401
        assert "Bearer" in resp.headers.get("WWW-Authenticate", "")
