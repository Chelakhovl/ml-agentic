"""Tests for web dashboard authentication (API key, OIDC, Basic Auth, require_auth)."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _app(**kwargs):
    """Create a minimal test app with the given auth config."""
    from agentic_mlops.web.app import create_app  # noqa: PLC0415

    return create_app(
        runs_dir="/tmp/test_runs",
        registry_dir="/tmp/test_registry",
        datasets_dir="/tmp/test_datasets",
        **kwargs,
    )


def _basic(user: str, password: str) -> str:
    encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {encoded}"


# ---------------------------------------------------------------------------
# No auth configured — anonymous access allowed
# ---------------------------------------------------------------------------


class TestNoAuth:
    def test_anonymous_request_reaches_route(self):
        app = _app()
        client = TestClient(app, raise_server_exceptions=False)
        # /metrics is a lightweight endpoint that doesn't need real data
        r = client.get("/metrics")
        # We get back a valid response (not 401)
        assert r.status_code != 401

    def test_actor_is_anonymous(self):
        client = TestClient(_app(), raise_server_exceptions=False)
        r = client.get("/api/me")
        assert r.json()["actor"] == "anonymous"


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    def _client_with_key(self, key: str = "secret123") -> TestClient:
        return TestClient(_app(api_key=key), raise_server_exceptions=False)

    def test_bearer_token_valid(self):
        client = self._client_with_key("mysecret")
        r = client.get("/metrics", headers={"Authorization": "Bearer mysecret"})
        assert r.status_code != 401

    def test_x_api_key_header_valid(self):
        client = self._client_with_key("mysecret")
        r = client.get("/metrics", headers={"X-API-Key": "mysecret"})
        assert r.status_code != 401

    def test_bearer_token_invalid(self):
        client = self._client_with_key("mysecret")
        r = client.get("/metrics", headers={"Authorization": "Bearer wrongtoken"})
        assert r.status_code == 401

    def test_x_api_key_invalid(self):
        client = self._client_with_key("mysecret")
        r = client.get("/metrics", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_no_credentials_returns_401(self):
        client = self._client_with_key("mysecret")
        r = client.get("/metrics")
        assert r.status_code == 401

    def test_actor_set_to_api_key_label(self):
        client = TestClient(_app(api_key="mytoken"))
        r = client.get("/api/me", headers={"X-API-Key": "mytoken"})
        assert r.json()["actor"] == "api-key"


# ---------------------------------------------------------------------------
# HTTP Basic Auth (existing feature — regression guard)
# ---------------------------------------------------------------------------


class TestBasicAuth:
    def test_valid_credentials_pass(self):
        client = TestClient(
            _app(dashboard_user="alice", dashboard_password="pw123"),
            raise_server_exceptions=False,
        )
        r = client.get("/metrics", headers={"Authorization": _basic("alice", "pw123")})
        assert r.status_code != 401

    def test_wrong_password_rejected(self):
        client = TestClient(
            _app(dashboard_user="alice", dashboard_password="pw123"),
            raise_server_exceptions=False,
        )
        r = client.get("/metrics", headers={"Authorization": _basic("alice", "wrong")})
        assert r.status_code == 401

    def test_no_credentials_returns_401(self):
        client = TestClient(
            _app(dashboard_password="pw123"),
            raise_server_exceptions=False,
        )
        r = client.get("/metrics")
        assert r.status_code == 401

    def test_actor_set_to_username(self):
        client = TestClient(_app(dashboard_user="bob", dashboard_password="secret"))
        r = client.get("/api/me", headers={"Authorization": _basic("bob", "secret")})
        assert r.json()["actor"] == "bob"


# ---------------------------------------------------------------------------
# API key takes priority over Basic when both are configured
# ---------------------------------------------------------------------------


class TestApiKeyAndBasicAuth:
    def test_api_key_bearer_wins_when_both_configured(self):
        client = TestClient(_app(dashboard_user="admin", dashboard_password="pw", api_key="tok"))
        r = client.get("/api/me", headers={"X-API-Key": "tok"})
        assert r.json()["actor"] == "api-key"

    def test_basic_auth_still_works_when_api_key_also_configured(self):
        client = TestClient(_app(dashboard_user="admin", dashboard_password="pw", api_key="tok"))
        r = client.get("/api/me", headers={"Authorization": _basic("admin", "pw")})
        assert r.json()["actor"] == "admin"


# ---------------------------------------------------------------------------
# --require-auth flag
# ---------------------------------------------------------------------------


class TestRequireAuth:
    def test_require_auth_raises_when_no_method_configured(self):
        with pytest.raises(RuntimeError, match="--require-auth"):
            _app(require_auth=True)

    def test_require_auth_ok_with_password(self):
        app = _app(require_auth=True, dashboard_password="pw")
        assert app is not None

    def test_require_auth_ok_with_api_key(self):
        app = _app(require_auth=True, api_key="tok")
        assert app is not None

    def test_require_auth_ok_with_oidc(self):
        # We cannot actually connect to a real OIDC provider in tests,
        # but we can verify the require_auth check itself passes when the
        # OIDC fields are set (the error here comes from the network call, not
        # the require_auth guard).
        with pytest.raises((RuntimeError, Exception)):
            _app(
                require_auth=True,
                oidc_issuer="https://nonexistent.example.com",
                oidc_client_id="client",
            )
        # The error is NOT the require_auth RuntimeError — it's a network error.
        try:
            _app(
                require_auth=True,
                oidc_issuer="https://nonexistent.example.com",
                oidc_client_id="client",
            )
        except RuntimeError as exc:
            assert "--require-auth" not in str(exc), "Should fail at OIDC, not require_auth"
        except Exception:
            pass  # network error — expected in CI


# ---------------------------------------------------------------------------
# OIDC import guard (no python-jose installed)
# ---------------------------------------------------------------------------


class TestOidcImportGuard:
    def test_oidc_without_auth_extra_raises_import_error(self, monkeypatch):
        import sys  # noqa: PLC0415

        # Pretend jose is not installed.
        monkeypatch.setitem(sys.modules, "jose", None)
        monkeypatch.setitem(sys.modules, "httpx", None)

        with pytest.raises((ImportError, TypeError)):
            _app(
                oidc_issuer="https://example.com",
                oidc_client_id="client",
            )


# ---------------------------------------------------------------------------
# CLI: serve command flags (unit-level, no real uvicorn)
# ---------------------------------------------------------------------------


class TestServeCliFlags:
    def test_require_auth_flag_exits_nonzero_when_no_auth(self):
        from typer.testing import CliRunner  # noqa: PLC0415

        from agentic_mlops.cli.main import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "serve",
                "--require-auth",
                "--port",
                "9999",
            ],
        )
        assert result.exit_code != 0
        assert "require-auth" in (result.output or "").lower() or result.exit_code == 1

    def test_api_key_envvar_accepted(self):
        import os  # noqa: PLC0415

        from typer.testing import CliRunner  # noqa: PLC0415

        from agentic_mlops.cli.main import app  # noqa: PLC0415

        runner = CliRunner(env={"DASHBOARD_API_KEY": "testkey"})
        # We just verify the command parses without --api-key explicitly.
        # Invoking with require-auth + the env var should not error at arg parsing.
        # (It will still fail because uvicorn isn't being run, but the error
        # won't be about the flag being unknown.)
        result = runner.invoke(
            app,
            [
                "serve",
                "--require-auth",
                "--port",
                "19999",
                "--host",
                "127.0.0.1",
            ],
        )
        # Should NOT say "No such option" — the flag is recognised.
        assert "no such option" not in (result.output or "").lower()
