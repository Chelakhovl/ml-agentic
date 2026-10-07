"""Tests for web dashboard dark mode, mobile responsive layout, and SSE endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from agentic_mlops.web.app import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(runs_dir: Path, registry_dir: Path | None = None):
    return create_app(
        runs_dir=str(runs_dir),
        registry_dir=str(registry_dir or (runs_dir / "registry")),
        datasets_dir=str(runs_dir / "datasets"),
    )


def _write_state(runs_dir: Path, wf_id: str, status: str = "running") -> dict:
    wf_dir = runs_dir / wf_id
    wf_dir.mkdir(parents=True)
    state = {
        "workflow_id": wf_id,
        "status": status,
        "current_state": status.upper(),
        "completed_steps": [],
        "step_outputs": {},
        "steps": [],
        "updated_at": "2026-01-01T00:00:00",
    }
    (wf_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return state


# ---------------------------------------------------------------------------
# Dark mode CSS — media query present in base HTML
# ---------------------------------------------------------------------------


class TestDarkModeCSS:
    def test_media_query_in_base_template(self):
        tpl_path = (
            Path(__file__).parents[2]
            / "src/agentic_mlops/web/templates/base.html"
        )
        content = tpl_path.read_text(encoding="utf-8")
        assert "@media (prefers-color-scheme: dark)" in content

    def test_dark_theme_css_vars_defined(self):
        tpl_path = (
            Path(__file__).parents[2]
            / "src/agentic_mlops/web/templates/base.html"
        )
        content = tpl_path.read_text(encoding="utf-8")
        for var in ("--bg", "--surface", "--text", "--accent"):
            assert var in content

    def test_data_theme_dark_selector_present(self):
        tpl_path = (
            Path(__file__).parents[2]
            / "src/agentic_mlops/web/templates/base.html"
        )
        content = tpl_path.read_text(encoding="utf-8")
        assert '[data-theme="dark"]' in content

    def test_not_light_guard_in_media_query(self):
        tpl_path = (
            Path(__file__).parents[2]
            / "src/agentic_mlops/web/templates/base.html"
        )
        content = tpl_path.read_text(encoding="utf-8")
        # The media query must guard against an explicitly-set light theme
        assert ':root:not([data-theme="light"])' in content


# ---------------------------------------------------------------------------
# Mobile responsive layout — hamburger button present in base HTML
# ---------------------------------------------------------------------------


class TestMobileLayout:
    def test_viewport_meta_present(self):
        tpl_path = (
            Path(__file__).parents[2]
            / "src/agentic_mlops/web/templates/base.html"
        )
        content = tpl_path.read_text(encoding="utf-8")
        assert 'name="viewport"' in content
        assert "width=device-width" in content

    def test_hamburger_button_present(self):
        tpl_path = (
            Path(__file__).parents[2]
            / "src/agentic_mlops/web/templates/base.html"
        )
        content = tpl_path.read_text(encoding="utf-8")
        assert "hamburger" in content

    def test_sidebar_overlay_present(self):
        tpl_path = (
            Path(__file__).parents[2]
            / "src/agentic_mlops/web/templates/base.html"
        )
        content = tpl_path.read_text(encoding="utf-8")
        assert "sidebarOverlay" in content

    def test_mobile_media_query_hides_sidebar_default(self):
        tpl_path = (
            Path(__file__).parents[2]
            / "src/agentic_mlops/web/templates/base.html"
        )
        content = tpl_path.read_text(encoding="utf-8")
        assert "@media (max-width: 768px)" in content

    def test_toggle_sidebar_js_function_present(self):
        tpl_path = (
            Path(__file__).parents[2]
            / "src/agentic_mlops/web/templates/base.html"
        )
        content = tpl_path.read_text(encoding="utf-8")
        assert "toggleSidebar" in content
        assert "closeSidebar" in content


# ---------------------------------------------------------------------------
# SSE endpoint — basic HTTP behavior
# ---------------------------------------------------------------------------


class TestSSEEndpoint:
    """Tests for the SSE stream endpoint at /api/workflows/{id}/stream.

    These tests consume the stream for a *completed* workflow only — that
    ensures the generator terminates immediately and the test does not hang.
    A running workflow would block the generator loop for up to 10 minutes,
    so testing that path is left to manual / integration tests.
    """

    def _read_sse(self, client: TestClient, path: str) -> tuple[int, str, list[dict]]:
        """Consume a short SSE stream; return (status, content_type, events)."""
        with client.stream("GET", path) as resp:
            content_type = resp.headers.get("content-type", "")
            events = []
            for raw in resp.iter_lines():
                if raw.startswith("data:"):
                    try:
                        events.append(json.loads(raw[len("data:"):].strip()))
                    except json.JSONDecodeError:
                        pass
            return resp.status_code, content_type, events

    def test_sse_stream_returns_200(self, tmp_path: Path):
        _write_state(tmp_path, "wf_test", status="completed")
        app = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        status, _, _ = self._read_sse(client, "/api/workflows/wf_test/stream")
        assert status == 200

    def test_sse_content_type_is_event_stream(self, tmp_path: Path):
        _write_state(tmp_path, "wf_ct", status="completed")
        app = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        _, content_type, _ = self._read_sse(client, "/api/workflows/wf_ct/stream")
        assert "text/event-stream" in content_type

    def test_sse_first_event_has_status_field(self, tmp_path: Path):
        _write_state(tmp_path, "wf_json", status="completed")
        app = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        _, _, events = self._read_sse(client, "/api/workflows/wf_json/stream")
        assert len(events) >= 1
        assert "status" in events[0]

    def test_sse_stream_stops_after_completed(self, tmp_path: Path):
        _write_state(tmp_path, "wf_stop", status="completed")
        app = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        _, _, events = self._read_sse(client, "/api/workflows/wf_stop/stream")
        # completed state must break the loop — at least one event received
        assert len(events) >= 1

    def test_sse_missing_workflow_returns_200_and_terminates(self, tmp_path: Path):
        app = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        status, _, _ = self._read_sse(client, "/api/workflows/nonexistent/stream")
        # SSE protocol: always 200; generator must close on missing workflow
        assert status == 200
