"""Tests for WebhookNotificationClient retry logic."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, call, patch

from agentic_mlops.contracts.notification import NotificationConfig
from agentic_mlops.integrations.notification_client import WebhookNotificationClient


def _make_config(url: str, max_retries: int = 2, retry_delay: float = 0.0) -> NotificationConfig:
    return NotificationConfig(
        teams_webhook_url=url,
        notify_on=["test_event"],
        webhook_max_retries=max_retries,
        webhook_retry_delay=retry_delay,
    )


def _fake_response(status: int) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestWebhookSuccess:
    def test_posts_on_matching_event(self) -> None:
        cfg = _make_config("http://example.com/hook")
        client = WebhookNotificationClient(cfg)
        resp = _fake_response(200)
        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            client.send("test_event", {"k": "v"})
        mock_open.assert_called_once()

    def test_no_post_on_unmatched_event(self) -> None:
        cfg = _make_config("http://example.com/hook")
        client = WebhookNotificationClient(cfg)
        with patch("urllib.request.urlopen") as mock_open:
            client.send("other_event", {})
        mock_open.assert_not_called()

    def test_200_returns_without_retry(self) -> None:
        cfg = _make_config("http://example.com/hook", max_retries=3)
        client = WebhookNotificationClient(cfg)
        resp = _fake_response(200)
        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            client.send("test_event", {})
        assert mock_open.call_count == 1


class TestWebhookRetryOn5xx:
    def test_retries_on_500(self) -> None:
        cfg = _make_config("http://example.com/hook", max_retries=2, retry_delay=0.0)
        client = WebhookNotificationClient(cfg)
        resp_500 = _fake_response(500)
        resp_200 = _fake_response(200)
        side = [resp_500, resp_500, resp_200]
        with patch("urllib.request.urlopen", side_effect=side) as mock_open:
            with patch("time.sleep"):
                client.send("test_event", {})
        assert mock_open.call_count == 3

    def test_retries_exhaust_all_attempts_on_5xx(self) -> None:
        cfg = _make_config("http://example.com/hook", max_retries=2, retry_delay=0.0)
        client = WebhookNotificationClient(cfg)
        resp_503 = _fake_response(503)
        with patch("urllib.request.urlopen", return_value=resp_503) as mock_open:
            with patch("time.sleep"):
                client.send("test_event", {})
        assert mock_open.call_count == 3  # 1 initial + 2 retries

    def test_sleeps_between_retries(self) -> None:
        cfg = _make_config("http://example.com/hook", max_retries=2, retry_delay=1.0)
        client = WebhookNotificationClient(cfg)
        resp_500 = _fake_response(500)
        resp_200 = _fake_response(200)
        with patch("urllib.request.urlopen", side_effect=[resp_500, resp_200]):
            with patch("time.sleep") as mock_sleep:
                client.send("test_event", {})
        mock_sleep.assert_called_once_with(1.0)

    def test_exponential_backoff(self) -> None:
        cfg = _make_config("http://example.com/hook", max_retries=3, retry_delay=1.0)
        client = WebhookNotificationClient(cfg)
        resp_500 = _fake_response(500)
        with patch("urllib.request.urlopen", return_value=resp_500):
            with patch("time.sleep") as mock_sleep:
                client.send("test_event", {})
        # delays should double: 1.0, 2.0, 4.0
        assert mock_sleep.call_args_list == [call(1.0), call(2.0), call(4.0)]


class TestWebhookNoRetryOn4xx:
    def test_no_retry_on_404(self) -> None:
        cfg = _make_config("http://example.com/hook", max_retries=3)
        client = WebhookNotificationClient(cfg)
        resp_404 = _fake_response(404)
        with patch("urllib.request.urlopen", return_value=resp_404) as mock_open:
            with patch("time.sleep") as mock_sleep:
                client.send("test_event", {})
        assert mock_open.call_count == 1
        mock_sleep.assert_not_called()

    def test_no_retry_on_400(self) -> None:
        cfg = _make_config("http://example.com/hook", max_retries=3)
        client = WebhookNotificationClient(cfg)
        resp_400 = _fake_response(400)
        with patch("urllib.request.urlopen", return_value=resp_400) as mock_open:
            with patch("time.sleep") as mock_sleep:
                client.send("test_event", {})
        assert mock_open.call_count == 1
        mock_sleep.assert_not_called()


class TestWebhookRetryOnNetworkError:
    def test_retries_on_url_error(self) -> None:
        cfg = _make_config("http://example.com/hook", max_retries=2, retry_delay=0.0)
        client = WebhookNotificationClient(cfg)
        resp_200 = _fake_response(200)
        network_err = urllib.error.URLError("connection refused")
        side = [network_err, network_err, resp_200]
        with patch("urllib.request.urlopen", side_effect=side) as mock_open:
            with patch("time.sleep"):
                client.send("test_event", {})
        assert mock_open.call_count == 3

    def test_never_raises(self) -> None:
        cfg = _make_config("http://example.com/hook", max_retries=1, retry_delay=0.0)
        client = WebhookNotificationClient(cfg)
        with patch("urllib.request.urlopen", side_effect=Exception("boom")):
            with patch("time.sleep"):
                client.send("test_event", {})  # must not raise
