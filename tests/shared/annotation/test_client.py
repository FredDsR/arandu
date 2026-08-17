"""The Label Studio client is transport only, and it never leaks the token."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from arandu.shared.annotation.client import (
    HttpLabelStudioClient,
    LabelStudioError,
    build_client_from_settings,
)
from arandu.shared.annotation.settings import LabelStudioSettings


def _client(handler: Any) -> HttpLabelStudioClient:
    transport = httpx.MockTransport(handler)
    return HttpLabelStudioClient(
        base_url="https://label.example.test",
        token="secret-token",
        transport=transport,
    )


class TestSettings:
    def test_reads_the_arandu_label_studio_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_URL", "https://label.example.test")
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_TOKEN", "abc")
        settings = LabelStudioSettings()
        assert settings.url == "https://label.example.test"
        assert settings.token.get_secret_value() == "abc"

    def test_token_never_appears_in_repr_or_model_dump(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_URL", "https://label.example.test")
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_TOKEN", "super-secret-value")
        settings = LabelStudioSettings()
        assert "super-secret-value" not in repr(settings)
        assert "super-secret-value" not in str(settings.model_dump())

    def test_missing_token_is_a_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARANDU_LABEL_STUDIO_TOKEN", raising=False)
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_URL", "https://label.example.test")
        with pytest.raises(ValueError):
            LabelStudioSettings(_env_file=None)

    def test_trailing_slash_is_normalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_URL", "https://label.example.test/")
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_TOKEN", "abc")
        assert LabelStudioSettings().url == "https://label.example.test"

    def test_builder_returns_a_configured_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_URL", "https://label.example.test")
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_TOKEN", "abc")
        assert isinstance(build_client_from_settings(LabelStudioSettings()), HttpLabelStudioClient)


class TestCreateProject:
    def test_posts_title_and_config_and_returns_the_id(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization")
            seen["body"] = request.read().decode()
            return httpx.Response(201, json={"id": 42, "title": "t"})

        assert _client(handler).create_project("t", "<View/>") == 42
        assert seen["url"] == "https://label.example.test/api/projects/"
        assert seen["auth"] == "Token secret-token"
        assert "<View/>" in seen["body"]

    def test_http_error_is_wrapped_without_the_token(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="forbidden")

        with pytest.raises(LabelStudioError) as excinfo:
            _client(handler).create_project("t", "<View/>")
        assert "403" in str(excinfo.value)
        assert "secret-token" not in str(excinfo.value)

    def test_transport_error_is_wrapped_without_the_token(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with pytest.raises(LabelStudioError) as excinfo:
            _client(handler).create_project("t", "<View/>")
        assert "connection refused" in str(excinfo.value)
        assert "secret-token" not in str(excinfo.value)

    def test_response_without_an_id_is_an_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json={"title": "t"})

        with pytest.raises(LabelStudioError, match="id"):
            _client(handler).create_project("t", "<View/>")


class TestImportTasks:
    def test_posts_the_task_list_and_returns_the_count(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/projects/42/import"
            return httpx.Response(201, json={"task_count": 2})

        assert _client(handler).import_tasks(42, [{"data": {}}, {"data": {}}]) == 2

    def test_error_is_wrapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        with pytest.raises(LabelStudioError, match="500"):
            _client(handler).import_tasks(42, [])


class TestExportAnnotations:
    def test_requests_json_export_and_returns_the_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/projects/42/export"
            assert request.url.params["exportType"] == "JSON"
            return httpx.Response(200, json=[{"id": 1, "data": {"task_id": 0}}])

        assert _client(handler).export_annotations(42)[0]["data"]["task_id"] == 0

    def test_non_list_payload_is_an_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"detail": "nope"})

        with pytest.raises(LabelStudioError, match="list"):
            _client(handler).export_annotations(42)
