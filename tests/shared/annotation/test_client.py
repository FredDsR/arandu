"""The Label Studio client is transport only, and it never leaks the token."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from arandu.shared.annotation.client import (
    HttpLabelStudioClient,
    LabelStudioError,
    build_client_from_settings,
)
from arandu.shared.annotation.settings import LabelStudioSettings

if TYPE_CHECKING:
    from pathlib import Path


def _client(handler: Any, token: str = "secret-token") -> HttpLabelStudioClient:
    transport = httpx.MockTransport(handler)
    return HttpLabelStudioClient(
        base_url="https://label.example.test",
        token=token,
        transport=transport,
    )


def _b64url(payload: dict[str, Any]) -> str:
    """Encode a JWT segment the way a real token does, without the padding."""
    raw = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _jwt(token_type: str = "refresh") -> str:
    """Build a JWT-shaped fixture token. No real credential is involved."""
    header = _b64url({"alg": "HS256", "typ": "JWT"})
    payload = _b64url({"token_type": token_type, "exp": 1, "jti": "fixture", "user_id": 1})
    return f"{header}.{payload}.c2lnbmF0dXJl"


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

    def test_an_empty_token_is_a_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`.env.example` ships the key with no value, so this is the copy-it flow.

        Without the constraint it validates and the push sends
        `Authorization: Token ` for a 401 mid-import, instead of the CLI's
        "Label Studio is not configured" message.
        """
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_URL", "https://label.example.test")
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_TOKEN", "")
        with pytest.raises(ValueError):
            LabelStudioSettings(_env_file=None)

    def test_trailing_slash_is_normalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_URL", "https://label.example.test/")
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_TOKEN", "abc")
        assert LabelStudioSettings().url == "https://label.example.test"

    def test_reads_a_dot_env_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """`.env.example` documents these vars, so `.env` has to be read."""
        monkeypatch.delenv("ARANDU_LABEL_STUDIO_URL", raising=False)
        monkeypatch.delenv("ARANDU_LABEL_STUDIO_TOKEN", raising=False)
        (tmp_path / ".env").write_text(
            "ARANDU_LABEL_STUDIO_URL=https://from-dotenv.example.test\n"
            "ARANDU_LABEL_STUDIO_TOKEN=dotenv-token\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        settings = LabelStudioSettings()

        assert settings.url == "https://from-dotenv.example.test"
        assert settings.token.get_secret_value() == "dotenv-token"

    def test_the_environment_wins_over_dot_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / ".env").write_text(
            "ARANDU_LABEL_STUDIO_URL=https://from-dotenv.example.test\n"
            "ARANDU_LABEL_STUDIO_TOKEN=dotenv-token\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_URL", "https://from-env.example.test")
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_TOKEN", "env-token")

        settings = LabelStudioSettings()

        assert settings.url == "https://from-env.example.test"
        assert settings.token.get_secret_value() == "env-token"

    def test_builder_returns_a_configured_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_URL", "https://label.example.test")
        monkeypatch.setenv("ARANDU_LABEL_STUDIO_TOKEN", "abc")
        assert isinstance(build_client_from_settings(LabelStudioSettings()), HttpLabelStudioClient)


class TestAuthentication:
    """Both token schemes, pinned by behaviour rather than by our assumption.

    The live instance issues JWT personal access tokens and rejects the legacy
    ``Token`` header outright. Asserting only the header our own code produces
    could never have caught that, so these tests assert what the server is
    asked for: which endpoints are called, in which order, with which body.
    """

    def test_a_legacy_token_is_sent_as_token_and_never_refreshes(self) -> None:
        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.url.path, request.headers.get("Authorization", "")))
            return httpx.Response(201, json={"id": 42})

        assert _client(handler, token="secret-token").create_project("t", "<View/>") == 42
        assert calls == [("/api/projects/", "Token secret-token")]

    def test_a_token_with_dots_but_no_json_payload_is_legacy(self) -> None:
        """Decode failures fall back to the legacy scheme instead of raising."""
        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.url.path, request.headers.get("Authorization", "")))
            return httpx.Response(201, json={"id": 42})

        _client(handler, token="not.a.jwt").create_project("t", "<View/>")
        assert calls == [("/api/projects/", "Token not.a.jwt")]

    def test_a_jwt_refresh_token_is_exchanged_for_a_bearer_access_token(self) -> None:
        refresh = _jwt()
        calls: list[tuple[str, str]] = []
        bodies: list[Any] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.url.path, request.headers.get("Authorization", "")))
            if request.url.path == "/api/token/refresh":
                bodies.append(json.loads(request.read().decode()))
                return httpx.Response(200, json={"access": "access-1"})
            return httpx.Response(201, json={"id": 42})

        assert _client(handler, token=refresh).create_project("t", "<View/>") == 42
        assert [path for path, _ in calls] == ["/api/token/refresh", "/api/projects/"]
        assert bodies == [{"refresh": refresh}]
        assert calls[1][1] == "Bearer access-1"

    def test_the_exchange_happens_once_and_is_cached(self) -> None:
        refreshes = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal refreshes
            if request.url.path == "/api/token/refresh":
                refreshes += 1
                return httpx.Response(200, json={"access": "access-1"})
            if request.url.path.endswith("/import"):
                return httpx.Response(201, json={"task_count": 1})
            return httpx.Response(200, json=[])

        client = _client(handler, token=_jwt())
        client.import_tasks(42, [{"data": {}}])
        client.export_annotations(42)

        assert refreshes == 1

    def test_a_401_triggers_exactly_one_re_exchange_and_one_retry(self) -> None:
        issued = ["access-1", "access-2"]
        seen: list[str] = []
        refreshes = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal refreshes
            if request.url.path == "/api/token/refresh":
                access = issued[refreshes]
                refreshes += 1
                return httpx.Response(200, json={"access": access})
            seen.append(request.headers.get("Authorization", ""))
            if len(seen) == 1:
                return httpx.Response(401, json={"detail": "token not valid"})
            return httpx.Response(200, json=[{"id": 1}])

        assert _client(handler, token=_jwt()).export_annotations(42) == [{"id": 1}]
        assert refreshes == 2
        assert seen == ["Bearer access-1", "Bearer access-2"]

    def test_a_second_consecutive_401_raises_without_looping(self) -> None:
        refresh = _jwt()
        refreshes = 0
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal refreshes, attempts
            if request.url.path == "/api/token/refresh":
                refreshes += 1
                return httpx.Response(200, json={"access": "access-token-value"})
            attempts += 1
            return httpx.Response(401, json={"detail": "token not valid"})

        with pytest.raises(LabelStudioError) as excinfo:
            _client(handler, token=refresh).export_annotations(42)

        assert refreshes == 2
        assert attempts == 2
        message = str(excinfo.value)
        assert "401" in message
        assert refresh not in message
        assert "access-token-value" not in message

    def test_a_failed_refresh_is_an_actionable_error_without_the_tokens(self) -> None:
        refresh = _jwt()

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/token/refresh"
            return httpx.Response(401, json={"detail": refresh, "code": "token_not_valid"})

        with pytest.raises(LabelStudioError) as excinfo:
            _client(handler, token=refresh).create_project("t", "<View/>")

        message = str(excinfo.value)
        assert "Access Token" in message
        assert "ARANDU_LABEL_STUDIO_URL" in message
        assert refresh not in message

    def test_a_refresh_response_without_an_access_token_is_an_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"refresh": "rotated"})

        with pytest.raises(LabelStudioError, match="access"):
            _client(handler, token=_jwt()).create_project("t", "<View/>")


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

    def test_settings_are_posted_alongside_title_and_config(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.read().decode())
            return httpx.Response(201, json={"id": 42})

        _client(handler).create_project("t", "<View/>", settings={"show_skip_button": False})
        assert seen["body"] == {
            "title": "t",
            "label_config": "<View/>",
            "show_skip_button": False,
        }

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

    def test_a_response_without_a_count_returns_none(self) -> None:
        """An unreported count must not be read as `len(tasks)`.

        That fallback would report success for a partial import and leave the
        missing pairs indistinguishable from unrated ones in every pull.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json={"import": 12, "status": "created"})

        assert _client(handler).import_tasks(42, [{"data": {}}, {"data": {}}]) is None

    def test_a_non_dict_response_returns_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json=[{"id": 1}])

        assert _client(handler).import_tasks(42, [{"data": {}}]) is None

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
