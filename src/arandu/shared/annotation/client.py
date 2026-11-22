"""Thin HTTP client for the Label Studio API (Community Edition 1.23).

Transport only: three calls, no domain logic. ``push`` and ``pull`` depend on
the :class:`LabelStudioClient` Protocol, so both are fully testable without a
server and without the network.

No ``label-studio-sdk`` dependency: the project rule is that provider SDKs stay
out of the pipeline layers, and three endpoints do not justify one.

Two authentication schemes are supported, because Label Studio issues either
one depending on how the organization is configured and the operator cannot be
expected to know which one they were handed:

* **Legacy API token.** Sent as ``Authorization: Token <token>``.
* **JWT personal access token.** What Account and Settings hands out on a
  current instance is a *refresh* token. It is not accepted on the API
  endpoints at all; it has to be exchanged at ``POST /api/token/refresh`` for a
  short-lived *access* token, which is then sent as
  ``Authorization: Bearer <access>``.

The token's own shape decides which path runs, so nothing has to be declared in
the environment. Access tokens are short-lived (a task import plus an export
can outlive one), so the exchange is lazy, the result is cached, and a single
401 triggers exactly one re-exchange and one retry.

Errors never carry the token. Neither the refresh token nor the derived access
token may appear in an exception message, a log line, or an artifact: a stack
trace or a log line that leaks the credential is a worse failure than the
request that produced it.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

import httpx

if TYPE_CHECKING:
    from arandu.shared.annotation.settings import LabelStudioSettings

logger = logging.getLogger(__name__)

_MAX_BODY_IN_ERROR = 500
_JWT_SEGMENTS = 3
_REFRESH_PATH = "/api/token/refresh"


def _decode_jwt_payload(token: str) -> dict[str, Any] | None:
    """Return a JWT's unverified payload claims, or ``None`` if it is not a JWT.

    Only enough of the token is read to pick a code path. Verifying the
    signature is the server's job, and a whole ``pyjwt`` dependency to read one
    unverified claim would not earn its place.

    Args:
        token: The credential as the operator pasted it.

    Returns:
        The decoded payload object, or ``None`` when the value is not a
        three-segment token with a base64url-encoded JSON object in the middle.
    """
    segments = token.split(".")
    if len(segments) != _JWT_SEGMENTS:
        return None
    payload_segment = segments[1]
    padded = payload_segment + "=" * (-len(payload_segment) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, binascii.Error):
        return None
    return payload if isinstance(payload, dict) else None


def _is_jwt_refresh_token(token: str) -> bool:
    """Report whether ``token`` is a JWT refresh token needing an exchange."""
    payload = _decode_jwt_payload(token)
    return payload is not None and payload.get("token_type") == "refresh"


class LabelStudioError(RuntimeError):
    """Raised when the Label Studio API rejects a request or answers unusably."""


class LabelStudioClient(Protocol):
    """The three operations the annotation instrument needs."""

    def create_project(
        self, title: str, label_config: str, *, settings: dict[str, Any] | None = None
    ) -> int:
        """Create a project and return its id."""
        ...

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> int | None:
        """Import tasks and return how many were accepted, or ``None`` if unreported."""
        ...

    def export_annotations(self, project_id: int) -> list[dict[str, Any]]:
        """Return the project's tasks with their annotations."""
        ...


class HttpLabelStudioClient:
    """:class:`LabelStudioClient` over httpx.

    Args:
        base_url: Instance base URL, without a trailing slash.
        token: API token. A legacy token is sent as
            ``Authorization: Token <token>``; a JWT refresh token is exchanged
            for an access token and sent as ``Authorization: Bearer <access>``.
        timeout: Per-request timeout in seconds.
        transport: Injected transport, for tests.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._token = token
        self._needs_exchange = _is_jwt_refresh_token(token)
        self._access_token: str | None = None
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )
        logger.debug(
            "Label Studio authentication scheme: %s.",
            "JWT bearer (refresh exchange)" if self._needs_exchange else "legacy token",
        )

    def _exchange_refresh_token(self) -> str:
        """Trade the JWT refresh token for a short-lived access token.

        The request body carries the credential, so neither it nor the response
        is ever logged or folded into an error message.

        Returns:
            The freshly issued access token.

        Raises:
            LabelStudioError: If the exchange fails or returns no access token.
        """
        try:
            response = self._client.post(_REFRESH_PATH, json={"refresh": self._token})
        except httpx.HTTPError as exc:
            raise LabelStudioError(
                f"Could not reach Label Studio to refresh the access token: {exc}"
            ) from exc
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LabelStudioError(
                f"Label Studio rejected the personal access token at {_REFRESH_PATH} "
                f"(HTTP {response.status_code}). Copy a current token from Account and "
                "Settings, Access Token, into ARANDU_LABEL_STUDIO_TOKEN, and check that "
                "ARANDU_LABEL_STUDIO_URL points at the instance that issued it."
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LabelStudioError(
                f"Label Studio answered {_REFRESH_PATH} with a non-JSON body."
            ) from exc
        access = data.get("access") if isinstance(data, dict) else None
        if not isinstance(access, str) or not access:
            raise LabelStudioError(
                f"Label Studio answered {_REFRESH_PATH} without an 'access' token."
            )
        logger.debug("Obtained a Label Studio access token.")
        return access

    def _authorization(self) -> str:
        """Return the ``Authorization`` header value, exchanging lazily."""
        if not self._needs_exchange:
            return f"Token {self._token}"
        if self._access_token is None:
            self._access_token = self._exchange_refresh_token()
        return f"Bearer {self._access_token}"

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Issue one authenticated request, wrapping transport errors."""
        headers = {**kwargs.pop("headers", {}), "Authorization": self._authorization()}
        try:
            return self._client.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise LabelStudioError(f"{method} {path} failed to reach Label Studio: {exc}") from exc

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Issue a request and turn any non-2xx into a token-free error.

        Access tokens expire mid-run, so a single 401 is treated as an expiry:
        the cached access token is dropped, exchanged once more, and the request
        is retried exactly once. A second 401 is a real credential problem and
        is reported rather than retried, so this can never loop.
        """
        response = self._send(method, path, **kwargs)
        if response.status_code == httpx.codes.UNAUTHORIZED and self._needs_exchange:
            logger.debug("%s %s returned 401; refreshing the access token once.", method, path)
            self._access_token = None
            response = self._send(method, path, **kwargs)
            if response.status_code == httpx.codes.UNAUTHORIZED:
                raise LabelStudioError(
                    f"{method} {path} returned 401 again after refreshing the access token. "
                    "The personal access token is most likely expired or revoked, or "
                    "ARANDU_LABEL_STUDIO_URL points at a different instance than the one "
                    "that issued it."
                )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            body = response.text[:_MAX_BODY_IN_ERROR]
            raise LabelStudioError(f"{method} {path} returned {response.status_code}: {body}")
        return response

    def create_project(
        self, title: str, label_config: str, *, settings: dict[str, Any] | None = None
    ) -> int:
        """Create a project and return its id.

        Args:
            title: Project title.
            label_config: The labeling config XML.
            settings: Extra project fields posted alongside ``title`` and
                ``label_config``. The Label Studio project serializer accepts
                the project's own settings in the create payload, so the caller
                decides the policy and this transport stays generic.

        Returns:
            The created project id.

        Raises:
            LabelStudioError: On any API failure, or if the response carries no
                integer id.
        """
        payload: dict[str, Any] = {"title": title, "label_config": label_config}
        if settings:
            payload.update(settings)
        data = self._request("POST", "/api/projects/", json=payload).json()
        project_id = data.get("id") if isinstance(data, dict) else None
        if not isinstance(project_id, int):
            raise LabelStudioError(
                f"Label Studio accepted the project but returned no integer id: {data!r}"
            )
        logger.info("Created Label Studio project %d.", project_id)
        return project_id

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> int | None:
        """Import tasks and return the accepted count.

        Returns:
            The count the instance reports it accepted, or ``None`` when the
            response carries no integer ``task_count``. An unreported count is
            not an accepted one: falling back to ``len(tasks)`` here would make
            the caller's partial-import guard unfireable in exactly the case it
            exists for, so what "no count" means is the caller's policy to
            decide and this transport stays generic.

        Raises:
            LabelStudioError: On any API failure.
        """
        data = self._request("POST", f"/api/projects/{project_id}/import", json=tasks).json()
        count = data.get("task_count") if isinstance(data, dict) else None
        return count if isinstance(count, int) else None

    def export_annotations(self, project_id: int) -> list[dict[str, Any]]:
        """Return the project's tasks with their annotations.

        Raises:
            LabelStudioError: On any API failure, or if the payload is not a
                list of tasks.
        """
        data = self._request(
            "GET", f"/api/projects/{project_id}/export", params={"exportType": "JSON"}
        ).json()
        if not isinstance(data, list):
            raise LabelStudioError(
                f"Expected a list of tasks from the export endpoint, got {type(data).__name__}."
            )
        return data

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()


def build_client_from_settings(settings: LabelStudioSettings) -> HttpLabelStudioClient:
    """Build a client from :class:`LabelStudioSettings`.

    Kept as a free function (the same shape as
    ``build_llm_client_from_settings``) so the CLI never constructs transport
    details by hand.

    The ``SecretStr`` token is unwrapped here, at the settings-to-client
    boundary: the client is transport only and should not know about
    Pydantic's secret-wrapper type. Which authentication scheme the token
    implies is decided by the client, from the token itself.
    """
    return HttpLabelStudioClient(
        base_url=settings.url,
        token=settings.token.get_secret_value(),
        timeout=settings.timeout,
    )
