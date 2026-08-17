"""Thin HTTP client for the Label Studio API (Community Edition 1.23).

Transport only: three calls, no domain logic. ``push`` and ``pull`` depend on
the :class:`LabelStudioClient` Protocol, so both are fully testable without a
server and without the network.

No ``label-studio-sdk`` dependency: the project rule is that provider SDKs stay
out of the pipeline layers, and three endpoints do not justify one.

Errors never carry the token. A stack trace or a log line that leaks the
credential is a worse failure than the request that produced it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import httpx

if TYPE_CHECKING:
    from arandu.shared.annotation.settings import LabelStudioSettings

logger = logging.getLogger(__name__)

_MAX_BODY_IN_ERROR = 500


class LabelStudioError(RuntimeError):
    """Raised when the Label Studio API rejects a request or answers unusably."""


class LabelStudioClient(Protocol):
    """The three operations the annotation instrument needs."""

    def create_project(
        self, title: str, label_config: str, *, settings: dict[str, Any] | None = None
    ) -> int:
        """Create a project and return its id."""
        ...

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> int:
        """Import tasks into a project and return how many were accepted."""
        ...

    def export_annotations(self, project_id: int) -> list[dict[str, Any]]:
        """Return the project's tasks with their annotations."""
        ...


class HttpLabelStudioClient:
    """:class:`LabelStudioClient` over httpx.

    Args:
        base_url: Instance base URL, without a trailing slash.
        token: API token, sent as ``Authorization: Token <token>``.
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
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Token {token}"},
            timeout=timeout,
            transport=transport,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Issue a request and turn any non-2xx into a token-free error."""
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise LabelStudioError(f"{method} {path} failed to reach Label Studio: {exc}") from exc
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

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> int:
        """Import tasks and return the accepted count.

        Raises:
            LabelStudioError: On any API failure.
        """
        data = self._request("POST", f"/api/projects/{project_id}/import", json=tasks).json()
        count = data.get("task_count") if isinstance(data, dict) else None
        return count if isinstance(count, int) else len(tasks)

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
    Pydantic's secret-wrapper type.
    """
    return HttpLabelStudioClient(
        base_url=settings.url,
        token=settings.token.get_secret_value(),
        timeout=settings.timeout,
    )
