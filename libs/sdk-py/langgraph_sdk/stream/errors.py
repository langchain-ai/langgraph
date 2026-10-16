"""Thread-stream failures, shared by the async and sync clients."""

from __future__ import annotations

from typing import Any

import httpx
from websockets.exceptions import ConnectionClosed, InvalidStatus


def protocol_error(response: dict[str, Any]) -> RuntimeError:
    """Build the error for a protocol `{"type": "error"}` envelope."""
    code = response.get("error", "unknown")
    message = response.get("message", "")
    return RuntimeError(f"Protocol error [{code}]: {message}")


def stream_error_frame(event: Any) -> RuntimeError | None:
    """Return the error in an unsolicited `{"type": "error", "id": null}` frame.

    The server sends one just before it closes a thread stream whose read failed.
    """
    if (
        isinstance(event, dict)
        and event.get("type") == "error"
        and event.get("id") is None
    ):
        return protocol_error(event)
    return None


def is_non_retryable_stream_error(error: BaseException | None) -> bool:
    """Whether reopening the stream would fail the same way.

    True for a 4xx other than 408 and 429: the status of a refused stream request
    or WebSocket upgrade, or a WebSocket close code of 4000 plus that status. A
    server that rejects a WebSocket after accepting it closes with such a code,
    because a browser can't read the status of a refused upgrade.
    """
    if isinstance(error, (httpx.HTTPStatusError, InvalidStatus)):
        status = error.response.status_code
    elif isinstance(error, ConnectionClosed) and error.rcvd is not None:
        status = error.rcvd.code - 4000
    else:
        return False
    return 400 <= status < 500 and status not in (408, 429)
