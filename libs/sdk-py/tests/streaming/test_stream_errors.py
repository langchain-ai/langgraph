from __future__ import annotations

import httpx
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

from langgraph_sdk.stream.errors import (
    is_non_retryable_stream_error,
    stream_error_frame,
)


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://test/threads/t-1/stream/events")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("scripted", request=request, response=response)


def _ws_close(code: int) -> ConnectionClosedError:
    return ConnectionClosedError(Close(code, "scripted"), None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(_http_error(403), True, id="http-403"),
        pytest.param(_http_error(404), True, id="http-404"),
        pytest.param(_http_error(408), False, id="http-408"),
        pytest.param(_http_error(429), False, id="http-429"),
        pytest.param(_http_error(503), False, id="http-503"),
        pytest.param(_ws_close(4404), True, id="ws-close-4404"),
        pytest.param(_ws_close(4422), True, id="ws-close-4422"),
        pytest.param(_ws_close(4429), False, id="ws-close-4429"),
        pytest.param(_ws_close(4503), False, id="ws-close-4503"),
        pytest.param(_ws_close(1011), False, id="ws-close-1011"),
        pytest.param(
            InvalidStatus(Response(403, "Forbidden", Headers())),
            True,
            id="ws-upgrade-403",
        ),
        pytest.param(RuntimeError("network down"), False, id="no-status"),
        pytest.param(None, False, id="none"),
    ],
)
def test_is_non_retryable_stream_error(
    error: BaseException | None, expected: bool
) -> None:
    assert is_non_retryable_stream_error(error) is expected


def test_stream_error_frame_reads_only_unsolicited_error_frames() -> None:
    frame = {
        "type": "error",
        "id": None,
        "error": "unknown_error",
        "message": "Thread event stream failed.",
    }

    assert (
        str(stream_error_frame(frame))
        == "Protocol error [unknown_error]: Thread event stream failed."
    )
    assert stream_error_frame({**frame, "id": 3}) is None
    assert stream_error_frame({"type": "event", "method": "values"}) is None
