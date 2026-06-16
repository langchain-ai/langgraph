from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import orjson
import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from langgraph_sdk.stream.transport.ws import ProtocolWebSocketTransport
from streaming._events import values_event


class _FakeAsyncWebSocket:
    def __init__(
        self,
        frames: list[dict[str, Any]],
        *,
        fail_after: int | None = None,
    ) -> None:
        self.frames = list(frames)
        self.fail_after = fail_after
        self.sent: list[str] = []
        self.closed = False

    async def __aenter__(self) -> _FakeAsyncWebSocket:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data.decode() if isinstance(data, bytes) else data)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str]:
        for index, frame in enumerate(self.frames, start=1):
            yield orjson.dumps(frame).decode()
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("scripted websocket failure")

    async def close(self, code: int = 1000, reason: str = "") -> None:  # noqa: ARG002
        self.closed = True
        self.close_code = code


class _ClosingFakeAsyncWebSocket:
    """Fake websocket that yields frames then raises a ConnectionClosed variant."""

    def __init__(
        self,
        frames: list[dict[str, Any]],
        *,
        close_exc: BaseException,
    ) -> None:
        self.frames = list(frames)
        self.close_exc = close_exc
        self.sent: list[str] = []
        self.closed = False

    async def __aenter__(self) -> _ClosingFakeAsyncWebSocket:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.closed = True

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data.decode() if isinstance(data, bytes) else data)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str]:
        for frame in self.frames:
            yield orjson.dumps(frame).decode()
        raise self.close_exc

    async def close(self, code: int = 1000, reason: str = "") -> None:  # noqa: ARG002
        self.closed = True


async def test_ws_close_code_1000_resolves_done_with_none():
    """Close code 1000 (normal closure) resolves `done` with None."""
    socket = _ClosingFakeAsyncWebSocket(
        [values_event(seq=1)],
        close_exc=ConnectionClosedOK(Close(1000, "OK"), None),
    )

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client, thread_id="t-1", connect=connect
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        _ = [e async for e in handle.events]
        err = await asyncio.wait_for(handle.done, timeout=1.0)
        await handle.close()

    assert err is None


async def test_ws_close_code_1006_resolves_done_with_error():
    """Close code 1006 (abnormal) resolves `done` with ConnectionClosedError."""
    socket = _ClosingFakeAsyncWebSocket(
        [],
        close_exc=ConnectionClosedError(None, Close(1006, "abnormal")),
    )

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client, thread_id="t-1", connect=connect
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        _ = [e async for e in handle.events]
        err = await asyncio.wait_for(handle.done, timeout=1.0)
        await handle.close()

    assert isinstance(err, ConnectionClosedError)


async def test_ws_close_code_4000_resolves_done_with_error():
    """4xxx close codes resolve `done` with ConnectionClosedError."""
    socket = _ClosingFakeAsyncWebSocket(
        [],
        close_exc=ConnectionClosedError(None, Close(4000, "app error")),
    )

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client, thread_id="t-1", connect=connect
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        _ = [e async for e in handle.events]
        err = await asyncio.wait_for(handle.done, timeout=1.0)
        await handle.close()

    assert isinstance(err, ConnectionClosedError)


class _RawFrameAsyncWebSocket:
    """Fake websocket that yields arbitrary raw strings (not dicts)."""

    def __init__(self, raw_frames: list[str]) -> None:
        self.raw_frames = list(raw_frames)
        self.sent: list[str] = []
        self.closed = False

    async def __aenter__(self) -> _RawFrameAsyncWebSocket:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.closed = True

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data.decode() if isinstance(data, bytes) else data)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str]:
        for frame in self.raw_frames:
            yield frame

    async def close(self, code: int = 1000, reason: str = "") -> None:  # noqa: ARG002
        self.closed = True


async def test_decode_frame_invalid_json_surfaces_error_on_done():
    """A WS frame with invalid JSON resolves `done` with a RuntimeError."""
    socket = _RawFrameAsyncWebSocket(["not-valid-json{{{"])

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client, thread_id="t-1", connect=connect
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        received = [e async for e in handle.events]
        err = await asyncio.wait_for(handle.done, timeout=1.0)
        await handle.close()

    assert received == []
    assert isinstance(err, RuntimeError)
    assert "not valid JSON" in str(err)


async def test_decode_frame_non_dict_surfaces_error_on_done():
    """A WS frame that is valid JSON but not an object resolves `done` with RuntimeError."""
    socket = _RawFrameAsyncWebSocket(["[1, 2, 3]"])

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client, thread_id="t-1", connect=connect
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        received = [e async for e in handle.events]
        err = await asyncio.wait_for(handle.done, timeout=1.0)
        await handle.close()

    assert received == []
    assert isinstance(err, RuntimeError)
    assert "not a JSON object" in str(err)


async def test_websocket_url_uses_ws_scheme_and_base_path():
    seen: list[tuple[str, list[tuple[str, str]] | None]] = []
    socket = _FakeAsyncWebSocket([])

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        seen.append((url, additional_headers))
        return socket

    async with httpx.AsyncClient(base_url="https://example.com/api") as client:
        transport = ProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            headers={"x-test": "1"},
            connect=connect,
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        _ = [event async for event in handle.events]
        await handle.close()

    assert seen == [
        (
            "wss://example.com/api/threads/t-1/stream/events",
            [("x-test", "1")],
        )
    ]


async def test_websocket_sends_subscribe_body_and_yields_events():
    event = values_event(seq=1, values={"counter": 1})
    socket = _FakeAsyncWebSocket([event])

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            connect=connect,
        )
        handle = transport.open_event_stream(
            {"channels": ["values"], "namespaces": [[]], "since": 7}
        )
        await asyncio.wait_for(handle.ready, timeout=1.0)
        received = [event async for event in handle.events]
        err = await asyncio.wait_for(handle.done, timeout=1.0)
        await handle.close()

    assert orjson.loads(socket.sent[0]) == {
        "id": 1,
        "method": "subscription.subscribe",
        "params": {
            "channels": ["values"],
            "namespaces": [[]],
            "since": 7,
        },
    }
    assert received == [event]
    assert err is None


async def test_websocket_done_records_post_ready_error():
    socket = _FakeAsyncWebSocket([values_event(seq=1)], fail_after=1)

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            connect=connect,
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        received = [event async for event in handle.events]
        err = await asyncio.wait_for(handle.done, timeout=1.0)
        await handle.close()

    assert received == [values_event(seq=1)]
    assert isinstance(err, RuntimeError)
    assert "scripted websocket failure" in str(err)


async def test_websocket_send_command_uses_http_commands_endpoint():
    from streaming._fake_server import FakeServer

    fake = FakeServer()
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ws = ProtocolWebSocketTransport(client=client, thread_id="t-1")
        result = await ws.send_command(
            {"id": 3, "method": "run.start", "params": {"input": {"x": 1}}}
        )

    assert result == {"type": "success", "id": 3, "result": {"run_id": "run-1"}}
    assert fake.received_commands[0]["method"] == "run.start"


async def test_websocket_open_event_stream_raises_when_closed():
    async with httpx.AsyncClient(base_url="http://test") as client:
        ws = ProtocolWebSocketTransport(client=client, thread_id="t-1")
        await ws.close()
        with pytest.raises(RuntimeError, match="closed"):
            ws.open_event_stream({"channels": ["values"]})


async def test_websocket_transport_feeds_async_stream_controller():
    from langgraph_sdk.stream.controller import StreamController

    socket = _FakeAsyncWebSocket(
        [
            values_event(seq=1, values={"counter": 1}),
            values_event(seq=2, values={"counter": 2}),
        ]
    )

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            connect=connect,
        )

        async def gate() -> None:
            return None

        controller = StreamController(transport=transport, run_start_gate=gate)
        sub = controller.register_subscription({"channels": ["values"]})
        await controller.reconcile_stream({"channels": ["values"]})
        controller.ensure_fanout_running()

        first = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
        second = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
        end = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
        await controller.close()
        await transport.close()

    assert first["seq"] == 1
    assert second["seq"] == 2
    assert end is None


async def test_ws_transport_accepts_max_queue_size_kwarg():
    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            max_queue_size=42,
        )
        assert transport._max_queue_size == 42


async def test_ws_transport_default_max_queue_size_is_1024():
    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
        )
        assert transport._max_queue_size == 1024


async def test_websocket_controller_reconnects_with_since_after_drop():
    from langgraph_sdk.stream.controller import StreamController

    first_socket = _FakeAsyncWebSocket(
        [values_event(seq=1, values={"counter": 1})],
        fail_after=1,
    )
    second_socket = _FakeAsyncWebSocket([values_event(seq=2, values={"counter": 2})])
    sockets = [first_socket, second_socket]
    sent_urls: list[str] = []

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = additional_headers
        sent_urls.append(url)
        return sockets.pop(0)

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            connect=connect,
        )

        async def gate() -> None:
            return None

        controller = StreamController(transport=transport, run_start_gate=gate)
        sub = controller.register_subscription({"channels": ["values"]})
        await controller.reconcile_stream({"channels": ["values"]})
        controller.ensure_fanout_running()

        first = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
        second = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
        end = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
        await controller.close()
        await transport.close()

    assert first["seq"] == 1
    assert second["seq"] == 2
    assert end is None
    assert len(sent_urls) == 2
    assert orjson.loads(second_socket.sent[0])["params"]["since"] == 1


async def test_async_close_sends_normal_close_frame():
    """`handle.close()` sends a WebSocket close frame with code 1000 explicitly."""
    import asyncio

    # Use an event to distinguish an explicit close(code=1000) call from
    # the implicit one in __aexit__ when the task is cancelled.
    explicit_1000: list[bool] = []
    ready_event = asyncio.Event()

    class _TrackingWebSocket(_FakeAsyncWebSocket):
        def __aiter__(self) -> AsyncIterator[str]:
            return self._wait_forever()

        async def _wait_forever(self) -> AsyncIterator[str]:  # type: ignore[override]
            ready_event.set()
            # Block until cancelled — never yield frames.
            await asyncio.sleep(9999)
            return
            yield  # make it an async generator

        async def close(  # type: ignore[override]
            self,
            code: int = 1000,
            reason: str = "",  # noqa: ARG002
        ) -> None:
            explicit_1000.append(code == 1000)
            self.closed = True

    socket = _TrackingWebSocket([])

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client, thread_id="t-1", connect=connect
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        await asyncio.wait_for(ready_event.wait(), timeout=1.0)
        await handle.close()

    # close() must have been called at least once with code=1000.
    assert explicit_1000, "ws.close() was never called"
    assert explicit_1000[0], "first ws.close() call did not use code=1000"


async def test_ws_transport_forwards_ping_kwargs():
    """ping_interval and ping_timeout are stored and forwarded to websockets.connect."""
    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            ping_interval=15.0,
            ping_timeout=20.0,
        )
        assert transport._ping_interval == 15.0
        assert transport._ping_timeout == 20.0


async def test_ws_transport_ping_kwargs_forwarded_to_connect():
    """ping_interval and ping_timeout are passed through to the connect callable."""
    captured_kwargs: list[dict[str, Any]] = []
    socket = _FakeAsyncWebSocket([])

    def connect(
        url: str,
        additional_headers: list[tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = (url, additional_headers)
        captured_kwargs.append(kwargs)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client,
            thread_id="t-1",
            connect=connect,
            ping_interval=15.0,
            ping_timeout=20.0,
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        _ = [e async for e in handle.events]
        await handle.close()

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0].get("ping_interval") == 15.0
    assert captured_kwargs[0].get("ping_timeout") == 20.0


async def test_ws_handshake_forwards_httpx_client_cookies():
    """Cookies set on the httpx.AsyncClient are forwarded to the WS handshake."""
    captured_headers: list[list[tuple[str, str]]] = []
    socket = _FakeAsyncWebSocket([])

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = url
        captured_headers.append(list(additional_headers or []))
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        client.cookies.set("session", "abc123")
        client.cookies.set("other", "xyz")
        transport = ProtocolWebSocketTransport(
            client=client, thread_id="t-1", connect=connect
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        _ = [e async for e in handle.events]
        await handle.close()

    assert len(captured_headers) == 1
    headers_dict = dict(captured_headers[0])
    assert "Cookie" in headers_dict
    cookie_val = headers_dict["Cookie"]
    assert "session=abc123" in cookie_val
    assert "other=xyz" in cookie_val


async def test_ws_server_initiated_mid_stream_close_surfaces_error():
    """A server that closes mid-stream with code 1011 surfaces ConnectionClosedError on done."""
    socket = _ClosingFakeAsyncWebSocket(
        [values_event(seq=1)],
        close_exc=ConnectionClosedError(None, Close(1011, "server error")),
    )

    def connect(
        url: str, additional_headers: list[tuple[str, str]] | None = None, **_kw: Any
    ):
        _ = (url, additional_headers)
        return socket

    async with httpx.AsyncClient(base_url="http://test") as client:
        transport = ProtocolWebSocketTransport(
            client=client, thread_id="t-1", connect=connect
        )
        handle = transport.open_event_stream({"channels": ["values"]})
        await asyncio.wait_for(handle.ready, timeout=1.0)
        received = [e async for e in handle.events]
        err = await asyncio.wait_for(handle.done, timeout=1.0)
        await handle.close()

    assert received == [values_event(seq=1)]
    assert isinstance(err, ConnectionClosedError)
