from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from langgraph.pregel._remote_run_stream import (
    _AsyncRemoteGraphRunStream,
    _RemoteGraphRunStream,
)


def _make_sync_adapter(*, run_start_returns=None, run_start_raises=None):
    sync_client = MagicMock()
    sdk_thread = MagicMock()
    sdk_thread.thread_id = "thread-abc"
    sdk_thread.__enter__ = MagicMock(return_value=sdk_thread)
    sdk_thread.__exit__ = MagicMock(return_value=None)
    sdk_thread.run = MagicMock()
    if run_start_raises is not None:
        sdk_thread.run.start = MagicMock(side_effect=run_start_raises)
    else:
        sdk_thread.run.start = MagicMock(
            return_value=run_start_returns or {"run_id": "run-xyz"}
        )
    adapter = _RemoteGraphRunStream(
        sync_client=sync_client,
        sdk_thread=sdk_thread,
        input={"x": 1},
        config={"configurable": {}},
        metadata=None,
    )
    return adapter, sync_client, sdk_thread


def test_enter_calls_sdk_enter_then_run_start_and_captures_run_id():
    adapter, _, sdk_thread = _make_sync_adapter()
    with adapter as stream:
        assert stream is adapter
        sdk_thread.__enter__.assert_called_once()
        sdk_thread.run.start.assert_called_once_with(
            input={"x": 1}, config={"configurable": {}}, metadata=None
        )
        assert adapter._run_id == "run-xyz"


def test_exit_delegates_to_sdk_exit_with_exc_info():
    adapter, _, sdk_thread = _make_sync_adapter()
    with adapter:
        pass
    sdk_thread.__exit__.assert_called_once_with(None, None, None)


def test_enter_unwinds_sdk_cm_when_run_start_raises():
    adapter, _, sdk_thread = _make_sync_adapter(
        run_start_raises=RuntimeError("start boom")
    )
    with pytest.raises(RuntimeError, match="start boom"):
        with adapter:
            pytest.fail("body should not run")
    sdk_thread.__enter__.assert_called_once()
    sdk_thread.__exit__.assert_called_once()
    exc_info = sdk_thread.__exit__.call_args.args
    assert exc_info[0] is RuntimeError
    assert isinstance(exc_info[1], RuntimeError)
    assert adapter._run_id is None


def test_output_interrupted_interrupts_passthrough():
    adapter, _, sdk_thread = _make_sync_adapter()
    sdk_thread.output = {"foo": 1}
    sdk_thread.interrupted = True
    sdk_thread.interrupts = [{"interrupt_id": "i1", "namespace": [], "value": "v"}]
    with adapter as stream:
        assert stream.output == {"foo": 1}
        assert stream.interrupted is True
        assert stream.interrupts == [
            {"interrupt_id": "i1", "namespace": [], "value": "v"}
        ]


def test_iter_caches_first_subscription():
    adapter, _, sdk_thread = _make_sync_adapter()
    fake_events = [object(), object(), object()]
    sdk_thread.events = iter(fake_events)
    with adapter as stream:
        first = iter(stream)
        second = iter(stream)
        assert first is second
        assert list(first) == fake_events


def test_abort_cancels_run_and_closes_sdk():
    adapter, sync_client, sdk_thread = _make_sync_adapter()
    with adapter as stream:
        stream.abort()
        sync_client.runs.cancel.assert_called_once_with(
            "thread-abc", "run-xyz", wait=False
        )
        sdk_thread.close.assert_called_once()


def test_abort_before_enter_skips_cancel_but_closes_sdk():
    adapter, sync_client, sdk_thread = _make_sync_adapter()
    adapter.abort()
    sync_client.runs.cancel.assert_not_called()
    sdk_thread.close.assert_called_once()


def test_abort_is_idempotent():
    adapter, sync_client, sdk_thread = _make_sync_adapter()
    with adapter as stream:
        stream.abort()
        stream.abort()
        assert sync_client.runs.cancel.call_count == 1
        assert sdk_thread.close.call_count == 1


def test_abort_swallows_cancel_failure_and_still_closes():
    adapter, sync_client, sdk_thread = _make_sync_adapter()
    sync_client.runs.cancel.side_effect = RuntimeError("cancel boom")
    with adapter as stream:
        stream.abort()
        sdk_thread.close.assert_called_once()


def test_sync_interleave_raises_not_implemented():
    adapter, _, _ = _make_sync_adapter()
    with adapter as stream:
        with pytest.raises(NotImplementedError, match="sync interleave"):
            list(stream.interleave("messages"))


def _make_async_adapter(*, run_start_returns=None, run_start_raises=None):
    client = MagicMock()
    client.runs.cancel = AsyncMock()
    sdk_thread = MagicMock()
    sdk_thread.thread_id = "thread-abc"
    sdk_thread.__aenter__ = AsyncMock(return_value=sdk_thread)
    sdk_thread.__aexit__ = AsyncMock(return_value=None)
    sdk_thread.close = AsyncMock()
    sdk_thread.run = MagicMock()
    if run_start_raises is not None:
        sdk_thread.run.start = AsyncMock(side_effect=run_start_raises)
    else:
        sdk_thread.run.start = AsyncMock(
            return_value=run_start_returns or {"run_id": "run-xyz"}
        )
    adapter = _AsyncRemoteGraphRunStream(
        client=client,
        sdk_thread=sdk_thread,
        input={"x": 1},
        config={"configurable": {}},
        metadata=None,
    )
    return adapter, client, sdk_thread


@pytest.mark.anyio
async def test_aenter_calls_sdk_aenter_then_run_start_and_captures_run_id():
    adapter, _, sdk_thread = _make_async_adapter()
    async with adapter as stream:
        assert stream is adapter
        sdk_thread.__aenter__.assert_awaited_once()
        sdk_thread.run.start.assert_awaited_once_with(
            input={"x": 1}, config={"configurable": {}}, metadata=None
        )
        assert adapter._run_id == "run-xyz"


@pytest.mark.anyio
async def test_aexit_delegates_to_sdk_aexit():
    adapter, _, sdk_thread = _make_async_adapter()
    async with adapter:
        pass
    sdk_thread.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.anyio
async def test_aenter_unwinds_sdk_cm_when_run_start_raises():
    adapter, _, sdk_thread = _make_async_adapter(
        run_start_raises=RuntimeError("start boom")
    )
    with pytest.raises(RuntimeError, match="start boom"):
        async with adapter:
            pytest.fail("body should not run")
    sdk_thread.__aenter__.assert_awaited_once()
    sdk_thread.__aexit__.assert_awaited_once()
    assert adapter._run_id is None
