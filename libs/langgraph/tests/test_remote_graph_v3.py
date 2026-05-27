from __future__ import annotations

import asyncio
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


@pytest.mark.anyio
async def test_async_output_interrupted_interrupts_passthrough():
    adapter, _, sdk_thread = _make_async_adapter()

    async def _fake_output_awaitable():
        return {"foo": 1}

    sdk_thread.output = _fake_output_awaitable()
    sdk_thread.interrupted = True
    sdk_thread.interrupts = [{"interrupt_id": "i1", "namespace": [], "value": "v"}]
    async with adapter as stream:
        assert await stream.output == {"foo": 1}
        assert await stream.interrupted is True
        assert await stream.interrupts == [
            {"interrupt_id": "i1", "namespace": [], "value": "v"}
        ]


@pytest.mark.anyio
async def test_aiter_caches_first_subscription():
    adapter, _, sdk_thread = _make_async_adapter()

    class _FakeAsyncEvents:
        def __init__(self, items):
            self._items = list(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    sdk_thread.events = _FakeAsyncEvents([object(), object()])
    async with adapter as stream:
        first = stream.__aiter__()
        second = stream.__aiter__()
        assert first is second


@pytest.mark.anyio
async def test_async_abort_cancels_run_and_closes_sdk():
    adapter, client, sdk_thread = _make_async_adapter()
    async with adapter as stream:
        await stream.abort()
        client.runs.cancel.assert_awaited_once_with(
            "thread-abc", "run-xyz", wait=False
        )
        sdk_thread.close.assert_awaited_once()


@pytest.mark.anyio
async def test_async_abort_before_aenter_skips_cancel():
    adapter, client, sdk_thread = _make_async_adapter()
    await adapter.abort()
    client.runs.cancel.assert_not_awaited()
    sdk_thread.close.assert_awaited_once()


@pytest.mark.anyio
async def test_async_abort_swallows_cancel_failure():
    adapter, client, sdk_thread = _make_async_adapter()
    client.runs.cancel.side_effect = RuntimeError("cancel boom")
    async with adapter as stream:
        await stream.abort()
        sdk_thread.close.assert_awaited_once()


class _FakeAsyncIter:
    def __init__(self, items, *, sleeps=None):
        self._items = list(items)
        self._sleeps = list(sleeps) if sleeps else [0.0] * len(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        delay = self._sleeps.pop(0)
        if delay:
            await asyncio.sleep(delay)
        return self._items.pop(0)


@pytest.mark.anyio
async def test_interleave_yields_tagged_tuples():
    adapter, _, sdk_thread = _make_async_adapter()
    sdk_thread.messages = _FakeAsyncIter(["m1", "m2"])
    sdk_thread.values = _FakeAsyncIter([{"v": 1}])
    async with adapter as stream:
        got = []
        async for name, item in stream.interleave("messages", "values"):
            got.append((name, item))
    names = [n for n, _ in got]
    assert sorted(names) == ["messages", "messages", "values"]


@pytest.mark.anyio
async def test_interleave_routes_unknown_through_extensions():
    adapter, _, sdk_thread = _make_async_adapter()
    sdk_thread.extensions = {"custom": _FakeAsyncIter(["c1", "c2"])}
    async with adapter as stream:
        got = [
            (name, item)
            async for name, item in stream.interleave("custom")
        ]
    assert got == [("custom", "c1"), ("custom", "c2")]


@pytest.mark.anyio
async def test_interleave_no_names_yields_nothing():
    adapter, _, _ = _make_async_adapter()
    async with adapter as stream:
        got = [
            (name, item)
            async for name, item in stream.interleave()
        ]
    assert got == []


@pytest.mark.anyio
async def test_interleave_cancels_drainers_on_early_break():
    adapter, _, sdk_thread = _make_async_adapter()
    sdk_thread.messages = _FakeAsyncIter(
        ["m1", "m2", "m3", "m4"], sleeps=[0.0, 0.05, 0.05, 0.05]
    )
    sdk_thread.values = _FakeAsyncIter(
        [{"v": 1}, {"v": 2}], sleeps=[0.0, 0.05]
    )
    async with adapter as stream:
        async for _name, _item in stream.interleave("messages", "values"):
            break
    # Reaching here without hanging or leaking tasks is the success condition.


from dataclasses import asdict

from langgraph.pregel.remote import (
    RemoteGraph,
    _translate_command_input,
    _V3_SUPPORTED_KWARGS,
)
from langgraph.types import Command


def _make_remote_graph() -> RemoteGraph:
    sync_client = MagicMock()
    async_client = MagicMock()
    rg = RemoteGraph(
        "agent",
        client=async_client,
        sync_client=sync_client,
    )
    return rg


def test_reject_v3_unsupported_passes_when_all_clear():
    rg = _make_remote_graph()
    rg._reject_v3_unsupported(
        control=None,
        transformers=None,
        interrupt_before=None,
        interrupt_after=None,
        extra_kwargs={},
    )


@pytest.mark.parametrize(
    "kwarg_name,kwarg_value",
    [
        ("control", object()),
        ("transformers", [object()]),
        ("interrupt_before", ["node_a"]),
        ("interrupt_after", ["node_b"]),
    ],
)
def test_reject_v3_unsupported_raises_per_kwarg(kwarg_name, kwarg_value):
    rg = _make_remote_graph()
    kwargs = dict(
        control=None,
        transformers=None,
        interrupt_before=None,
        interrupt_after=None,
        extra_kwargs={},
    )
    kwargs[kwarg_name] = kwarg_value
    with pytest.raises(NotImplementedError, match=f"`{kwarg_name}=`"):
        rg._reject_v3_unsupported(**kwargs)


def test_reject_v3_unsupported_raises_on_unknown_extra_kwarg():
    rg = _make_remote_graph()
    with pytest.raises(NotImplementedError, match="context"):
        rg._reject_v3_unsupported(
            control=None,
            transformers=None,
            interrupt_before=None,
            interrupt_after=None,
            extra_kwargs={"context": {}},
        )


def test_reject_v3_unsupported_allows_metadata_and_headers():
    rg = _make_remote_graph()
    rg._reject_v3_unsupported(
        control=None,
        transformers=None,
        interrupt_before=None,
        interrupt_after=None,
        extra_kwargs={"metadata": {"a": 1}, "headers": {"X": "y"}},
    )


def test_translate_command_input_converts_command_to_dict():
    cmd = Command(update={"a": 1})
    assert _translate_command_input(cmd) == asdict(cmd)


def test_translate_command_input_passes_through_non_command():
    assert _translate_command_input({"a": 1}) == {"a": 1}
    assert _translate_command_input(None) is None


def test_v3_supported_kwargs_known_set():
    assert _V3_SUPPORTED_KWARGS == frozenset({"metadata", "headers"})
