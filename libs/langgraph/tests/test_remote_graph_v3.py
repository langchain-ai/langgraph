from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from langgraph.pregel._remote_run_stream import (
    _AsyncRemoteGraphRunStream,
    _ChannelProjection,
    _ProjectionRegistry,
    _RemoteGraphRunStream,
    _translate_command_input,
)
from langgraph.pregel.remote import (
    _V3_SUPPORTED_KWARGS,
    RemoteGraph,
)
from langgraph.types import Command


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


def test_sync_projection_attrs_forward_to_sdk():
    adapter, _, sdk_thread = _make_sync_adapter()
    sdk_thread.values = object()
    sdk_thread.messages = object()
    sdk_thread.tool_calls = object()
    sdk_thread.subgraphs = object()
    with adapter as stream:
        assert stream.values is sdk_thread.values
        assert stream.messages is sdk_thread.messages
        assert stream.tool_calls is sdk_thread.tool_calls
        assert stream.subgraphs is sdk_thread.subgraphs
        assert set(stream.extensions) == set(_ProjectionRegistry._NATIVE)
        assert stream.extensions["values"] is sdk_thread.values


def test_projection_registry_typed_decoded_and_custom():
    sdk = MagicMock()
    sdk.values = object()
    sdk.messages = object()
    sdk.tool_calls = object()
    sdk.subgraphs = object()
    custom_named = object()
    sdk.extensions = {"my_custom": custom_named}
    registry = _ProjectionRegistry(sdk)

    # Typed channels resolve to the SDK's decoded projections.
    assert registry["values"] is sdk.values
    assert registry["tool_calls"] is sdk.tool_calls
    assert registry["subgraphs"] is sdk.subgraphs
    # Channels without a typed projection resolve to a decoding _ChannelProjection.
    ckpt = registry["checkpoints"]
    assert isinstance(ckpt, _ChannelProjection)
    assert ckpt._channel == "checkpoints"
    assert isinstance(registry["updates"], _ChannelProjection)
    # A non-protocol name is a specific custom-extension channel.
    assert registry["my_custom"] is custom_named
    # Enumerable set is the typed + decoded channels (no `lifecycle`, no `debug`).
    assert list(registry) == [
        "values",
        "messages",
        "tool_calls",
        "subgraphs",
        "updates",
        "checkpoints",
        "tasks",
        "custom",
    ]
    assert len(registry) == 8


def test_channel_projection_decodes_params_data():
    sdk = MagicMock()
    # Real wire events carry `method`; the SDK `DataDecoder` yields matching
    # events' `params.data` and skips dataless and off-channel ones.
    sdk.subscribe = MagicMock(
        return_value=iter(
            [
                {"method": "checkpoints", "params": {"data": {"n": 1}}},
                {"method": "checkpoints", "params": {}},  # no data -> skipped
                {"method": "checkpoints", "params": {"data": {"n": 2}}},
                {"method": "lifecycle", "params": {"data": {"n": 3}}},  # other channel
            ]
        )
    )
    proj = _ChannelProjection(sdk, "checkpoints")
    assert list(proj) == [{"n": 1}, {"n": 2}]
    sdk.subscribe.assert_called_once_with(["checkpoints"])


@pytest.mark.anyio
async def test_channel_projection_decodes_params_data_async():
    """Async lane mirrors the sync lane: `async for` over the SDK's async
    subscription, decoded through the same `DataDecoder`."""

    class _FakeAsyncEvents:
        def __init__(self, items):
            self._items = list(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    sdk = MagicMock()
    sdk.subscribe = MagicMock(
        return_value=_FakeAsyncEvents(
            [
                {"method": "checkpoints", "params": {"data": {"n": 1}}},
                {"method": "checkpoints", "params": {}},  # no data -> skipped
                {"method": "checkpoints", "params": {"data": {"n": 2}}},
                {"method": "lifecycle", "params": {"data": {"n": 3}}},  # other channel
            ]
        )
    )
    proj = _ChannelProjection(sdk, "checkpoints")
    assert [item async for item in proj] == [{"n": 1}, {"n": 2}]
    sdk.subscribe.assert_called_once_with(["checkpoints"])


def test_sync_adapter_translates_command_input():
    sync_client = MagicMock()
    sdk_thread = MagicMock()
    adapter = _RemoteGraphRunStream(
        sync_client=sync_client,
        sdk_thread=sdk_thread,
        input=Command(resume="go"),
        config=None,
        metadata=None,
    )
    assert adapter._start_kwargs["input"] == "go"


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


def test_sync_interleave_delegates_to_interleave_projections():
    adapter, _, sdk_thread = _make_sync_adapter()
    pairs = [("values", {"x": 1}), ("messages", object())]
    sdk_thread.interleave_projections.return_value = pairs
    with adapter as stream:
        result = list(stream.interleave("values", "messages"))
    assert result == pairs
    sdk_thread.interleave_projections.assert_called_once_with(["values", "messages"])


def test_async_adapter_has_no_interleave():
    """Async adapter intentionally lacks `interleave` (mirrors local
    `AsyncGraphRunStream`, which doesn't have one either). Async callers
    compose with `asyncio.gather` / `asyncio.as_completed`.
    """
    assert not hasattr(_AsyncRemoteGraphRunStream, "interleave")


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
        assert await stream.output() == {"foo": 1}
        assert await stream.interrupted() is True
        assert await stream.interrupts() == [
            {"interrupt_id": "i1", "namespace": [], "value": "v"}
        ]


@pytest.mark.anyio
async def test_async_projection_attrs_forward_to_sdk():
    adapter, _, sdk_thread = _make_async_adapter()
    sdk_thread.values = object()
    sdk_thread.messages = object()
    sdk_thread.tool_calls = object()
    sdk_thread.subgraphs = object()
    async with adapter as stream:
        assert stream.values is sdk_thread.values
        assert stream.messages is sdk_thread.messages
        assert stream.tool_calls is sdk_thread.tool_calls
        assert stream.subgraphs is sdk_thread.subgraphs
        assert set(stream.extensions) == set(_ProjectionRegistry._NATIVE)
        assert stream.extensions["messages"] is sdk_thread.messages


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
        client.runs.cancel.assert_awaited_once_with("thread-abc", "run-xyz", wait=False)
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


def test_translate_command_input_surfaces_raw_resume_value():
    # The v3 server wraps the resume `input` as {"resume": input} itself, so the
    # wire `input` must be the raw resume value, not the serialized dataclass.
    assert _translate_command_input(Command(resume="go")) == "go"
    assert _translate_command_input(Command(resume={"id": "v"})) == {"id": "v"}


def test_translate_command_input_rejects_goto_and_update():
    with pytest.raises(NotImplementedError, match="goto"):
        _translate_command_input(Command(goto="node_b"))
    with pytest.raises(NotImplementedError, match="update"):
        _translate_command_input(Command(update={"a": 1}))


def test_translate_command_input_passes_through_non_command():
    assert _translate_command_input({"a": 1}) == {"a": 1}
    assert _translate_command_input(None) is None


def test_v3_supported_kwargs_known_set():
    assert _V3_SUPPORTED_KWARGS == frozenset({"metadata", "headers"})


def test_stream_events_v3_constructs_sdk_thread_with_sanitized_args():
    sync_client = MagicMock()
    sdk_thread = MagicMock()
    sync_client.threads.stream.return_value = sdk_thread
    rg = RemoteGraph(
        "agent",
        client=MagicMock(),
        sync_client=sync_client,
    )
    result = rg.stream_events(
        {"input_key": 1},
        config={"configurable": {"thread_id": "t1", "user": "u"}},
        version="v3",
    )
    assert isinstance(result, _RemoteGraphRunStream)
    sync_client.threads.stream.assert_called_once()
    call = sync_client.threads.stream.call_args
    assert call.kwargs["thread_id"] == "t1"
    assert call.kwargs["assistant_id"] == "agent"
    assert call.kwargs["headers"] is None


def test_stream_events_v3_passes_none_thread_id_when_absent():
    sync_client = MagicMock()
    sync_client.threads.stream.return_value = MagicMock()
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=sync_client)
    rg.stream_events({"x": 1}, version="v3")
    call = sync_client.threads.stream.call_args
    assert call.kwargs["thread_id"] is None


def test_stream_events_v3_rejects_unsupported_kwargs_before_sdk_call():
    sync_client = MagicMock()
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=sync_client)
    with pytest.raises(NotImplementedError, match="control"):
        rg.stream_events({"x": 1}, version="v3", control=object())
    sync_client.threads.stream.assert_not_called()


def test_stream_events_v3_translates_command_input():
    sync_client = MagicMock()
    sync_client.threads.stream.return_value = MagicMock()
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=sync_client)
    # Resume Command surfaces its raw resume value as the wire `input`; the v3
    # server wraps it as {"resume": input} once it detects the interrupt.
    adapter = rg.stream_events(Command(resume="go"), version="v3")
    assert adapter._start_kwargs["input"] == "go"


def test_stream_events_v3_rejects_goto_update_command():
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=MagicMock())
    with pytest.raises(NotImplementedError, match="goto"):
        rg.stream_events(Command(goto="node_b"), version="v3")


def test_stream_events_v3_strips_checkpoint_keys_from_configurable():
    sync_client = MagicMock()
    sync_client.threads.stream.return_value = MagicMock()
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=sync_client)
    adapter = rg.stream_events(
        {"x": 1},
        config={
            "configurable": {
                "thread_id": "t1",
                "checkpoint_id": "c1",
                "checkpoint_ns": "ns",
                "user": "u",
            }
        },
        version="v3",
    )
    sent_config = adapter._start_kwargs["config"]
    assert "checkpoint_id" not in sent_config["configurable"]
    assert "checkpoint_ns" not in sent_config["configurable"]
    assert sent_config["configurable"]["user"] == "u"


def test_stream_events_v3_merges_tracing_headers_when_distributed_tracing(
    monkeypatch,
):
    from langgraph.pregel import remote as remote_mod

    sync_client = MagicMock()
    sync_client.threads.stream.return_value = MagicMock()
    rg = RemoteGraph(
        "agent",
        client=MagicMock(),
        sync_client=sync_client,
        distributed_tracing=True,
    )
    captured = {}

    def fake_merge(headers):
        captured["arg"] = headers
        return {"x-ls-trace": "1", **(headers or {})}

    monkeypatch.setattr(remote_mod, "_merge_tracing_headers", fake_merge)
    rg.stream_events({"x": 1}, version="v3", headers={"X-Custom": "y"})
    assert captured["arg"] == {"X-Custom": "y"}
    sent_headers = sync_client.threads.stream.call_args.kwargs["headers"]
    assert sent_headers["x-ls-trace"] == "1"
    assert sent_headers["X-Custom"] == "y"


def test_stream_events_v3_passes_headers_unchanged_without_tracing():
    sync_client = MagicMock()
    sync_client.threads.stream.return_value = MagicMock()
    rg = RemoteGraph(
        "agent",
        client=MagicMock(),
        sync_client=sync_client,
        distributed_tracing=False,
    )
    rg.stream_events({"x": 1}, version="v3", headers={"X-Custom": "y"})
    sent_headers = sync_client.threads.stream.call_args.kwargs["headers"]
    assert sent_headers == {"X-Custom": "y"}


def test_stream_events_non_v3_delegates_to_super():
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=MagicMock())
    sync_client_attr = rg.sync_client
    try:
        rg.stream_events({"x": 1}, version="v2")
    except Exception:
        pass
    sync_client_attr.threads.stream.assert_not_called()


@pytest.mark.anyio
async def test_astream_events_v3_constructs_sdk_thread():
    client = MagicMock()
    sdk_thread = MagicMock()
    client.threads.stream.return_value = sdk_thread
    rg = RemoteGraph("agent", client=client, sync_client=MagicMock())
    result = await rg.astream_events(
        {"x": 1},
        config={"configurable": {"thread_id": "t1"}},
        version="v3",
    )
    assert isinstance(result, _AsyncRemoteGraphRunStream)
    call = client.threads.stream.call_args
    assert call.kwargs["thread_id"] == "t1"
    assert call.kwargs["assistant_id"] == "agent"


@pytest.mark.anyio
async def test_astream_events_v3_rejects_unsupported_kwargs():
    client = MagicMock()
    rg = RemoteGraph("agent", client=client, sync_client=MagicMock())
    with pytest.raises(NotImplementedError, match="transformers"):
        await rg.astream_events({"x": 1}, version="v3", transformers=[object()])
    client.threads.stream.assert_not_called()


@pytest.mark.anyio
async def test_astream_events_non_v3_raises_not_implemented():
    rg = RemoteGraph("agent", client=MagicMock(), sync_client=MagicMock())
    with pytest.raises(NotImplementedError, match="not implemented"):
        await rg.astream_events({"x": 1}, version="v2")
