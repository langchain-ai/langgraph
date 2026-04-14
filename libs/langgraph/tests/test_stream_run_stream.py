import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from langgraph.stream._mux import AsyncStreamMux, StreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.chat_model_stream import ChatModelStream
from langgraph.stream.run_stream import (
    AsyncGraphRunStream,
    AsyncSubgraphRunStream,
    create_async_graph_run_stream,
    create_graph_run_stream,
)
from langgraph.stream.transformers import MessagesTransformer, ValuesTransformer


async def _mock_source(
    chunks: list[tuple[tuple[str, ...], str, Any]],
) -> AsyncIterator[tuple[tuple[str, ...], str, Any]]:
    for chunk in chunks:
        yield chunk


@pytest.mark.anyio
async def test_aiter_yields_all_events():
    chunks = [
        ((), "values", {"step": 1}),
        ((), "values", {"step": 2}),
        ((), "updates", {"node": "a"}),
    ]
    run = await create_async_graph_run_stream(_mock_source(chunks))
    await asyncio.sleep(0.05)

    collected: list[ProtocolEvent] = []
    async for event in run:
        collected.append(event)
    assert len(collected) == 3
    assert collected[0]["method"] == "values"
    assert collected[2]["method"] == "updates"


@pytest.mark.anyio
async def test_subgraph_name_and_index():
    vr, mr = ValuesTransformer(), MessagesTransformer()
    mux = AsyncStreamMux(transformers=[vr, mr])
    sub = AsyncSubgraphRunStream(
        mux=mux,
        namespace=["researcher:2"],
        transformers=[vr, mr],
    )
    assert sub.name == "researcher"
    assert sub.index == 2


@pytest.mark.anyio
async def test_subgraph_name_no_index():
    vr, mr = ValuesTransformer(), MessagesTransformer()
    mux = AsyncStreamMux(transformers=[vr, mr])
    sub = AsyncSubgraphRunStream(
        mux=mux, namespace=["agent"], transformers=[vr, mr]
    )
    assert sub.name == "agent"
    assert sub.index == 0


@pytest.mark.anyio
async def test_values_iterable():
    chunks = [
        ((), "values", {"v": 1}),
        ((), "values", {"v": 2}),
    ]
    run = await create_async_graph_run_stream(_mock_source(chunks))
    await asyncio.sleep(0.05)

    collected = []
    async for v in run.values:
        collected.append(v)
    assert len(collected) == 2
    assert collected[0] == {"v": 1}
    assert collected[1] == {"v": 2}


@pytest.mark.anyio
async def test_values_awaitable():
    chunks = [
        ((), "values", {"v": 1}),
        ((), "values", {"v": 2}),
    ]
    run = await create_async_graph_run_stream(_mock_source(chunks))
    await asyncio.sleep(0.05)

    result = await run.values
    assert result == {"v": 2}


@pytest.mark.anyio
async def test_output_resolves():
    chunks = [((), "values", {"final": True})]
    run = await create_async_graph_run_stream(_mock_source(chunks))
    await asyncio.sleep(0.05)
    result = await run.output
    assert result == {"final": True}


@pytest.mark.anyio
async def test_messages_yields_streams():
    chunks = [
        ((), "messages", {"event": "message-start", "message_id": "m1"}),
        (
            (),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": "hi"},
            },
        ),
        ((), "messages", {"event": "message-finish", "reason": "stop"}),
    ]
    run = await create_async_graph_run_stream(_mock_source(chunks))
    await asyncio.sleep(0.05)

    collected: list[ChatModelStream] = []
    async for stream in run.messages:
        collected.append(stream)
    assert len(collected) == 1
    assert isinstance(collected[0], ChatModelStream)
    assert collected[0].done


@pytest.mark.anyio
async def test_interrupted_false_by_default():
    vr, mr = ValuesTransformer(), MessagesTransformer()
    mux = AsyncStreamMux(transformers=[vr, mr])
    run = AsyncGraphRunStream(mux=mux, transformers=[vr, mr])
    assert run.interrupted is False


@pytest.mark.anyio
async def test_abort_sets_signal():
    vr, mr = ValuesTransformer(), MessagesTransformer()
    mux = AsyncStreamMux(transformers=[vr, mr])
    run = AsyncGraphRunStream(mux=mux, transformers=[vr, mr])
    assert not run.signal.is_set()
    run.abort()
    assert run.signal.is_set()


@pytest.mark.anyio
async def test_abort_stops_pump():
    """Calling abort() should stop the pump from processing further chunks."""
    gate = asyncio.Event()

    async def _gated_source():
        yield ((), "values", {"v": 1})
        yield ((), "values", {"v": 2})
        await gate.wait()  # Block until released
        yield ((), "values", {"v": 3})  # Should not be processed

    run = await create_async_graph_run_stream(_gated_source())
    await asyncio.sleep(0.05)  # Let first two events through
    run.abort()
    gate.set()  # Unblock the source so the pump can check abort and exit
    await asyncio.sleep(0.05)  # Let pump close the mux

    collected = []
    async for event in run:
        if event["method"] == "values":
            collected.append(event["params"]["data"])
    # v:3 should not have been processed because abort was set
    assert all(v.get("v") != 3 for v in collected)


@pytest.mark.anyio
async def test_messages_from_filters_by_node():
    """messages_from(node) should only yield messages from the specified node."""
    chunks = [
        (
            (),
            "messages",
            {"event": "message-start", "message_id": "m1", "__node__": "agent"},
        ),
        (
            (),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": "from agent"},
                "__node__": "agent",
            },
        ),
        (
            (),
            "messages",
            {"event": "message-finish", "reason": "stop", "__node__": "agent"},
        ),
        (
            (),
            "messages",
            {"event": "message-start", "message_id": "m2", "__node__": "tools"},
        ),
        (
            (),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": "from tools"},
                "__node__": "tools",
            },
        ),
        (
            (),
            "messages",
            {"event": "message-finish", "reason": "stop", "__node__": "tools"},
        ),
    ]
    run = await create_async_graph_run_stream(_mock_source(chunks))
    await asyncio.sleep(0.05)

    agent_msgs: list[ChatModelStream] = []
    async for stream in run.messages_from("agent"):
        agent_msgs.append(stream)
    assert len(agent_msgs) == 1
    assert agent_msgs[0].node == "agent"


# ---------------------------------------------------------------------------
# GraphRunStream / create_graph_run_stream
# ---------------------------------------------------------------------------


def _sync_source(
    chunks: list[tuple[tuple[str, ...], str, Any]],
) -> Iterator[tuple[tuple[str, ...], str, Any]]:
    yield from chunks


def test_sync_create_yields_all_events():
    chunks = [
        ((), "values", {"step": 1}),
        ((), "values", {"step": 2}),
        ((), "updates", {"node": "a"}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    collected = list(run)
    assert len(collected) == 3
    assert collected[0]["method"] == "values"
    assert collected[2]["method"] == "updates"


def test_sync_output():
    chunks = [
        ((), "values", {"v": 1}),
        ((), "values", {"v": 2}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    assert run.output == {"v": 2}


def test_sync_values_iteration():
    chunks = [
        ((), "values", {"v": 1}),
        ((), "values", {"v": 2}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    collected = list(run.values)
    assert len(collected) == 2
    assert collected[0] == {"v": 1}
    assert collected[1] == {"v": 2}


def test_sync_messages():
    chunks = [
        ((), "messages", {"event": "message-start", "message_id": "m1"}),
        (
            (),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": "hi"},
            },
        ),
        ((), "messages", {"event": "message-finish", "reason": "stop"}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    collected = list(run.messages)
    assert len(collected) == 1
    assert isinstance(collected[0], ChatModelStream)
    assert collected[0].done


def test_sync_messages_text_accessible():
    """Sync consumers should be able to read ChatModelStream text content
    without an async event loop.
    """
    chunks = [
        ((), "messages", {"event": "message-start", "message_id": "m1"}),
        (
            (),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": "Hello"},
            },
        ),
        (
            (),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": " world"},
            },
        ),
        ((), "messages", {"event": "message-finish", "reason": "stop"}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))

    for msg in run.messages:
        assert isinstance(msg.text, str)
        assert msg.text == "Hello world"
        assert msg.done


def test_sync_messages_content_populated_when_yielded():
    """When sync run.messages yields a ChatModelStream, its content should
    be fully populated (done=True) with all text accumulated.
    """
    chunks = [
        ((), "messages", {"event": "message-start", "message_id": "m1"}),
        (
            (),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": "answer"},
            },
        ),
        ((), "messages", {"event": "message-finish", "reason": "stop"}),
        ((), "messages", {"event": "message-start", "message_id": "m2"}),
        (
            (),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": "second"},
            },
        ),
        ((), "messages", {"event": "message-finish", "reason": "stop"}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))

    messages = list(run.messages)
    assert len(messages) == 2
    assert messages[0].done
    assert messages[0].text == "answer"
    assert messages[1].done
    assert messages[1].text == "second"


def test_sync_output_mapper():
    chunks = [((), "values", {"v": 1})]
    run = create_graph_run_stream(
        _sync_source(chunks), output_mapper=lambda x: {"mapped": x["v"]}
    )
    assert run.output == {"mapped": 1}


def test_sync_interrupted_false():
    chunks = [((), "values", {"v": 1})]
    run = create_graph_run_stream(_sync_source(chunks))
    assert run.interrupted is False


def test_sync_source_error():
    """If the source raises, the mux should fail and the error should propagate."""

    def _bad_source():
        yield ((), "values", {"v": 1})
        raise ValueError("source error")

    run = create_graph_run_stream(_bad_source())
    collected = list(run)
    # Events before the error are still accessible
    assert len(collected) >= 1
    assert collected[0]["method"] == "values"
    # The mux recorded the failure
    assert run._mux._error is not None
    assert isinstance(run._mux._error, ValueError)
    assert "source error" in str(run._mux._error)


# ---------------------------------------------------------------------------
# GraphRunStream — lazy consumption tests
# ---------------------------------------------------------------------------


def test_sync_lazy_not_consumed_on_creation():
    """Source iterator should not be consumed when the stream is created."""
    consumed = 0

    def counting_source():
        nonlocal consumed
        for chunk in [
            ((), "values", {"v": 1}),
            ((), "values", {"v": 2}),
            ((), "values", {"v": 3}),
        ]:
            consumed += 1
            yield chunk

    create_graph_run_stream(counting_source())
    assert consumed == 0


def test_sync_lazy_values_pull_incrementally():
    """Iterating .values should pull from the source one event at a time."""
    consumed = 0

    def counting_source():
        nonlocal consumed
        for chunk in [
            ((), "values", {"v": 1}),
            ((), "values", {"v": 2}),
            ((), "values", {"v": 3}),
        ]:
            consumed += 1
            yield chunk

    run = create_graph_run_stream(counting_source())
    assert consumed == 0

    it = iter(run.values)
    v = next(it)
    assert v == {"v": 1}
    assert consumed == 1

    v = next(it)
    assert v == {"v": 2}
    assert consumed == 2

    # Source not fully drained yet
    assert consumed < 3


def test_sync_lazy_output_drains_all():
    """Accessing .output should drain the entire source."""
    consumed = 0

    def counting_source():
        nonlocal consumed
        for chunk in [((), "values", {"v": i}) for i in range(5)]:
            consumed += 1
            yield chunk

    run = create_graph_run_stream(counting_source())
    assert consumed == 0
    assert run.output == {"v": 4}
    assert consumed == 5


def test_sync_lazy_early_break():
    """Breaking out of a projection early should leave the source partially consumed."""
    consumed = 0

    def counting_source():
        nonlocal consumed
        for chunk in [((), "values", {"v": i}) for i in range(10)]:
            consumed += 1
            yield chunk

    run = create_graph_run_stream(counting_source())
    for v in run.values:
        break  # consume only the first value
    assert consumed == 1
    assert consumed < 10


def test_sync_lazy_interleaved_projections():
    """Switching between projections replays buffered items then resumes pumping."""
    consumed = 0

    def counting_source():
        nonlocal consumed
        for chunk in [
            ((), "values", {"v": 1}),
            ((), "messages", {"event": "message-start", "message_id": "m1"}),
            ((), "messages", {"event": "message-finish", "reason": "stop"}),
            ((), "values", {"v": 2}),
        ]:
            consumed += 1
            yield chunk

    run = create_graph_run_stream(counting_source())

    # Pull first value — consumes 1 source item
    vit = iter(run.values)
    assert next(vit) == {"v": 1}
    assert consumed == 1

    # Pull first message — pumps until message-finish (item 3) so the
    # ChatModelStream is fully populated before yielding.
    mit = iter(run.messages)
    msg = next(mit)
    assert isinstance(msg, ChatModelStream)
    assert msg.done
    assert consumed == 3

    # Pull second value — pumps values (item 4)
    assert next(vit) == {"v": 2}
    assert consumed == 4


def test_sync_lazy_iter_pulls_incrementally():
    """Raw __iter__ should pull from the source lazily."""
    consumed = 0

    def counting_source():
        nonlocal consumed
        for chunk in [
            ((), "values", {"v": 1}),
            ((), "updates", {"node": "a"}),
            ((), "values", {"v": 2}),
        ]:
            consumed += 1
            yield chunk

    run = create_graph_run_stream(counting_source())
    it = iter(run)
    event = next(it)
    assert event["method"] == "values"
    assert consumed == 1

    event = next(it)
    assert event["method"] == "updates"
    assert consumed == 2


def test_sync_lazy_source_error():
    """If the source raises mid-stream, earlier events are still accessible."""
    consumed = 0

    def bad_source():
        nonlocal consumed
        consumed += 1
        yield ((), "values", {"v": 1})
        raise ValueError("boom")

    run = create_graph_run_stream(bad_source())
    collected = list(run)
    assert len(collected) >= 1
    assert collected[0]["method"] == "values"


@pytest.mark.anyio
async def test_subgraph_child_values_receive_post_discovery_events():
    """Child AsyncSubgraphRunStream.values iteration should include events
    that arrive AFTER the subgraph namespace is first discovered.

    ``_SubgraphsProjection`` creates a local ``ValuesTransformer`` for
    each child and replays existing events, but never registers the
    transformer with the mux.  Events that arrive after discovery are
    not routed to it, and ``finalize()`` is not called (the mux wasn't
    closed at discovery time), so the child's values_log is never
    closed and iteration hangs.
    """
    gate = asyncio.Event()

    async def _source() -> AsyncIterator[tuple[tuple[str, ...], str, Any]]:
        # First event from child namespace — triggers discovery
        yield (("child:0",), "values", {"v": 1})
        await gate.wait()
        # Second event from same child — arrives after discovery
        yield (("child:0",), "values", {"v": 2})
        # Root event so the mux tracks output
        yield ((), "values", {"done": True})

    run = await create_async_graph_run_stream(_source())
    await asyncio.sleep(0.05)  # let pump process first event

    # Get the first subgraph while the mux is still open
    sub = None
    async for s in run.subgraphs:
        sub = s
        break

    assert sub is not None

    # Release the gate so the pump finishes
    gate.set()
    await asyncio.sleep(0.05)  # let pump close mux

    # ``await sub.output`` uses the mux's output future — works fine
    output = await sub.output
    assert output == {"v": 2}, "await sub.output should reflect the latest value"

    # But ``async for v in sub.values`` only gets the replayed event
    # and then hangs because the child's values_log is never closed.
    values: list[Any] = []
    try:
        async with asyncio.timeout(1.0):
            async for v in sub.values:
                values.append(v)
    except (asyncio.TimeoutError, TimeoutError):
        pass

    assert len(values) == 2, (
        f"Expected 2 child value snapshots but got {len(values)}: {values}.  "
        "Child transformer missed post-discovery events."
    )
