import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from langgraph.stream._mux import AsyncStreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.chat_model_stream import ChatModelStream
from langgraph.stream.run_stream import (
    AsyncGraphRunStream,
    AsyncSubgraphRunStream,
    SubgraphRunStream,
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


def test_sync_messages_text_streaming():
    """Sync consumers can iterate msg.text for deltas."""
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

    # Iterate deltas
    run = create_graph_run_stream(_sync_source(chunks))
    for msg in run.messages:
        deltas = list(msg.text)
        assert deltas == ["Hello", " world"]
        assert msg.done

    # str() returns full text
    run = create_graph_run_stream(_sync_source(chunks))
    for msg in run.messages:
        assert str(msg.text) == "Hello world"

    # After message is done, .text returns plain str
    run = create_graph_run_stream(_sync_source(chunks))
    for msg in run.messages:
        list(msg.text)  # exhaust deltas
        assert isinstance(msg.text, str)
        assert msg.text == "Hello world"


def test_sync_messages_multiple():
    """Multiple sync messages each stream their own deltas."""
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

    all_deltas = []
    for msg in run.messages:
        all_deltas.append(list(msg.text))
    assert all_deltas == [["answer"], ["second"]]


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

    # Pull first message — yielded on message-start (item 2).
    # Consuming str(msg.text) drives the pump to message-finish (item 3).
    mit = iter(run.messages)
    msg = next(mit)
    assert isinstance(msg, ChatModelStream)
    assert consumed == 2
    assert not msg.done
    str(msg.text)  # pump until message completes
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


# ---------------------------------------------------------------------------
# SubgraphRunStream — sync subgraph tests
# ---------------------------------------------------------------------------


def test_sync_subgraphs_discovery():
    """Iterating .subgraphs should discover child namespaces and yield
    SubgraphRunStream instances with correct name and index.
    """
    chunks = [
        (("agent:0",), "values", {"v": 1}),
        (("agent:1",), "values", {"v": 2}),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    subs = list(run.subgraphs)
    assert len(subs) == 2
    assert all(isinstance(s, SubgraphRunStream) for s in subs)
    assert subs[0].name == "agent"
    assert subs[0].index == 0
    assert subs[1].name == "agent"
    assert subs[1].index == 1


def test_sync_subgraph_name_no_index():
    """Subgraph without a colon-delimited index should have index=0."""
    chunks = [
        (("planner",), "values", {"v": 1}),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    subs = list(run.subgraphs)
    assert len(subs) == 1
    assert subs[0].name == "planner"
    assert subs[0].index == 0


def test_sync_subgraph_no_subgraphs():
    """When all events are root-level, .subgraphs should yield nothing."""
    chunks = [
        ((), "values", {"v": 1}),
        ((), "values", {"v": 2}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    subs = list(run.subgraphs)
    assert subs == []


def test_sync_subgraph_values():
    """SubgraphRunStream.values should yield only values from the child namespace."""
    chunks = [
        (("child:0",), "values", {"v": 1}),
        ((), "values", {"root": True}),
        (("child:0",), "values", {"v": 2}),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    for sub in run.subgraphs:
        vals = list(sub.values)
        assert vals == [{"v": 1}, {"v": 2}]


def test_sync_subgraph_values_multiple_children():
    """Each child stream should only see its own values."""
    chunks = [
        (("a:0",), "values", {"who": "a0"}),
        (("b:0",), "values", {"who": "b0"}),
        (("a:0",), "values", {"who": "a0-2"}),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    children: dict[str, list[Any]] = {}
    for sub in run.subgraphs:
        children[f"{sub.name}:{sub.index}"] = list(sub.values)

    assert children["a:0"] == [{"who": "a0"}, {"who": "a0-2"}]
    assert children["b:0"] == [{"who": "b0"}]


def test_sync_subgraph_output():
    """SubgraphRunStream.output should return the last values for the child."""
    chunks = [
        (("child:0",), "values", {"v": 1}),
        (("child:0",), "values", {"v": 2}),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    for sub in run.subgraphs:
        assert sub.output == {"v": 2}


def test_sync_subgraph_output_with_mapper():
    """Output mapper should apply to subgraph output."""
    chunks = [
        (("child:0",), "values", {"v": 42}),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(
        _sync_source(chunks), output_mapper=lambda x: {"mapped": x.get("v")}
    )
    for sub in run.subgraphs:
        assert sub.output == {"mapped": 42}


def test_sync_subgraph_values_with_mapper():
    """Output mapper should apply to each yielded value snapshot."""
    chunks = [
        (("child:0",), "values", {"v": 1}),
        (("child:0",), "values", {"v": 2}),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(
        _sync_source(chunks), output_mapper=lambda x: {"m": x.get("v")}
    )
    for sub in run.subgraphs:
        vals = list(sub.values)
        assert vals == [{"m": 1}, {"m": 2}]


def test_sync_subgraph_messages():
    """SubgraphRunStream.messages should yield fully populated ChatModelStream instances."""
    chunks = [
        (
            ("agent:0",),
            "messages",
            {"event": "message-start", "message_id": "m1", "__node__": "agent"},
        ),
        (
            ("agent:0",),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": "hello"},
                "__node__": "agent",
            },
        ),
        (
            ("agent:0",),
            "messages",
            {"event": "message-finish", "reason": "stop", "__node__": "agent"},
        ),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    for sub in run.subgraphs:
        msgs = list(sub.messages)
        assert len(msgs) == 1
        assert isinstance(msgs[0], ChatModelStream)
        assert msgs[0].done
        assert msgs[0].text == "hello"


def test_sync_subgraph_messages_isolated():
    """Messages from different subgraphs should not leak between children."""
    chunks = [
        (
            ("a:0",),
            "messages",
            {"event": "message-start", "message_id": "m-a", "__node__": "a"},
        ),
        (
            ("a:0",),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": "from-a"},
                "__node__": "a",
            },
        ),
        (
            ("a:0",),
            "messages",
            {"event": "message-finish", "reason": "stop", "__node__": "a"},
        ),
        (
            ("b:0",),
            "messages",
            {"event": "message-start", "message_id": "m-b", "__node__": "b"},
        ),
        (
            ("b:0",),
            "messages",
            {
                "event": "content-block-delta",
                "content_block": {"type": "text", "text": "from-b"},
                "__node__": "b",
            },
        ),
        (
            ("b:0",),
            "messages",
            {"event": "message-finish", "reason": "stop", "__node__": "b"},
        ),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    msg_texts: dict[str, list[str]] = {}
    for sub in run.subgraphs:
        msg_texts[sub.name] = [str(m.text) for m in sub.messages]

    assert msg_texts["a"] == ["from-a"]
    assert msg_texts["b"] == ["from-b"]


def test_sync_subgraph_raw_iter():
    """Iterating a SubgraphRunStream directly should yield events scoped
    to the child namespace.
    """
    chunks = [
        (("child:0",), "values", {"v": 1}),
        ((), "values", {"root": True}),
        (("child:0",), "updates", {"node": "x"}),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    for sub in run.subgraphs:
        events = list(sub)
        methods = [e["method"] for e in events]
        assert "values" in methods
        assert "updates" in methods
        # Root events should not appear
        for e in events:
            assert e["params"]["namespace"] == ["child:0"]


def test_sync_subgraph_events_after_discovery():
    """Events arriving after a namespace is first discovered should still
    be visible in the child's values iteration.
    """
    chunks = [
        (("child:0",), "values", {"v": 1}),  # triggers discovery
        ((), "values", {"root": 1}),
        (("child:0",), "values", {"v": 2}),  # after discovery
        (("child:0",), "values", {"v": 3}),  # after discovery
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    for sub in run.subgraphs:
        vals = list(sub.values)
        assert vals == [{"v": 1}, {"v": 2}, {"v": 3}]


def test_sync_subgraph_lazy_pump():
    """Subgraph iteration should pump the source lazily."""
    consumed = 0

    def counting_source():
        nonlocal consumed
        for chunk in [
            (("child:0",), "values", {"v": 1}),
            (("child:0",), "values", {"v": 2}),
            (("child:0",), "values", {"v": 3}),
            ((), "values", {"done": True}),
        ]:
            consumed += 1
            yield chunk

    run = create_graph_run_stream(counting_source())
    assert consumed == 0

    for sub in run.subgraphs:
        # Discovery pumped the first event
        it = iter(sub.values)
        v = next(it)
        assert v == {"v": 1}
        # Should not have consumed everything yet
        assert consumed < 4
        break  # don't exhaust subgraphs


def test_sync_subgraph_interleave_parent_values():
    """Parent values and subgraph values should both be accessible
    when interleaving iteration.
    """
    chunks = [
        ((), "values", {"root": 1}),
        (("child:0",), "values", {"child": 1}),
        ((), "values", {"root": 2}),
        (("child:0",), "values", {"child": 2}),
        ((), "values", {"root": 3}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))

    # First drain parent values
    root_vals = list(run.values)
    assert root_vals == [{"root": 1}, {"root": 2}, {"root": 3}]

    # Source is exhausted, but subgraph transformers were registered
    # via replay — subgraph iteration should still see buffered events
    # Note: subgraphs must be iterated while source is being pumped
    # to discover namespaces. Since we drained via values, namespace
    # "child:0" was already discovered. But subgraphs iteration also
    # needs to pump — and the source is exhausted. Let's verify it
    # yields the discovered child.
    subs = list(run.subgraphs)
    assert len(subs) == 1
    assert subs[0].name == "child"
    # The child transformer was registered via replay, so it saw the events
    vals = list(subs[0].values)
    assert vals == [{"child": 1}, {"child": 2}]


def test_sync_subgraph_interrupted():
    """Subgraph .interrupted should reflect the mux's interrupt state."""

    class _FakeInterrupt:
        def __init__(self, id: str):
            self.id = id

    chunks = [
        (("child:0",), "values", {"__interrupt__": [_FakeInterrupt("i1")]}),
        ((), "values", {"done": True}),
    ]
    run = create_graph_run_stream(_sync_source(chunks))
    for sub in run.subgraphs:
        # Pump to process the interrupt
        _ = sub.output
        assert sub.interrupted is True
        assert len(sub.interrupts) == 1


def test_sync_subgraph_source_error():
    """If the source raises mid-stream, subgraphs that were already
    discovered should still have their buffered data.
    """

    def bad_source():
        yield (("child:0",), "values", {"v": 1})
        yield (("child:0",), "values", {"v": 2})
        raise ValueError("boom")

    run = create_graph_run_stream(bad_source())
    for sub in run.subgraphs:
        vals = list(sub.values)
        assert vals == [{"v": 1}, {"v": 2}]
        assert run._mux._error is not None


def test_sync_subgraph_output_drains_source():
    """Accessing subgraph .output should drain the full source."""
    consumed = 0

    def counting_source():
        nonlocal consumed
        for chunk in [
            (("child:0",), "values", {"v": 1}),
            (("child:0",), "values", {"v": 2}),
            ((), "values", {"done": True}),
        ]:
            consumed += 1
            yield chunk

    run = create_graph_run_stream(counting_source())
    for sub in run.subgraphs:
        result = sub.output
        assert result == {"v": 2}
        assert consumed == 3
