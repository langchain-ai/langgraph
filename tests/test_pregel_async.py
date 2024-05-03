import asyncio
import json
import operator
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Optional,
    TypedDict,
    Union,
)
from uuid import UUID

import pytest
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion

from langgraph.channels.base import InvalidUpdateError
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.context import Context
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.checkpoint.base import CheckpointAt
from langgraph.graph import END, Graph, StateGraph
from langgraph.graph.message import MessageGraph
from langgraph.prebuilt.chat_agent_executor import (
    create_function_calling_executor,
    create_tool_calling_executor,
)
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.pregel import Channel, GraphRecursionError, Pregel, StateSnapshot
from tests.any_str import AnyStr
from tests.memory_assert import MemorySaverAssertImmutable


async def test_node_cancellation_on_external_cancel() -> None:
    inner_task_cancelled = False

    async def awhile(input: Any) -> None:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise

    builder = Graph()
    builder.add_node("agent", awhile)
    builder.set_entry_point("agent")
    builder.set_finish_point("agent")

    graph = builder.compile()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(graph.ainvoke(1), 0.5)

    assert inner_task_cancelled


async def test_node_cancellation_on_other_node_exception() -> None:
    inner_task_cancelled = False

    async def awhile(input: Any) -> None:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise

    async def iambad(input: Any) -> None:
        raise ValueError("I am bad")

    builder = Graph()
    builder.add_node("agent", awhile)
    builder.add_node("bad", iambad)
    builder.set_conditional_entry_point(lambda _: ["agent", "bad"], then=END)

    graph = builder.compile()

    with pytest.raises(ValueError, match="I am bad"):
        await graph.ainvoke(1)

    assert inner_task_cancelled


async def test_invoke_single_process_in_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={
            "one": chain,
        },
        channels={
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )
    graph = Graph()
    graph.add_node("add_one", add_one)
    graph.set_entry_point("add_one")
    graph.set_finish_point("add_one")
    gapp = graph.compile()

    assert app.input_schema.schema() == {"title": "LangGraphInput", "type": "integer"}
    assert app.output_schema.schema() == {"title": "LangGraphOutput", "type": "integer"}
    assert await app.ainvoke(2) == 3
    assert await app.ainvoke(2, output_keys=["output"]) == {"output": 3}

    assert await gapp.ainvoke(2) == 3


@pytest.mark.parametrize(
    "falsy_value",
    [None, False, 0, "", [], {}, set(), frozenset(), 0.0, 0j],
)
async def test_invoke_single_process_in_out_falsy_values(falsy_value: Any) -> None:
    graph = Graph()
    graph.add_node("return_falsy_const", lambda *args, **kwargs: falsy_value)
    graph.set_entry_point("return_falsy_const")
    graph.set_finish_point("return_falsy_const")
    gapp = graph.compile()
    assert falsy_value == await gapp.ainvoke(1)


async def test_invoke_single_process_in_write_kwargs(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = (
        Channel.subscribe_to("input")
        | add_one
        | Channel.write_to("output", fixed=5, output_plus_one=lambda x: x + 1)
    )

    app = Pregel(
        nodes={"one": chain},
        channels={
            "input": LastValue(int),
            "output": LastValue(int),
            "fixed": LastValue(int),
            "output_plus_one": LastValue(int),
        },
        output_channels=["output", "fixed", "output_plus_one"],
        input_channels="input",
    )

    assert app.input_schema.schema() == {"title": "LangGraphInput", "type": "integer"}
    assert app.output_schema.schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {
            "output": {"title": "Output", "type": "integer"},
            "fixed": {"title": "Fixed", "type": "integer"},
            "output_plus_one": {"title": "Output Plus One", "type": "integer"},
        },
    }
    assert await app.ainvoke(2) == {"output": 3, "fixed": 5, "output_plus_one": 4}


async def test_invoke_single_process_in_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": chain},
        channels={"input": LastValue(int), "output": LastValue(int)},
        input_channels="input",
        output_channels=["output"],
    )

    assert app.input_schema.schema() == {"title": "LangGraphInput", "type": "integer"}
    assert app.output_schema.schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {"output": {"title": "Output", "type": "integer"}},
    }
    assert await app.ainvoke(2) == {"output": 3}


async def test_invoke_single_process_in_dict_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": chain},
        channels={"input": LastValue(int), "output": LastValue(int)},
        input_channels=["input"],
        output_channels=["output"],
    )

    assert app.input_schema.schema() == {
        "title": "LangGraphInput",
        "type": "object",
        "properties": {"input": {"title": "Input", "type": "integer"}},
    }
    assert app.output_schema.schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {"output": {"title": "Output", "type": "integer"}},
    }
    assert await app.ainvoke({"input": 2}) == {"output": 3}


async def test_invoke_two_processes_in_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = Channel.subscribe_to("inbox") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "inbox": LastValue(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
        stream_channels=["inbox", "output"],
    )

    assert await app.ainvoke(2) == 4

    assert await app.ainvoke(2, input_keys="inbox") == 3

    with pytest.raises(GraphRecursionError):
        await app.ainvoke(2, {"recursion_limit": 1})

    step = 0
    async for values in app.astream(2):
        step += 1
        if step == 1:
            assert values == {
                "inbox": 3,
            }
        elif step == 2:
            assert values == {
                "inbox": 3,
                "output": 4,
            }
    assert step == 2

    graph = Graph()
    graph.add_node("add_one", add_one)
    graph.add_node("add_one_more", add_one)
    graph.set_entry_point("add_one")
    graph.set_finish_point("add_one_more")
    graph.add_edge("add_one", "add_one_more")
    gapp = graph.compile()

    assert await gapp.ainvoke(2) == 4

    step = 0
    async for values in gapp.astream(2):
        step += 1
        if step == 1:
            assert values == {
                "add_one": 3,
            }
        elif step == 2:
            assert values == {
                "add_one_more": 4,
            }
    assert step == 2


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_invoke_two_processes_in_out_interrupt(
    mocker: MockerFixture, checkpoint_at: CheckpointAt
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = Channel.subscribe_to("inbox") | add_one | Channel.write_to("output")

    memory = MemorySaverAssertImmutable(at=checkpoint_at)
    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "inbox": LastValue(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
        checkpointer=memory,
        interrupt_after_nodes=["one"],
    )

    # start execution, stop at inbox
    assert await app.ainvoke(2, {"configurable": {"thread_id": 1}}) is None

    # inbox == 3
    checkpoint = await memory.aget({"configurable": {"thread_id": 1}})
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 3

    # resume execution, finish
    assert await app.ainvoke(None, {"configurable": {"thread_id": 1}}) == 4

    # start execution again, stop at inbox
    assert await app.ainvoke(20, {"configurable": {"thread_id": 1}}) is None

    # inbox == 21
    checkpoint = await memory.aget({"configurable": {"thread_id": 1}})
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 21

    # send a new value in, interrupting the previous execution
    assert await app.ainvoke(3, {"configurable": {"thread_id": 1}}) is None
    assert await app.ainvoke(None, {"configurable": {"thread_id": 1}}) == 5

    # start execution again, stopping at inbox
    assert await app.ainvoke(20, {"configurable": {"thread_id": 2}}) is None

    # inbox == 21
    snapshot = await app.aget_state({"configurable": {"thread_id": 2}})
    assert snapshot.values["inbox"] == 21
    assert snapshot.next == ("two",)

    # update the state, resume
    await app.aupdate_state({"configurable": {"thread_id": 2}}, 25, as_node="one")
    assert await app.ainvoke(None, {"configurable": {"thread_id": 2}}) == 26

    # no pending tasks
    snapshot = await app.aget_state({"configurable": {"thread_id": 2}})
    assert snapshot.next == ()


async def test_invoke_two_processes_in_dict_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = (
        Channel.subscribe_to("inbox")
        | RunnableLambda(add_one).abatch
        | Channel.write_to("output").abatch
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "inbox": Topic(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels=["input", "inbox"],
        stream_channels=["output", "inbox"],
        output_channels=["output"],
    )

    # [12 + 1, 2 + 1 + 1]
    assert [
        c
        async for c in app.astream(
            {"input": 2, "inbox": 12}, output_keys="output", stream_mode="updates"
        )
    ] == [
        {"two": 13},
        {"two": 4},
    ]
    assert [
        c async for c in app.astream({"input": 2, "inbox": 12}, output_keys="output")
    ] == [13, 4]

    assert [
        c async for c in app.astream({"input": 2, "inbox": 12}, stream_mode="updates")
    ] == [
        {"one": {"inbox": 3}, "two": {"output": 13}},
        {"two": {"output": 4}},
    ]
    assert [c async for c in app.astream({"input": 2, "inbox": 12})] == [
        {"inbox": [3], "output": 13},
        {"inbox": [], "output": 4},
    ]
    assert [
        c async for c in app.astream({"input": 2, "inbox": 12}, stream_mode="debug")
    ] == [
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": "7a3cc398-2e02-5023-ad7b-e4848d3b67fa",
                "name": "one",
                "input": 2,
                "triggers": ["input"],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": "34e90af0-f97e-54e0-a159-691da37f175f",
                "name": "two",
                "input": [12],
                "triggers": ["inbox"],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": "7a3cc398-2e02-5023-ad7b-e4848d3b67fa",
                "name": "one",
                "result": [("inbox", 3)],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": "34e90af0-f97e-54e0-a159-691da37f175f",
                "name": "two",
                "result": [("output", 13)],
            },
        },
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {"config": None, "values": {"output": 13, "inbox": [3]}},
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": "cf7cf374-2a2a-556f-8561-91737af89d2f",
                "name": "two",
                "input": [3],
                "triggers": ["inbox"],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": "cf7cf374-2a2a-556f-8561-91737af89d2f",
                "name": "two",
                "result": [("output", 4)],
            },
        },
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {"config": None, "values": {"output": 4, "inbox": []}},
        },
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {"config": None, "values": {"output": 4, "inbox": []}},
        },
    ]


async def test_batch_two_processes_in_out() -> None:
    async def add_one_with_delay(inp: int) -> int:
        await asyncio.sleep(inp / 10)
        return inp + 1

    one = Channel.subscribe_to("input") | add_one_with_delay | Channel.write_to("one")
    two = Channel.subscribe_to("one") | add_one_with_delay | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "one": LastValue(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    assert await app.abatch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]
    assert await app.abatch([3, 2, 1, 3, 5], output_keys=["output"]) == [
        {"output": 5},
        {"output": 4},
        {"output": 3},
        {"output": 5},
        {"output": 7},
    ]

    graph = Graph()
    graph.add_node("add_one", add_one_with_delay)
    graph.add_node("add_one_more", add_one_with_delay)
    graph.set_entry_point("add_one")
    graph.set_finish_point("add_one_more")
    graph.add_edge("add_one", "add_one_more")
    gapp = graph.compile()

    assert await gapp.abatch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]


async def test_invoke_many_processes_in_out(mocker: MockerFixture) -> None:
    test_size = 100
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    nodes = {"-1": Channel.subscribe_to("input") | add_one | Channel.write_to("-1")}
    for i in range(test_size - 2):
        nodes[str(i)] = (
            Channel.subscribe_to(str(i - 1)) | add_one | Channel.write_to(str(i))
        )
    nodes["last"] = Channel.subscribe_to(str(i)) | add_one | Channel.write_to("output")

    app = Pregel(
        nodes=nodes,
        channels={str(i): LastValue(int) for i in range(-1, test_size - 2)}
        | {"input": LastValue(int), "output": LastValue(int)},
        input_channels="input",
        output_channels="output",
    )

    # No state is left over from previous invocations
    for _ in range(10):
        assert await app.ainvoke(2, {"recursion_limit": test_size}) == 2 + test_size

    # Concurrent invocations do not interfere with each other
    assert await asyncio.gather(
        *(app.ainvoke(2, {"recursion_limit": test_size}) for _ in range(10))
    ) == [2 + test_size for _ in range(10)]


async def test_batch_many_processes_in_out(mocker: MockerFixture) -> None:
    test_size = 100
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    nodes = {"-1": Channel.subscribe_to("input") | add_one | Channel.write_to("-1")}
    for i in range(test_size - 2):
        nodes[str(i)] = (
            Channel.subscribe_to(str(i - 1)) | add_one | Channel.write_to(str(i))
        )
    nodes["last"] = Channel.subscribe_to(str(i)) | add_one | Channel.write_to("output")

    app = Pregel(
        nodes=nodes,
        channels={str(i): LastValue(int) for i in range(-1, test_size - 2)}
        | {"input": LastValue(int), "output": LastValue(int)},
        input_channels="input",
        output_channels="output",
    )

    # No state is left over from previous invocations
    for _ in range(3):
        # Then invoke pubsub
        assert await app.abatch([2, 1, 3, 4, 5], {"recursion_limit": test_size}) == [
            2 + test_size,
            1 + test_size,
            3 + test_size,
            4 + test_size,
            5 + test_size,
        ]

    # Concurrent invocations do not interfere with each other
    assert await asyncio.gather(
        *(app.abatch([2, 1, 3, 4, 5], {"recursion_limit": test_size}) for _ in range(3))
    ) == [
        [2 + test_size, 1 + test_size, 3 + test_size, 4 + test_size, 5 + test_size]
        for _ in range(3)
    ]


async def test_invoke_two_processes_two_in_two_out_invalid(
    mocker: MockerFixture,
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = Channel.subscribe_to("input") | add_one | Channel.write_to("output")
    two = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={"output": LastValue(int), "input": LastValue(int)},
        input_channels="input",
        output_channels="output",
    )

    with pytest.raises(InvalidUpdateError):
        # LastValue channels can only be updated once per iteration
        await app.ainvoke(2)


async def test_invoke_two_processes_two_in_two_out_valid(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = Channel.subscribe_to("input") | add_one | Channel.write_to("output")
    two = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "input": LastValue(int),
            "output": Topic(int),
        },
        input_channels="input",
        output_channels="output",
    )

    # An Topic channel accumulates updates into a sequence
    assert await app.ainvoke(2) == [3, 3]


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_invoke_checkpoint(
    mocker: MockerFixture, checkpoint_at: CheckpointAt
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x["total"] + x["input"])

    def raise_if_above_10(input: int) -> int:
        if input > 10:
            raise ValueError("Input is too large")
        return input

    one = (
        Channel.subscribe_to(["input"]).join(["total"])
        | add_one
        | Channel.write_to("output", "total")
        | raise_if_above_10
    )

    memory = MemorySaverAssertImmutable(at=checkpoint_at)

    app = Pregel(
        nodes={"one": one},
        channels={
            "total": BinaryOperatorAggregate(int, operator.add),
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
        checkpointer=memory,
    )

    # total starts out as 0, so output is 0+2=2
    assert await app.ainvoke(2, {"configurable": {"thread_id": "1"}}) == 2
    checkpoint = await memory.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 2
    # total is now 2, so output is 2+3=5
    assert await app.ainvoke(3, {"configurable": {"thread_id": "1"}}) == 5
    checkpoint = await memory.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        await app.ainvoke(4, {"configurable": {"thread_id": "1"}})
    # checkpoint is not updated
    checkpoint = await memory.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert await app.ainvoke(5, {"configurable": {"thread_id": "2"}}) == 5
    checkpoint = await memory.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    checkpoint = await memory.aget({"configurable": {"thread_id": "2"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 5


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_invoke_checkpoint_aiosqlite(
    mocker: MockerFixture, checkpoint_at: CheckpointAt
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x["total"] + x["input"])

    def raise_if_above_10(input: int) -> int:
        if input > 10:
            raise ValueError("Input is too large")
        return input

    one = (
        Channel.subscribe_to(["input"]).join(["total"])
        | add_one
        | Channel.write_to("output", "total")
        | raise_if_above_10
    )

    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        memory.at = checkpoint_at
        app = Pregel(
            nodes={"one": one},
            channels={
                "total": BinaryOperatorAggregate(int, operator.add),
                "input": LastValue(int),
                "output": LastValue(int),
            },
            input_channels="input",
            output_channels="output",
            checkpointer=memory,
            debug=True,
        )

        thread_1 = {"configurable": {"thread_id": "1"}}
        # total starts out as 0, so output is 0+2=2
        assert await app.ainvoke(2, thread_1) == 2
        state = await app.aget_state(thread_1)
        assert state is not None
        assert state.values.get("total") == 2
        assert (
            state.config["configurable"]["thread_ts"]
            == (await memory.aget(thread_1))["ts"]
        )
        # total is now 2, so output is 2+3=5
        assert await app.ainvoke(3, thread_1) == 5
        state = await app.aget_state(thread_1)
        assert state is not None
        assert state.values.get("total") == 7
        assert (
            state.config["configurable"]["thread_ts"]
            == (await memory.aget(thread_1))["ts"]
        )
        # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
        with pytest.raises(ValueError):
            await app.ainvoke(4, thread_1)
        # checkpoint is not updated
        state = await app.aget_state(thread_1)
        assert state is not None
        assert state.values.get("total") == 7

        thread_2 = {"configurable": {"thread_id": "2"}}
        # on a new thread, total starts out as 0, so output is 0+5=5
        assert await app.ainvoke(5, thread_2) == 5
        state = await app.aget_state({"configurable": {"thread_id": "1"}})
        assert state is not None
        assert state.values.get("total") == 7
        assert state.next == ()
        state = await app.aget_state(thread_2)
        assert state is not None
        assert state.values.get("total") == 5
        assert state.next == ()

        assert len([c async for c in app.aget_state_history(thread_1, limit=1)]) == 1
        # list all checkpoints for thread 1
        thread_1_history = [c async for c in app.aget_state_history(thread_1)]
        # there are 2: one for each successful ainvoke()
        assert len(thread_1_history) == 2
        # sorted descending
        assert (
            thread_1_history[0].config["configurable"]["thread_ts"]
            > thread_1_history[1].config["configurable"]["thread_ts"]
        )
        # cursor pagination
        cursored = [
            c
            async for c in app.aget_state_history(
                thread_1, limit=1, before=thread_1_history[0].config
            )
        ]
        assert len(cursored) == 1
        assert cursored[0].config == thread_1_history[1].config
        # the second checkpoint
        assert thread_1_history[0].values["total"] == 7
        # the first checkpoint
        assert thread_1_history[1].values["total"] == 2
        # can get each checkpoint using aget with config
        assert (await memory.aget(thread_1_history[0].config))[
            "ts"
        ] == thread_1_history[0].config["configurable"]["thread_ts"]
        assert (await memory.aget(thread_1_history[1].config))[
            "ts"
        ] == thread_1_history[1].config["configurable"]["thread_ts"]

        thread_1_next_config = await app.aupdate_state(thread_1_history[1].config, 10)
        # update creates a new checkpoint
        assert (
            thread_1_next_config["configurable"]["thread_ts"]
            > thread_1_history[0].config["configurable"]["thread_ts"]
        )
        # 1 more checkpoint in history
        assert len([h async for h in app.aget_state_history(thread_1)]) == 3
        # the latest checkpoint is the updated one
        assert await app.aget_state(thread_1) == await app.aget_state(
            thread_1_next_config
        )


async def test_invoke_two_processes_two_in_join_two_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: sorted(y + 10 for y in x))

    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    chain_three = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    chain_four = (
        Channel.subscribe_to("inbox") | add_10_each | Channel.write_to("output")
    )

    app = Pregel(
        nodes={
            "one": one,
            "chain_three": chain_three,
            "chain_four": chain_four,
        },
        channels={
            "inbox": Topic(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    # Then invoke app
    # We get a single array result as chain_four waits for all publishers to finish
    # before operating on all elements published to topic_two as an array
    for _ in range(100):
        assert await app.ainvoke(2) == [13, 13]

    assert await asyncio.gather(*(app.ainvoke(2) for _ in range(100))) == [
        [13, 13] for _ in range(100)
    ]


async def test_invoke_join_then_call_other_pregel(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    inner_app = Pregel(
        nodes={
            "one": Channel.subscribe_to("input") | add_one | Channel.write_to("output")
        },
        channels={
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    one = (
        Channel.subscribe_to("input")
        | add_10_each
        | Channel.write_to("inbox_one").map()
    )
    two = (
        Channel.subscribe_to("inbox_one")
        | inner_app.map()
        | sorted
        | Channel.write_to("outbox_one")
    )
    chain_three = Channel.subscribe_to("outbox_one") | sum | Channel.write_to("output")

    app = Pregel(
        nodes={
            "one": one,
            "two": two,
            "chain_three": chain_three,
        },
        channels={
            "inbox_one": Topic(int),
            "outbox_one": LastValue(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    # Then invoke pubsub
    for _ in range(10):
        assert await app.ainvoke([2, 3]) == 27

    assert await asyncio.gather(*(app.ainvoke([2, 3]) for _ in range(10))) == [
        27 for _ in range(10)
    ]


async def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = (
        Channel.subscribe_to("input")
        | add_one
        | Channel.write_to(output=RunnablePassthrough(), between=RunnablePassthrough())
    )
    two = Channel.subscribe_to("between") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "input": LastValue(int),
            "between": LastValue(int),
            "output": LastValue(int),
        },
        stream_channels=["output", "between"],
        input_channels="input",
        output_channels="output",
    )

    # Then invoke pubsub
    assert [c async for c in app.astream(2)] == [
        {"between": 3, "output": 3},
        {"between": 3, "output": 4},
    ]


async def test_invoke_two_processes_no_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("between")
    two = Channel.subscribe_to("between") | add_one

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "input": LastValue(int),
            "between": LastValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    # It finishes executing (once no more messages being published)
    # but returns nothing, as nothing was published to "output" topic
    assert await app.ainvoke(2) is None


async def test_channel_enter_exit_timing(mocker: MockerFixture) -> None:
    setup_sync = mocker.Mock()
    cleanup_sync = mocker.Mock()
    setup_async = mocker.Mock()
    cleanup_async = mocker.Mock()

    @contextmanager
    def an_int() -> Generator[int, None, None]:
        setup_sync()
        try:
            yield 5
        finally:
            cleanup_sync()

    @asynccontextmanager
    async def an_int_async() -> AsyncGenerator[int, None]:
        setup_async()
        try:
            yield 5
        finally:
            cleanup_async()

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = (
        Channel.subscribe_to("inbox")
        | RunnableLambda(add_one).abatch
        | Channel.write_to("output").abatch
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "input": LastValue(int),
            "output": LastValue(int),
            "inbox": Topic(int),
            "ctx": Context(an_int, an_int_async, typ=int),
        },
        input_channels="input",
        output_channels=["inbox", "output"],
        stream_channels=["inbox", "output"],
    )

    async def aenumerate(aiter: AsyncIterator[Any]) -> AsyncIterator[tuple[int, Any]]:
        i = 0
        async for chunk in aiter:
            yield i, chunk
            i += 1

    assert setup_sync.call_count == 0
    assert cleanup_sync.call_count == 0
    assert setup_async.call_count == 0
    assert cleanup_async.call_count == 0
    async for i, chunk in aenumerate(app.astream(2)):
        assert setup_sync.call_count == 0, "Sync context manager should not be used"
        assert cleanup_sync.call_count == 0, "Sync context manager should not be used"
        assert setup_async.call_count == 1, "Expected setup to be called once"
        assert cleanup_async.call_count == 0, "Expected cleanup to not be called yet"
        if i == 0:
            assert chunk == {"inbox": [3]}
        elif i == 1:
            assert chunk == {"inbox": [], "output": 4}
        else:
            assert False, "Expected only two chunks"
    assert setup_sync.call_count == 0
    assert cleanup_sync.call_count == 0
    assert setup_async.call_count == 1, "Expected setup to be called once"
    assert cleanup_async.call_count == 1, "Expected cleanup to be called once"


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_conditional_graph(checkpoint_at: CheckpointAt) -> None:
    from copy import deepcopy

    from langchain.llms.fake import FakeStreamingListLLM
    from langchain_community.tools import tool
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    # Assemble the tools
    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    # Construct the agent
    prompt = PromptTemplate.from_template("Hello!")

    llm = FakeStreamingListLLM(
        responses=[
            "tool:search_api:query",
            "tool:search_api:another",
            "finish:answer",
        ]
    )

    async def agent_parser(input: str) -> Union[AgentAction, AgentFinish]:
        if input.startswith("finish"):
            _, answer = input.split(":")
            return AgentFinish(return_values={"answer": answer}, log=input)
        else:
            _, tool_name, tool_input = input.split(":")
            return AgentAction(tool=tool_name, tool_input=tool_input, log=input)

    agent = RunnablePassthrough.assign(agent_outcome=prompt | llm | agent_parser)

    # Define tool execution logic
    async def execute_tools(data: dict) -> dict:
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = await {t.name: t for t in tools}[agent_action.tool].ainvoke(
            agent_action.tool_input
        )
        if data.get("intermediate_steps") is None:
            data["intermediate_steps"] = []
        data["intermediate_steps"].append((agent_action, observation))
        return data

    # Define decision-making logic
    async def should_continue(data: dict, config: RunnableConfig) -> str:
        # Logic to decide whether to continue in the loop or exit
        if isinstance(data["agent_outcome"], AgentFinish):
            return "exit"
        else:
            return "continue"

    # Define a new graph
    workflow = Graph()

    workflow.add_node("agent", agent)
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    assert await app.ainvoke({"input": "what is weather in sf"}) == {
        "input": "what is weather in sf",
        "intermediate_steps": [
            (
                AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:query",
                ),
                "result for query",
            ),
            (
                AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
                "result for another",
            ),
        ],
        "agent_outcome": AgentFinish(
            return_values={"answer": "answer"}, log="finish:answer"
        ),
    }

    # deepcopy because the nodes mutate the data
    assert [
        deepcopy(c) async for c in app.astream({"input": "what is weather in sf"})
    ] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ),
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ),
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ),
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ),
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "answer"}, log="finish:answer"
                ),
            }
        },
    ]

    patches = [c async for c in app.astream_log({"input": "what is weather in sf"})]
    patch_paths = {op["path"] for log in patches for op in log.ops}

    # Check that agent (one of the nodes) has its output streamed to the logs
    assert "/logs/agent/streamed_output/-" in patch_paths
    assert "/logs/agent:2/streamed_output/-" in patch_paths
    assert "/logs/agent:3/streamed_output/-" in patch_paths
    # Check that agent (one of the nodes) has its final output set in the logs
    assert "/logs/agent/final_output" in patch_paths
    assert "/logs/agent:2/final_output" in patch_paths
    assert "/logs/agent:3/final_output" in patch_paths
    assert [
        p["value"]
        for log in patches
        for p in log.ops
        if p["path"] == "/logs/agent/final_output"
        or p["path"] == "/logs/agent:2/final_output"
        or p["path"] == "/logs/agent:3/final_output"
    ] == [
        {
            "input": "what is weather in sf",
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
        },
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                )
            ],
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="another",
                log="tool:search_api:another",
            ),
        },
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                ),
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="another",
                        log="tool:search_api:another",
                    ),
                    "result for another",
                ),
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "answer"}, log="finish:answer"
            ),
        },
    ]

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(at=checkpoint_at),
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config
        )
    ] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    await app_w_interrupt.aupdate_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "input": "what is weather in sf",
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
        },
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    await app_w_interrupt.aupdate_state(
        config,
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                )
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                ),
            },
        },
        next=(),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(at=checkpoint_at),
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config
        )
    ] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    await app_w_interrupt.aupdate_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "input": "what is weather in sf",
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
        },
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    await app_w_interrupt.aupdate_state(
        config,
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                )
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                ),
            },
        },
        next=(),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    # test re-invoke to continue with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(at=checkpoint_at),
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config
        )
    ] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ),
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ),
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ),
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ),
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "answer"}, log="finish:answer"
                ),
            }
        },
    ]


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_conditional_graph_state(checkpoint_at: CheckpointAt) -> None:
    from langchain.llms.fake import FakeStreamingListLLM
    from langchain_community.tools import tool
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.prompts import PromptTemplate

    class AgentState(TypedDict):
        input: str
        agent_outcome: Optional[Union[AgentAction, AgentFinish]]
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    # Assemble the tools
    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    # Construct the agent
    prompt = PromptTemplate.from_template("Hello!")

    llm = FakeStreamingListLLM(
        responses=[
            "tool:search_api:query",
            "tool:search_api:another",
            "finish:answer",
        ]
    )

    def agent_parser(input: str) -> dict[str, Union[AgentAction, AgentFinish]]:
        if input.startswith("finish"):
            _, answer = input.split(":")
            return {
                "agent_outcome": AgentFinish(
                    return_values={"answer": answer}, log=input
                )
            }
        else:
            _, tool_name, tool_input = input.split(":")
            return {
                "agent_outcome": AgentAction(
                    tool=tool_name, tool_input=tool_input, log=input
                )
            }

    agent = prompt | llm | agent_parser

    # Define tool execution logic
    def execute_tools(data: AgentState) -> dict:
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        return {"intermediate_steps": [(agent_action, observation)]}

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        # Logic to decide whether to continue in the loop or exit
        if isinstance(data["agent_outcome"], AgentFinish):
            return "exit"
        else:
            return "continue"

    # Define a new graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    assert await app.ainvoke({"input": "what is weather in sf"}) == {
        "input": "what is weather in sf",
        "intermediate_steps": [
            (
                AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:query",
                ),
                "result for query",
            ),
            (
                AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
                "result for another",
            ),
        ],
        "agent_outcome": AgentFinish(
            return_values={"answer": "answer"}, log="finish:answer"
        ),
    }

    assert [c async for c in app.astream({"input": "what is weather in sf"})] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
        {
            "tools": {
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
        {
            "tools": {
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ),
                ],
            }
        },
        {
            "agent": {
                "agent_outcome": AgentFinish(
                    return_values={"answer": "answer"}, log="finish:answer"
                ),
            }
        },
    ]

    patches = [c async for c in app.astream_log({"input": "what is weather in sf"})]
    patch_paths = {op["path"] for log in patches for op in log.ops}

    # Check that agent (one of the nodes) has its output streamed to the logs
    assert "/logs/agent/streamed_output/-" in patch_paths
    # Check that agent (one of the ndoes) has its final output set in the logs
    assert "/logs/agent/final_output" in patch_paths
    assert [
        p["value"]
        for log in patches
        for p in log.ops
        if p["path"] == "/logs/agent/final_output"
        or p["path"] == "/logs/agent:2/final_output"
        or p["path"] == "/logs/agent:3/final_output"
    ] == [
        {
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            )
        },
        {
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="another", log="tool:search_api:another"
            )
        },
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "answer"}, log="finish:answer"
            ),
        },
    ]

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(at=checkpoint_at),
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config
        )
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "input": "what is weather in sf",
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:query",
            ),
            "intermediate_steps": [],
        },
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    await app_w_interrupt.aupdate_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            )
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "input": "what is weather in sf",
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    await app_w_interrupt.aupdate_state(
        config,
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            )
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "input": "what is weather in sf",
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                )
            ],
        },
        next=(),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(at=checkpoint_at),
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config
        )
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "input": "what is weather in sf",
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    await app_w_interrupt.aupdate_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            )
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "input": "what is weather in sf",
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    await app_w_interrupt.aupdate_state(
        config,
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            )
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "input": "what is weather in sf",
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                )
            ],
        },
        next=(),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )


async def test_conditional_entrypoint_graph() -> None:
    async def left(data: str) -> str:
        return data + "->left"

    async def right(data: str) -> str:
        return data + "->right"

    def should_start(data: str) -> str:
        # Logic to decide where to start
        if len(data) > 10:
            return "go-right"
        else:
            return "go-left"

    # Define a new graph
    workflow = Graph()

    workflow.add_node("left", left)
    workflow.add_node("right", right)

    workflow.set_conditional_entry_point(
        should_start, {"go-left": "left", "go-right": "right"}
    )

    workflow.add_conditional_edges("left", lambda data: END)
    workflow.add_edge("right", END)

    app = workflow.compile()

    assert await app.ainvoke("what is weather in sf") == "what is weather in sf->right"

    assert [c async for c in app.astream("what is weather in sf")] == [
        {"right": "what is weather in sf->right"},
    ]


async def test_conditional_entrypoint_graph_state() -> None:
    class AgentState(TypedDict, total=False):
        input: str
        output: str
        steps: Annotated[list[str], operator.add]

    async def left(data: AgentState) -> AgentState:
        return {"output": data["input"] + "->left"}

    async def right(data: AgentState) -> AgentState:
        return {"output": data["input"] + "->right"}

    def should_start(data: AgentState) -> str:
        assert data["steps"] == [], "Expected input to be read from the state"
        # Logic to decide where to start
        if len(data["input"]) > 10:
            return "go-right"
        else:
            return "go-left"

    # Define a new graph
    workflow = StateGraph(AgentState)

    workflow.add_node("left", left)
    workflow.add_node("right", right)

    workflow.set_conditional_entry_point(
        should_start, {"go-left": "left", "go-right": "right"}
    )

    workflow.add_conditional_edges("left", lambda data: END)
    workflow.add_edge("right", END)

    app = workflow.compile()

    assert await app.ainvoke({"input": "what is weather in sf"}) == {
        "input": "what is weather in sf",
        "output": "what is weather in sf->right",
        "steps": [],
    }

    assert [c async for c in app.astream({"input": "what is weather in sf"})] == [
        {"right": {"output": "what is weather in sf->right"}},
    ]


async def test_prebuilt_tool_chat() -> None:
    from langchain.chat_models.fake import FakeMessagesListChatModel
    from langchain_community.tools import tool
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_tools(self, functions: list):
            return self

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    app = create_tool_calling_executor(
        FakeFuntionChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another"},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one"},
                        },
                    ],
                ),
                AIMessage(content="answer"),
            ]
        ),
        tools,
    )

    assert await app.ainvoke(
        {"messages": [HumanMessage(content="what is weather in sf")]}
    ) == {
        "messages": [
            HumanMessage(content="what is weather in sf", id=AnyStr()),
            AIMessage(
                id=AnyStr(),
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            ToolMessage(
                content="result for query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(
                id=AnyStr(),
                content="",
                tool_calls=[
                    {
                        "id": "tool_call234",
                        "name": "search_api",
                        "args": {"query": "another"},
                    },
                    {
                        "id": "tool_call567",
                        "name": "search_api",
                        "args": {"query": "a third one"},
                    },
                ],
            ),
            ToolMessage(
                content="result for another",
                name="search_api",
                tool_call_id="tool_call234",
                id=AnyStr(),
            ),
            ToolMessage(
                content="result for a third one",
                name="search_api",
                tool_call_id="tool_call567",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id=AnyStr()),
        ]
    }

    assert [
        c
        async for c in app.astream(
            {"messages": [HumanMessage(content="what is weather in sf")]}
        )
    ] == [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        id=AnyStr(),
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call123",
                                "name": "search_api",
                                "args": {"query": "query"},
                            },
                        ],
                    )
                ]
            }
        },
        {
            "action": {
                "messages": [
                    ToolMessage(
                        content="result for query",
                        name="search_api",
                        tool_call_id="tool_call123",
                        id=AnyStr(),
                    )
                ]
            }
        },
        {
            "agent": {
                "messages": [
                    AIMessage(
                        id=AnyStr(),
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call234",
                                "name": "search_api",
                                "args": {"query": "another"},
                            },
                            {
                                "id": "tool_call567",
                                "name": "search_api",
                                "args": {"query": "a third one"},
                            },
                        ],
                    )
                ]
            }
        },
        {
            "action": {
                "messages": [
                    ToolMessage(
                        content="result for another",
                        tool_call_id="tool_call234",
                        name="search_api",
                        id=AnyStr(),
                    ),
                    ToolMessage(
                        content="result for a third one",
                        tool_call_id="tool_call567",
                        name="search_api",
                        id=AnyStr(),
                    ),
                ]
            }
        },
        {"agent": {"messages": [AIMessage(content="answer", id=AnyStr())]}},
    ]


async def test_prebuilt_chat() -> None:
    from langchain.chat_models.fake import FakeMessagesListChatModel
    from langchain_community.tools import tool
    from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
            return self

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    app = create_function_calling_executor(
        FakeFuntionChatModel(
            responses=[
                AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "search_api",
                            "arguments": json.dumps("query"),
                        }
                    },
                ),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "search_api",
                            "arguments": json.dumps("another"),
                        }
                    },
                ),
                AIMessage(content="answer"),
            ]
        ),
        tools,
    )

    assert await app.ainvoke(
        {"messages": [HumanMessage(content="what is weather in sf")]}
    ) == {
        "messages": [
            HumanMessage(content="what is weather in sf", id=AnyStr()),
            AIMessage(
                id=AnyStr(),
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"query"'}
                },
            ),
            FunctionMessage(content="result for query", name="search_api", id=AnyStr()),
            AIMessage(
                id=AnyStr(),
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"another"'}
                },
            ),
            FunctionMessage(
                content="result for another", name="search_api", id=AnyStr()
            ),
            AIMessage(content="answer", id=AnyStr()),
        ]
    }

    assert [
        c
        async for c in app.astream(
            {"messages": [HumanMessage(content="what is weather in sf")]}
        )
    ] == [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        id=AnyStr(),
                        content="",
                        additional_kwargs={
                            "function_call": {
                                "name": "search_api",
                                "arguments": '"query"',
                            }
                        },
                    )
                ]
            }
        },
        {
            "action": {
                "messages": [
                    FunctionMessage(
                        content="result for query", name="search_api", id=AnyStr()
                    )
                ]
            }
        },
        {
            "agent": {
                "messages": [
                    AIMessage(
                        id=AnyStr(),
                        content="",
                        additional_kwargs={
                            "function_call": {
                                "name": "search_api",
                                "arguments": '"another"',
                            }
                        },
                    )
                ]
            }
        },
        {
            "action": {
                "messages": [
                    FunctionMessage(
                        content="result for another", name="search_api", id=AnyStr()
                    )
                ]
            }
        },
        {"agent": {"messages": [AIMessage(content="answer", id=AnyStr())]}},
    ]


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_message_graph(checkpoint_at: CheckpointAt) -> None:
    from langchain.chat_models.fake import FakeMessagesListChatModel
    from langchain_community.tools import tool
    from langchain_core.agents import AgentAction
    from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
            return self

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    model = FakeFuntionChatModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "search_api",
                        "arguments": json.dumps("query"),
                    }
                },
                id="ai1",
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "search_api",
                        "arguments": json.dumps("another"),
                    }
                },
                id="ai2",
            ),
            AIMessage(content="answer", id="ai3"),
        ]
    )

    tool_executor = ToolExecutor(tools)

    # Define the function that determines whether to continue or not
    def should_continue(messages):
        last_message = messages[-1]
        # If there is no function call, then we finish
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    async def call_tool(messages):
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an AgentAction from the function_call
        action = AgentAction(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(
                last_message.additional_kwargs["function_call"]["arguments"]
            ),
            log="",
        )
        # We call the tool_executor and get back a response
        response = await tool_executor.ainvoke(action)
        # We use the response to create a FunctionMessage
        return FunctionMessage(content=str(response), name=action.tool)

    # Define a new graph
    workflow = MessageGraph()

    # Define the two nodes we will cycle between
    workflow.add_node("agent", model)
    workflow.add_node("action", call_tool)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("action", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    assert await app.ainvoke(HumanMessage(content="what is weather in sf")) == [
        HumanMessage(content="what is weather in sf", id=AnyStr()),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {"name": "search_api", "arguments": '"query"'}
            },
            id="ai1",
        ),
        FunctionMessage(content="result for query", name="search_api", id=AnyStr()),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {"name": "search_api", "arguments": '"another"'}
            },
            id="ai2",
        ),
        FunctionMessage(content="result for another", name="search_api", id=AnyStr()),
        AIMessage(content="answer", id="ai3"),
    ]

    assert [
        c async for c in app.astream([HumanMessage(content="what is weather in sf")])
    ] == [
        {
            "agent": AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"query"'}
                },
                id="ai1",
            )
        },
        {
            "action": FunctionMessage(
                content="result for query", name="search_api", id=AnyStr()
            )
        },
        {
            "agent": AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"another"'}
                },
                id="ai2",
            )
        },
        {
            "action": FunctionMessage(
                content="result for another", name="search_api", id=AnyStr()
            )
        },
        {"agent": AIMessage(content="answer", id="ai3")},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(at=checkpoint_at),
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            HumanMessage(content="what is weather in sf"), config
        )
    ] == [
        {
            "agent": AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"query"'}
                },
                id="ai1",
            )
        },
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            HumanMessage(
                content="what is weather in sf",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"query"'}
                },
                id="ai1",
            ),
        ],
        next=("action",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        metadata={},
    )

    # modify ai message
    last_message = (await app_w_interrupt.aget_state(config)).values[-1]
    last_message.additional_kwargs["function_call"]["arguments"] = '"a different query"'
    await app_w_interrupt.aupdate_state(config, last_message)

    # message was replaced instead of appended
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            HumanMessage(
                content="what is weather in sf",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "search_api",
                        "arguments": '"a different query"',
                    }
                },
                id="ai1",
            ),
        ],
        next=("action",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        metadata={},
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "action": FunctionMessage(
                content="result for a different query",
                name="search_api",
                id=AnyStr(),
            )
        },
        {
            "agent": AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"another"'}
                },
                id="ai2",
            )
        },
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            HumanMessage(
                content="what is weather in sf",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "search_api",
                        "arguments": '"a different query"',
                    }
                },
                id="ai1",
            ),
            FunctionMessage(
                content="result for a different query",
                name="search_api",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"another"'}
                },
                id="ai2",
            ),
        ],
        next=("action",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        metadata={},
    )

    await app_w_interrupt.aupdate_state(
        config,
        AIMessage(content="answer", id="ai2"),
    )

    # replaces message even if object identity is different, as long as id is the same
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            HumanMessage(
                content="what is weather in sf",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "search_api",
                        "arguments": '"a different query"',
                    }
                },
                id="ai1",
            ),
            FunctionMessage(
                content="result for a different query",
                name="search_api",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        metadata={},
    )


async def test_in_one_fan_out_out_one_graph_state() -> None:
    def sorted_add(x: list[str], y: list[str]) -> list[str]:
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    workflow.add_edge("retriever_one", "qa")
    workflow.add_edge("retriever_two", "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "retriever_two": {"docs": ["doc3", "doc4"]},
            "retriever_one": {"docs": ["doc1", "doc2"]},
        },
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    assert [
        c
        async for c in app.astream(
            {"query": "what is weather in sf"}, stream_mode="values"
        )
    ] == [
        {"query": "what is weather in sf", "docs": []},
        {"query": "query: what is weather in sf", "docs": []},
        {
            "query": "query: what is weather in sf",
            "docs": ["doc1", "doc2", "doc3", "doc4"],
        },
        {
            "query": "query: what is weather in sf",
            "docs": ["doc1", "doc2", "doc3", "doc4"],
            "answer": "doc1,doc2,doc3,doc4",
        },
    ]


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_start_branch_then(
    snapshot: SnapshotAssertion, checkpoint_at: CheckpointAt
) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two_slow", lambda s, config: {"my_key": " slow"})
    tool_two_graph.add_node("tool_two_fast", lambda s: {"my_key": " fast"})
    tool_two_graph.set_conditional_entry_point(
        lambda s: "tool_two_slow" if s["market"] == "DE" else "tool_two_fast", then=END
    )
    tool_two = tool_two_graph.compile()

    assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}) == {
        "my_key": "value slow",
        "market": "DE",
    }
    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value fast",
        "market": "US",
    }

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        saver.at = checkpoint_at
        tool_two = tool_two_graph.compile(
            checkpointer=saver, interrupt_before=["tool_two_fast", "tool_two_slow"]
        )

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            await tool_two.ainvoke({"my_key": "value", "market": "DE"})

        thread1 = {"configurable": {"thread_id": "1"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}, thread1) == {
            "my_key": "value",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value", "market": "DE"},
            next=("tool_two_slow",),
            config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
            metadata={},
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread1, debug=1) == {
            "my_key": "value slow",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value slow", "market": "DE"},
            next=(),
            config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
            metadata={},
        )

        thread2 = {"configurable": {"thread_id": "2"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread2) == {
            "my_key": "value",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value", "market": "US"},
            next=("tool_two_fast",),
            config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
            metadata={},
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread2, debug=1) == {
            "my_key": "value fast",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value fast", "market": "US"},
            next=(),
            config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
            metadata={},
        )


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_branch_then(
    snapshot: SnapshotAssertion, checkpoint_at: CheckpointAt
) -> None:
    pass

    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_graph = StateGraph(State)
    tool_two_graph.set_entry_point("prepare")
    tool_two_graph.set_finish_point("finish")
    tool_two_graph.add_conditional_edges(
        source="prepare",
        path=lambda s: "tool_two_slow" if s["market"] == "DE" else "tool_two_fast",
        then="finish",
    )
    tool_two_graph.add_node("prepare", lambda s: {"my_key": " prepared"})
    tool_two_graph.add_node("tool_two_slow", lambda s: {"my_key": " slow"})
    tool_two_graph.add_node("tool_two_fast", lambda s: {"my_key": " fast"})
    tool_two_graph.add_node("finish", lambda s: {"my_key": " finished"})
    tool_two = tool_two_graph.compile()

    assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}, debug=1) == {
        "my_key": "value prepared slow finished",
        "market": "DE",
    }
    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value prepared fast finished",
        "market": "US",
    }

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        saver.at = checkpoint_at

        # test stream_mode=debug
        tool_two = tool_two_graph.compile(checkpointer=saver)
        thread10 = {"configurable": {"thread_id": "10"}}
        if checkpoint_at is CheckpointAt.END_OF_RUN:
            assert [
                c
                async for c in tool_two.astream(
                    {"my_key": "value", "market": "DE"}, thread10, stream_mode="debug"
                )
            ] == [
                {
                    "type": "checkpoint",
                    "timestamp": AnyStr(),
                    "step": 0,
                    "payload": {
                        "config": None,
                        "values": {"my_key": "value", "market": "DE"},
                    },
                },
                {
                    "type": "task",
                    "timestamp": AnyStr(),
                    "step": 1,
                    "payload": {
                        "id": "e7879e70-6335-5867-9ec6-957fbb3da6fa",
                        "name": "prepare",
                        "input": {"my_key": "value", "market": "DE"},
                        "triggers": ["start:prepare"],
                    },
                },
                {
                    "type": "task_result",
                    "timestamp": AnyStr(),
                    "step": 1,
                    "payload": {
                        "id": "e7879e70-6335-5867-9ec6-957fbb3da6fa",
                        "name": "prepare",
                        "result": [("my_key", " prepared")],
                    },
                },
                {
                    "type": "checkpoint",
                    "timestamp": AnyStr(),
                    "step": 1,
                    "payload": {
                        "config": None,
                        "values": {"my_key": "value prepared", "market": "DE"},
                    },
                },
                {
                    "type": "task",
                    "timestamp": AnyStr(),
                    "step": 2,
                    "payload": {
                        "id": "122f31bd-0e14-5b8f-91e7-4f241047a3fd",
                        "name": "tool_two_slow",
                        "input": {"my_key": "value prepared", "market": "DE"},
                        "triggers": ["branch:prepare:condition:tool_two_slow"],
                    },
                },
                {
                    "type": "task_result",
                    "timestamp": AnyStr(),
                    "step": 2,
                    "payload": {
                        "id": "122f31bd-0e14-5b8f-91e7-4f241047a3fd",
                        "name": "tool_two_slow",
                        "result": [("my_key", " slow")],
                    },
                },
                {
                    "type": "checkpoint",
                    "timestamp": AnyStr(),
                    "step": 2,
                    "payload": {
                        "config": None,
                        "values": {"my_key": "value prepared slow", "market": "DE"},
                    },
                },
                {
                    "type": "task",
                    "timestamp": AnyStr(),
                    "step": 3,
                    "payload": {
                        "id": "48a16051-2c14-5ff5-9cfe-e8c7c32d5c83",
                        "name": "finish",
                        "input": {"my_key": "value prepared slow", "market": "DE"},
                        "triggers": ["branch:prepare:condition:then"],
                    },
                },
                {
                    "type": "task_result",
                    "timestamp": AnyStr(),
                    "step": 3,
                    "payload": {
                        "id": "48a16051-2c14-5ff5-9cfe-e8c7c32d5c83",
                        "name": "finish",
                        "result": [("my_key", " finished")],
                    },
                },
                {
                    "type": "checkpoint",
                    "timestamp": AnyStr(),
                    "step": 3,
                    "payload": {
                        "config": None,
                        "values": {
                            "my_key": "value prepared slow finished",
                            "market": "DE",
                        },
                    },
                },
                {
                    "type": "checkpoint",
                    "timestamp": AnyStr(),
                    "step": 4,
                    "payload": {
                        "config": {
                            "configurable": {
                                "thread_id": "10",
                                "thread_ts": AnyStr(),
                            }
                        },
                        "values": {
                            "my_key": "value prepared slow finished",
                            "market": "DE",
                        },
                    },
                },
            ]
        else:
            assert [
                c
                async for c in tool_two.astream(
                    {"my_key": "value", "market": "DE"}, thread10, stream_mode="debug"
                )
            ] == [
                {
                    "type": "checkpoint",
                    "timestamp": AnyStr(),
                    "step": 0,
                    "payload": {
                        "config": {
                            "configurable": {"thread_id": "10", "thread_ts": AnyStr()}
                        },
                        "values": {"my_key": "value", "market": "DE"},
                    },
                },
                {
                    "type": "task",
                    "timestamp": AnyStr(),
                    "step": 1,
                    "payload": {
                        "id": "e7879e70-6335-5867-9ec6-957fbb3da6fa",
                        "name": "prepare",
                        "input": {"my_key": "value", "market": "DE"},
                        "triggers": ["start:prepare"],
                    },
                },
                {
                    "type": "task_result",
                    "timestamp": AnyStr(),
                    "step": 1,
                    "payload": {
                        "id": "e7879e70-6335-5867-9ec6-957fbb3da6fa",
                        "name": "prepare",
                        "result": [("my_key", " prepared")],
                    },
                },
                {
                    "type": "checkpoint",
                    "timestamp": AnyStr(),
                    "step": 1,
                    "payload": {
                        "config": {
                            "configurable": {"thread_id": "10", "thread_ts": AnyStr()}
                        },
                        "values": {"my_key": "value prepared", "market": "DE"},
                    },
                },
                {
                    "type": "task",
                    "timestamp": AnyStr(),
                    "step": 2,
                    "payload": {
                        "id": "122f31bd-0e14-5b8f-91e7-4f241047a3fd",
                        "name": "tool_two_slow",
                        "input": {"my_key": "value prepared", "market": "DE"},
                        "triggers": ["branch:prepare:condition:tool_two_slow"],
                    },
                },
                {
                    "type": "task_result",
                    "timestamp": AnyStr(),
                    "step": 2,
                    "payload": {
                        "id": "122f31bd-0e14-5b8f-91e7-4f241047a3fd",
                        "name": "tool_two_slow",
                        "result": [("my_key", " slow")],
                    },
                },
                {
                    "type": "checkpoint",
                    "timestamp": AnyStr(),
                    "step": 2,
                    "payload": {
                        "config": {
                            "configurable": {"thread_id": "10", "thread_ts": AnyStr()}
                        },
                        "values": {"my_key": "value prepared slow", "market": "DE"},
                    },
                },
                {
                    "type": "task",
                    "timestamp": AnyStr(),
                    "step": 3,
                    "payload": {
                        "id": "48a16051-2c14-5ff5-9cfe-e8c7c32d5c83",
                        "name": "finish",
                        "input": {"my_key": "value prepared slow", "market": "DE"},
                        "triggers": ["branch:prepare:condition:then"],
                    },
                },
                {
                    "type": "task_result",
                    "timestamp": AnyStr(),
                    "step": 3,
                    "payload": {
                        "id": "48a16051-2c14-5ff5-9cfe-e8c7c32d5c83",
                        "name": "finish",
                        "result": [("my_key", " finished")],
                    },
                },
                {
                    "type": "checkpoint",
                    "timestamp": AnyStr(),
                    "step": 3,
                    "payload": {
                        "config": {
                            "configurable": {"thread_id": "10", "thread_ts": AnyStr()}
                        },
                        "values": {
                            "my_key": "value prepared slow finished",
                            "market": "DE",
                        },
                    },
                },
            ]

        tool_two = tool_two_graph.compile(
            checkpointer=saver, interrupt_before=["tool_two_fast", "tool_two_slow"]
        )

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            await tool_two.ainvoke({"my_key": "value", "market": "DE"})

        thread1 = {"configurable": {"thread_id": "1"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}, thread1) == {
            "my_key": "value prepared",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared", "market": "DE"},
            next=("tool_two_slow",),
            config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
            metadata={},
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread1, debug=1) == {
            "my_key": "value prepared slow finished",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared slow finished", "market": "DE"},
            next=(),
            config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
            metadata={},
        )

        thread2 = {"configurable": {"thread_id": "2"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread2) == {
            "my_key": "value prepared",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared", "market": "US"},
            next=("tool_two_fast",),
            config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
            metadata={},
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread2, debug=1) == {
            "my_key": "value prepared fast finished",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared fast finished", "market": "US"},
            next=(),
            config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
            metadata={},
        )

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        saver.at = checkpoint_at
        tool_two = tool_two_graph.compile(
            checkpointer=saver, interrupt_after=["prepare"]
        )

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            await tool_two.ainvoke({"my_key": "value", "market": "DE"})

        thread1 = {"configurable": {"thread_id": "1"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}, thread1) == {
            "my_key": "value prepared",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared", "market": "DE"},
            next=("tool_two_slow",),
            config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
            metadata={},
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread1, debug=1) == {
            "my_key": "value prepared slow finished",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared slow finished", "market": "DE"},
            next=(),
            config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
            metadata={},
        )

        thread2 = {"configurable": {"thread_id": "2"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread2) == {
            "my_key": "value prepared",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared", "market": "US"},
            next=("tool_two_fast",),
            config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
            metadata={},
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread2, debug=1) == {
            "my_key": "value prepared fast finished",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared fast finished", "market": "US"},
            next=(),
            config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
            metadata={},
        )


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_in_one_fan_out_state_graph_waiting_edge(
    checkpoint_at: CheckpointAt,
) -> None:
    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(at=checkpoint_at),
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"query": "what is weather in sf"}, config
        )
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
    ]

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_in_one_fan_out_state_graph_waiting_edge_via_branch(
    snapshot: SnapshotAssertion,
    checkpoint_at: CheckpointAt,
) -> None:
    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_conditional_edges(
        "rewrite_query", lambda _: "retriever_two", {"retriever_two": "retriever_two"}
    )
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.get_graph().draw_ascii() == snapshot

    assert await app.ainvoke({"query": "what is weather in sf"}, debug=True) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(at=checkpoint_at),
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"query": "what is weather in sf"}, config
        )
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
    ]

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class(
    snapshot: SnapshotAssertion,
    checkpoint_at: CheckpointAt,
) -> None:
    from langchain_core.pydantic_v1 import BaseModel, ValidationError

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(BaseModel):
        query: str
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]

    async def rewrite_query(data: State) -> State:
        return {"query": f"query: {data.query}"}

    async def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data.query}"}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data.docs)}

    async def decider(data: State) -> str:
        assert isinstance(data, State)
        return "retriever_two"

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_conditional_edges(
        "rewrite_query", decider, {"retriever_two": "retriever_two"}
    )
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.get_graph().draw_ascii() == snapshot

    with pytest.raises(ValidationError):
        await app.ainvoke({"query": {}})

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(at=checkpoint_at),
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"query": "what is weather in sf"}, config
        )
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
    ]

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


@pytest.mark.parametrize(
    "checkpoint_at", [CheckpointAt.END_OF_RUN, CheckpointAt.END_OF_STEP]
)
async def test_in_one_fan_out_state_graph_waiting_edge_plus_regular(
    checkpoint_at: CheckpointAt,
) -> None:
    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    # silly edge, to make sure having been triggered before doesn't break
    # semantics of named barrier (== waiting edges)
    workflow.add_edge("rewrite_query", "qa")

    app = workflow.compile()

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
            "qa": {"answer": ""},
        },
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(at=checkpoint_at),
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"query": "what is weather in sf"}, config
        )
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
            "qa": {"answer": ""},
        },
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
    ]

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


async def test_in_one_fan_out_state_graph_waiting_edge_multiple() -> None:
    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    async def decider(data: State) -> None:
        return None

    def decider_cond(data: State) -> str:
        if data["query"].count("analyzed") > 1:
            return "qa"
        else:
            return "rewrite_query"

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("decider", decider)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    workflow.add_edge(["retriever_one", "retriever_two"], "decider")
    workflow.add_conditional_edges("decider", decider_cond)
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: analyzed: query: what is weather in sf",
        "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"rewrite_query": {"query": "query: analyzed: query: what is weather in sf"}},
        {
            "analyzer_one": {
                "query": "analyzed: query: analyzed: query: what is weather in sf"
            },
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {
            "retriever_one": {"docs": ["doc1", "doc2"]},
        },
        {"qa": {"answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4"}},
    ]


async def test_in_one_fan_out_state_graph_waiting_edge_multiple_cond_edge() -> None:
    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def retriever_picker(data: State) -> list[str]:
        return ["analyzer_one", "retriever_two"]

    async def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    async def decider(data: State) -> None:
        return None

    def decider_cond(data: State) -> str:
        if data["query"].count("analyzed") > 1:
            return "qa"
        else:
            return "rewrite_query"

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("decider", decider)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_conditional_edges("rewrite_query", retriever_picker)
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge(["retriever_one", "retriever_two"], "decider")
    workflow.add_conditional_edges("decider", decider_cond)
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: analyzed: query: what is weather in sf",
        "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"rewrite_query": {"query": "query: analyzed: query: what is weather in sf"}},
        {
            "analyzer_one": {
                "query": "analyzed: query: analyzed: query: what is weather in sf"
            },
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {
            "retriever_one": {"docs": ["doc1", "doc2"]},
        },
        {"qa": {"answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4"}},
    ]


async def test_nested_graph(snapshot: SnapshotAssertion) -> None:
    def never_called_fn(state: Any):
        assert 0, "This function should never be called"

    never_called = RunnableLambda(never_called_fn)

    class InnerState(TypedDict):
        my_key: str
        my_other_key: str

    def up(state: InnerState):
        return {"my_key": state["my_key"] + " there", "my_other_key": state["my_key"]}

    inner = StateGraph(InnerState)
    inner.add_node("up", up)
    inner.set_entry_point("up")
    inner.set_finish_point("up")

    class State(TypedDict):
        my_key: str
        never_called: Any

    async def side(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile())
    graph.add_node("side", side)
    graph.set_entry_point("inner")
    graph.add_edge("inner", "side")
    graph.set_finish_point("side")

    app = graph.compile()

    assert app.get_graph().draw_ascii() == snapshot
    assert await app.ainvoke({"my_key": "my value", "never_called": never_called}) == {
        "my_key": "my value there and back again",
        "never_called": never_called,
    }
    assert [
        chunk
        async for chunk in app.astream(
            {"my_key": "my value", "never_called": never_called}
        )
    ] == [
        {"inner": {"my_key": "my value there"}},
        {"side": {"my_key": "my value there and back again"}},
    ]
    assert [
        chunk
        async for chunk in app.astream(
            {"my_key": "my value", "never_called": never_called}, stream_mode="values"
        )
    ] == [
        {"my_key": "my value", "never_called": never_called},
        {"my_key": "my value there", "never_called": never_called},
        {"my_key": "my value there and back again", "never_called": never_called},
    ]
    times_called = 0
    async for event in app.astream_events(
        {"my_key": "my value", "never_called": never_called},
        version="v1",
        config={"run_id": UUID(int=0)},
        stream_mode="values",
    ):
        if event["event"] == "on_chain_end" and event["run_id"] == str(UUID(int=0)):
            times_called += 1
            assert event["data"] == {
                "output": {
                    "my_key": "my value there and back again",
                    "never_called": never_called,
                }
            }
    assert times_called == 1
    times_called = 0
    async for event in app.astream_events(
        {"my_key": "my value", "never_called": never_called},
        version="v1",
        config={"run_id": UUID(int=0)},
    ):
        if event["event"] == "on_chain_end" and event["run_id"] == str(UUID(int=0)):
            times_called += 1
            assert event["data"] == {
                "output": [
                    {"inner": {"my_key": "my value there"}},
                    {"side": {"my_key": "my value there and back again"}},
                ]
            }
    assert times_called == 1

    chain = app | RunnablePassthrough()

    assert await chain.ainvoke(
        {"my_key": "my value", "never_called": never_called}
    ) == {
        "my_key": "my value there and back again",
        "never_called": never_called,
    }
    assert [
        chunk
        async for chunk in chain.astream(
            {"my_key": "my value", "never_called": never_called}
        )
    ] == [
        {"inner": {"my_key": "my value there"}},
        {"side": {"my_key": "my value there and back again"}},
    ]
    times_called = 0
    async for event in chain.astream_events(
        {"my_key": "my value", "never_called": never_called},
        version="v1",
        config={"run_id": UUID(int=0)},
    ):
        if event["event"] == "on_chain_end" and event["run_id"] == str(UUID(int=0)):
            times_called += 1
            assert event["data"] == {
                "output": [
                    {"inner": {"my_key": "my value there"}},
                    {"side": {"my_key": "my value there and back again"}},
                ]
            }
    assert times_called == 1
