import json
import operator
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Annotated, Generator, Optional, TypedDict, Union

import pytest
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion

from langgraph.channels.base import InvalidUpdateError
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.context import Context
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, Graph
from langgraph.graph.message import MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.prebuilt.chat_agent_executor import (
    create_function_calling_executor,
    create_tool_calling_executor,
)
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.pregel import Channel, GraphRecursionError, Pregel, StateSnapshot
from langgraph.pregel.reserved import ReservedChannels
from tests.memory_assert import MemorySaverAssertImmutable


def test_invoke_single_process_in_out(mocker: MockerFixture) -> None:
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
        input="input",
        output="output",
    )
    graph = Graph()
    graph.add_node("add_one", add_one)
    graph.set_entry_point("add_one")
    graph.set_finish_point("add_one")
    gapp = graph.compile()

    assert app.input_schema.schema() == {"title": "LangGraphInput", "type": "integer"}
    assert app.output_schema.schema() == {"title": "LangGraphOutput", "type": "integer"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # raise warnings as errors
        assert app.config_schema().schema() == {
            "properties": {},
            "title": "LangGraphConfig",
            "type": "object",
        }
    assert app.invoke(2) == 3
    assert app.invoke(2, output_keys=["output"]) == {"output": 3}
    assert repr(app), "does not raise recursion error"

    assert gapp.invoke(2) == 3


def test_invoke_single_process_in_out_implicit_channels(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(nodes={"one": chain})

    assert app.input_schema.schema() == {"title": "LangGraphInput"}
    assert app.output_schema.schema() == {"title": "LangGraphOutput"}
    assert app.invoke(2) == 3


def test_invoke_single_process_in_write_kwargs(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = (
        Channel.subscribe_to("input")
        | add_one
        | Channel.write_to("output", fixed=5, output_plus_one=lambda x: x + 1)
    )

    app = Pregel(nodes={"one": chain}, output=["output", "fixed", "output_plus_one"])

    assert app.input_schema.schema() == {"title": "LangGraphInput"}
    assert app.output_schema.schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {
            "output": {"title": "Output"},
            "fixed": {"title": "Fixed"},
            "output_plus_one": {"title": "Output Plus One"},
        },
    }
    assert app.invoke(2) == {"output": 3, "fixed": 5, "output_plus_one": 4}


def test_invoke_single_process_in_out_reserved_is_last(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: {**x, "input": x["input"] + 1})

    chain = (
        Channel.subscribe_to(["input"]).join([ReservedChannels.is_last_step])
        | add_one
        | Channel.write_to("output")
    )

    app = Pregel(nodes={"one": chain})

    assert app.input_schema.schema() == {"title": "LangGraphInput"}
    assert app.output_schema.schema() == {"title": "LangGraphOutput"}
    assert app.invoke(2) == {"input": 3, "is_last_step": False}
    assert app.invoke(2, {"recursion_limit": 1}) == {"input": 3, "is_last_step": True}


def test_invoke_single_process_in_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={
            "one": chain,
        },
        output=["output"],
    )

    assert app.input_schema.schema() == {"title": "LangGraphInput"}
    assert app.output_schema.schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {"output": {"title": "Output"}},
    }
    assert app.invoke(2) == {"output": 3}


def test_invoke_single_process_in_dict_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={
            "one": chain,
        },
        input=["input"],
        output=["output"],
    )

    assert app.input_schema.schema() == {
        "title": "LangGraphInput",
        "type": "object",
        "properties": {"input": {"title": "Input"}},
    }
    assert app.output_schema.schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {"output": {"title": "Output"}},
    }
    assert app.invoke({"input": 2}) == {"output": 3}


def test_invoke_two_processes_in_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = Channel.subscribe_to("inbox") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
    )

    assert app.invoke(2) == 4

    assert app.invoke(2, input_keys="inbox") == 3

    with pytest.raises(GraphRecursionError):
        app.invoke(2, {"recursion_limit": 1})

    for step, values in enumerate(app.stream(2), start=1):
        if step == 1:
            assert values == {
                "inbox": 3,
            }
        elif step == 2:
            assert values == {
                "output": 4,
            }

    for step, values in enumerate(app.stream(2), start=1):
        if step == 1:
            assert values == {
                "inbox": 3,
            }
            # modify inbox value
            values["inbox"] = 5
        elif step == 2:
            # output is different now
            assert values == {
                "output": 6,
            }

    graph = Graph()
    graph.add_node("add_one", add_one)
    graph.add_node("add_one_more", add_one)
    graph.set_entry_point("add_one")
    graph.set_finish_point("add_one_more")
    graph.add_edge("add_one", "add_one_more")
    gapp = graph.compile()

    assert gapp.invoke(2) == 4

    for step, values in enumerate(gapp.stream(2), start=1):
        if step == 1:
            assert values == {
                "add_one": 3,
            }
        elif step == 2:
            assert values == {
                "add_one_more": 4,
            }
        elif step == 3:
            assert values == {
                "__end__": 4,
            }
        else:
            assert 0, f"{step}:{values}"
    assert step == 3

    for step, values in enumerate(gapp.stream(2), start=1):
        if step == 1:
            assert values == {
                "add_one": 3,
            }
            # modify value before next step
            values["add_one"] = 5
        elif step == 2:
            assert values == {
                "add_one_more": 6,
            }
        elif step == 3:
            assert values == {
                "__end__": 6,
            }
        else:
            assert 0, "Should not get here"
    assert step == 3


def test_invoke_two_processes_in_out_interrupt(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = Channel.subscribe_to("inbox") | add_one | Channel.write_to("output")

    memory = MemorySaverAssertImmutable()
    app = Pregel(
        nodes={"one": one, "two": two},
        checkpointer=memory,
        interrupt_after_nodes=["inbox"],
    )

    # start execution, stop at inbox
    assert app.invoke(2, {"configurable": {"thread_id": 1}}) is None

    # inbox == 3
    checkpoint = memory.get({"configurable": {"thread_id": 1}})
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 3

    # resume execution, finish
    assert app.invoke(None, {"configurable": {"thread_id": 1}}) == 4

    # start execution again, stop at inbox
    assert app.invoke(20, {"configurable": {"thread_id": 1}}) is None

    # inbox == 21
    checkpoint = memory.get({"configurable": {"thread_id": 1}})
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 21

    # send a new value in, interrupting the previous execution
    assert app.invoke(3, {"configurable": {"thread_id": 1}}) is None
    assert app.invoke(None, {"configurable": {"thread_id": 1}}) == 5

    # start execution again, stopping at inbox
    assert app.invoke(20, {"configurable": {"thread_id": 2}}) is None

    # inbox == 21
    snapshot = app.get_state({"configurable": {"thread_id": 2}})
    assert snapshot.values["inbox"] == 21
    assert snapshot.next == ("two",)

    # update the state, resume
    app.update_state({"configurable": {"thread_id": 2}}, {"inbox": 25})
    assert app.invoke(None, {"configurable": {"thread_id": 2}}) == 26

    # no pending tasks
    snapshot = app.get_state({"configurable": {"thread_id": 2}})
    assert snapshot.next == ()


def test_invoke_two_processes_in_dict_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = (
        Channel.subscribe_to("inbox")
        | RunnableLambda(add_one).batch
        | Channel.write_to("output").batch
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={"inbox": Topic(int)},
        input=["input", "inbox"],
    )

    assert [*app.stream({"input": 2, "inbox": 12}, output_keys="output")] == [
        13,
        4,
    ]  # [12 + 1, 2 + 1 + 1]
    assert [*app.stream({"input": 2, "inbox": 12})] == [
        {"inbox": [3], "output": 13},
        {"output": 4},
    ]


def test_batch_two_processes_in_out() -> None:
    def add_one_with_delay(inp: int) -> int:
        time.sleep(inp / 10)
        return inp + 1

    one = Channel.subscribe_to("input") | add_one_with_delay | Channel.write_to("one")
    two = Channel.subscribe_to("one") | add_one_with_delay | Channel.write_to("output")

    app = Pregel(nodes={"one": one, "two": two})

    assert app.batch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]
    assert app.batch([3, 2, 1, 3, 5], output_keys=["output"]) == [
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

    assert gapp.batch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]


def test_invoke_many_processes_in_out(mocker: MockerFixture) -> None:
    test_size = 100
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    nodes = {"-1": Channel.subscribe_to("input") | add_one | Channel.write_to("-1")}
    for i in range(test_size - 2):
        nodes[str(i)] = (
            Channel.subscribe_to(str(i - 1)) | add_one | Channel.write_to(str(i))
        )
    nodes["last"] = Channel.subscribe_to(str(i)) | add_one | Channel.write_to("output")

    app = Pregel(nodes=nodes)

    for _ in range(10):
        assert app.invoke(2, {"recursion_limit": test_size}) == 2 + test_size

    with ThreadPoolExecutor() as executor:
        assert [
            *executor.map(app.invoke, [2] * 10, [{"recursion_limit": test_size}] * 10)
        ] == [2 + test_size] * 10


def test_batch_many_processes_in_out(mocker: MockerFixture) -> None:
    test_size = 100
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    nodes = {"-1": Channel.subscribe_to("input") | add_one | Channel.write_to("-1")}
    for i in range(test_size - 2):
        nodes[str(i)] = (
            Channel.subscribe_to(str(i - 1)) | add_one | Channel.write_to(str(i))
        )
    nodes["last"] = Channel.subscribe_to(str(i)) | add_one | Channel.write_to("output")

    app = Pregel(nodes=nodes)

    for _ in range(3):
        assert app.batch([2, 1, 3, 4, 5], {"recursion_limit": test_size}) == [
            2 + test_size,
            1 + test_size,
            3 + test_size,
            4 + test_size,
            5 + test_size,
        ]

    with ThreadPoolExecutor() as executor:
        assert [
            *executor.map(
                app.batch, [[2, 1, 3, 4, 5]] * 3, [{"recursion_limit": test_size}] * 3
            )
        ] == [
            [2 + test_size, 1 + test_size, 3 + test_size, 4 + test_size, 5 + test_size]
        ] * 3


def test_invoke_two_processes_two_in_two_out_invalid(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = Channel.subscribe_to("input") | add_one | Channel.write_to("output")
    two = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(nodes={"one": one, "two": two})

    with pytest.raises(InvalidUpdateError):
        # LastValue channels can only be updated once per iteration
        app.invoke(2)


def test_invoke_two_processes_two_in_two_out_valid(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = Channel.subscribe_to("input") | add_one | Channel.write_to("output")
    two = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={"output": Topic(int)},
    )

    # An Inbox channel accumulates updates into a sequence
    assert app.invoke(2) == [3, 3]


def test_invoke_checkpoint(mocker: MockerFixture) -> None:
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

    memory = MemorySaverAssertImmutable()

    app = Pregel(
        nodes={"one": one},
        channels={"total": BinaryOperatorAggregate(int, operator.add)},
        checkpointer=memory,
    )

    # total starts out as 0, so output is 0+2=2
    assert app.invoke(2, {"configurable": {"thread_id": "1"}}) == 2
    checkpoint = memory.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 2
    # total is now 2, so output is 2+3=5
    assert app.invoke(3, {"configurable": {"thread_id": "1"}}) == 5
    checkpoint = memory.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        app.invoke(4, {"configurable": {"thread_id": "1"}})
    # checkpoint is not updated
    checkpoint = memory.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert app.invoke(5, {"configurable": {"thread_id": "2"}}) == 5
    checkpoint = memory.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    checkpoint = memory.get({"configurable": {"thread_id": "2"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 5


def test_invoke_checkpoint_sqlite(mocker: MockerFixture) -> None:
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

    with SqliteSaver.from_conn_string(":memory:") as memory:
        app = Pregel(
            nodes={"one": one},
            channels={"total": BinaryOperatorAggregate(int, operator.add)},
            checkpointer=memory,
        )

        thread_1 = {"configurable": {"thread_id": "1"}}
        # total starts out as 0, so output is 0+2=2
        assert app.invoke(2, thread_1) == 2
        state = app.get_state(thread_1)
        assert state is not None
        assert state.values.get("total") == 2
        assert state.config["configurable"]["thread_ts"] == memory.get(thread_1)["ts"]
        # total is now 2, so output is 2+3=5
        assert app.invoke(3, thread_1) == 5
        state = app.get_state(thread_1)
        assert state is not None
        assert state.values.get("total") == 7
        assert state.config["configurable"]["thread_ts"] == memory.get(thread_1)["ts"]
        # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
        with pytest.raises(ValueError):
            app.invoke(4, thread_1)
        # checkpoint is not updated
        state = app.get_state(thread_1)
        assert state is not None
        assert state.values.get("total") == 7

        thread_2 = {"configurable": {"thread_id": "2"}}
        # on a new thread, total starts out as 0, so output is 0+5=5
        assert app.invoke(5, thread_2) == 5
        state = app.get_state({"configurable": {"thread_id": "1"}})
        assert state is not None
        assert state.values.get("total") == 7
        state = app.get_state(thread_2)
        assert state is not None
        assert state.values.get("total") == 5

        # list all checkpoints for thread 1
        thread_1_history = [c for c in app.get_state_history(thread_1)]
        # there are 2: one for each successful ainvoke()
        assert len(thread_1_history) == 2
        # sorted descending
        assert (
            thread_1_history[0].config["configurable"]["thread_ts"]
            > thread_1_history[1].config["configurable"]["thread_ts"]
        )
        # the second checkpoint
        assert thread_1_history[0].values["total"] == 7
        # the first checkpoint
        assert thread_1_history[1].values["total"] == 2
        # can get each checkpoint using aget with config
        assert (
            memory.get(thread_1_history[0].config)["ts"]
            == thread_1_history[0].config["configurable"]["thread_ts"]
        )
        assert (
            memory.get(thread_1_history[1].config)["ts"]
            == thread_1_history[1].config["configurable"]["thread_ts"]
        )

        thread_1_next_config = app.update_state(
            thread_1_history[1].config, {"total": 10}
        )
        # update creates a new checkpoint
        assert (
            thread_1_next_config["configurable"]["thread_ts"]
            > thread_1_history[0].config["configurable"]["thread_ts"]
        )
        # 1 more checkpoint in history
        assert len(list(app.get_state_history(thread_1))) == 3
        # the latest checkpoint is the updated one
        assert app.get_state(thread_1) == app.get_state(thread_1_next_config)


def test_invoke_two_processes_two_in_join_two_out(mocker: MockerFixture) -> None:
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
        channels={"inbox": Topic(int)},
    )

    # Then invoke app
    # We get a single array result as chain_four waits for all publishers to finish
    # before operating on all elements published to topic_two as an array
    for _ in range(100):
        assert app.invoke(2) == [13, 13]

    with ThreadPoolExecutor() as executor:
        assert [*executor.map(app.invoke, [2] * 100)] == [[13, 13]] * 100


def test_invoke_join_then_call_other_app(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    inner_app = Pregel(
        nodes={
            "one": Channel.subscribe_to("input") | add_one | Channel.write_to("output")
        }
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
        channels={"inbox_one": Topic(int)},
    )

    for _ in range(10):
        assert app.invoke([2, 3]) == 27

    with ThreadPoolExecutor() as executor:
        assert [*executor.map(app.invoke, [[2, 3]] * 10)] == [27] * 10


def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = (
        Channel.subscribe_to("input")
        | add_one
        | Channel.write_to(output=RunnablePassthrough(), between=RunnablePassthrough())
    )
    two = Channel.subscribe_to("between") | add_one | Channel.write_to("output")

    app = Pregel(nodes={"one": one, "two": two})

    assert [c for c in app.stream(2)] == [{"between": 3, "output": 3}, {"output": 4}]


def test_invoke_two_processes_no_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("between")
    two = Channel.subscribe_to("between") | add_one

    app = Pregel(nodes={"one": one, "two": two})

    # It finishes executing (once no more messages being published)
    # but returns nothing, as nothing was published to OUT topic
    assert app.invoke(2) is None


def test_invoke_two_processes_no_in(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = Channel.subscribe_to("between") | add_one | Channel.write_to("output")
    two = Channel.subscribe_to("between") | add_one

    with pytest.raises(ValueError):
        Pregel(nodes={"one": one, "two": two})


def test_channel_enter_exit_timing(mocker: MockerFixture) -> None:
    setup = mocker.Mock()
    cleanup = mocker.Mock()

    @contextmanager
    def an_int() -> Generator[int, None, None]:
        setup()
        try:
            yield 5
        finally:
            cleanup()

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = (
        Channel.subscribe_to("inbox")
        | RunnableLambda(add_one).batch
        | Channel.write_to("output").batch
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "inbox": Topic(int),
            "ctx": Context(an_int, typ=int),
        },
        output=["inbox", "output"],
    )

    assert setup.call_count == 0
    assert cleanup.call_count == 0
    for i, chunk in enumerate(app.stream(2)):
        assert setup.call_count == 1, "Expected setup to be called once"
        assert cleanup.call_count == 0, "Expected cleanup to not be called yet"
        if i == 0:
            assert chunk == {"inbox": [3]}
        elif i == 1:
            assert chunk == {"output": 4}
        else:
            assert False, "Expected only two chunks"
    assert cleanup.call_count == 1, "Expected cleanup to be called once"


def test_conditional_graph(snapshot: SnapshotAssertion) -> None:
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

    def agent_parser(input: str) -> Union[AgentAction, AgentFinish]:
        if input.startswith("finish"):
            _, answer = input.split(":")
            return AgentFinish(return_values={"answer": answer}, log=input)
        else:
            _, tool_name, tool_input = input.split(":")
            return AgentAction(tool=tool_name, tool_input=tool_input, log=input)

    agent = RunnablePassthrough.assign(agent_outcome=prompt | llm | agent_parser)

    # Define tool execution logic
    def execute_tools(data: dict) -> dict:
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        if data.get("intermediate_steps") is None:
            data["intermediate_steps"] = []
        data["intermediate_steps"].append((agent_action, observation))
        return data

    # Define decision-making logic
    def should_continue(data: dict) -> str:
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

    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_ascii() == snapshot
    assert json.dumps(app.get_graph(xray=True).to_json(), indent=2) == snapshot
    assert app.get_graph(xray=True).draw_ascii() == snapshot

    assert app.invoke({"input": "what is weather in sf"}) == {
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
    assert [deepcopy(c) for c in app.stream({"input": "what is weather in sf"})] == [
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
        {
            "__end__": {
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

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(), interrupt_after=["agent"]
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
            "tools": None,
        },
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )
    assert (
        app_w_interrupt.checkpointer.get_tuple(config).config["configurable"][
            "thread_ts"
        ]
        is not None
    )

    app_w_interrupt.update_state(
        config,
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
            "tools": None,
        },
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    app_w_interrupt.update_state(
        config,
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
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                ),
            }
        },
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "__end__": {
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
            }
        }
    ]

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(), interrupt_before=["tools"]
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
            "tools": None,
        },
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    app_w_interrupt.update_state(
        config,
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
            "tools": None,
        },
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    app_w_interrupt.update_state(
        config,
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
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                ),
            }
        },
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "__end__": {
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
            }
        }
    ]

    # test re-invoke to continue with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(), interrupt_before=["tools"]
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
            "tools": None,
        },
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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
        {
            "__end__": {
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


def test_conditional_graph_state(snapshot: SnapshotAssertion) -> None:
    from langchain.llms.fake import FakeStreamingListLLM
    from langchain_community.tools import tool
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.prompts import PromptTemplate

    class AgentState(TypedDict, total=False):
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

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_ascii() == snapshot

    assert app.invoke({"input": "what is weather in sf"}) == {
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

    assert [*app.stream({"input": "what is weather in sf"})] == [
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
        {
            "__end__": {
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

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(), interrupt_after=["agent"]
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "input": "what is weather in sf",
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            )
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "input": "what is weather in sf",
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            )
        },
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "__end__": {
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
            }
        }
    ]

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_before=["tools"],
        debug=True,
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "input": "what is weather in sf",
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            )
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "input": "what is weather in sf",
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            )
        },
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "__end__": {
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
            }
        }
    ]


def test_conditional_entrypoint_graph(snapshot: SnapshotAssertion) -> None:
    def left(data: str) -> str:
        return data + "->left"

    def right(data: str) -> str:
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

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_ascii() == snapshot

    assert app.invoke("what is weather in sf") == "what is weather in sf->right"

    assert [*app.stream("what is weather in sf")] == [
        {"right": "what is weather in sf->right"},
        {"__end__": "what is weather in sf->right"},
    ]


def test_conditional_entrypoint_graph_state(snapshot: SnapshotAssertion) -> None:
    class AgentState(TypedDict, total=False):
        input: str
        output: str

    def left(data: AgentState) -> AgentState:
        return {"output": data["input"] + "->left"}

    def right(data: AgentState) -> AgentState:
        return {"output": data["input"] + "->right"}

    def should_start(data: AgentState) -> str:
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

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_ascii() == snapshot

    assert app.invoke({"input": "what is weather in sf"}) == {
        "input": "what is weather in sf",
        "output": "what is weather in sf->right",
    }

    assert [*app.stream({"input": "what is weather in sf"})] == [
        {"right": {"output": "what is weather in sf->right"}},
        {
            "__end__": {
                "input": "what is weather in sf",
                "output": "what is weather in sf->right",
            }
        },
    ]


def test_prebuilt_tool_chat(snapshot: SnapshotAssertion) -> None:
    from langchain.chat_models.fake import FakeMessagesListChatModel
    from langchain_community.tools import tool
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
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
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "tool_call123",
                                "type": "function",
                                "function": {
                                    "name": "search_api",
                                    "arguments": json.dumps("query"),
                                },
                            }
                        ]
                    },
                ),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "tool_call234",
                                "type": "function",
                                "function": {
                                    "name": "search_api",
                                    "arguments": json.dumps("another"),
                                },
                            },
                            {
                                "id": "tool_call567",
                                "type": "function",
                                "function": {
                                    "name": "search_api",
                                    "arguments": '"a third one"',
                                },
                            },
                        ]
                    },
                ),
                AIMessage(content="answer"),
            ]
        ),
        tools,
    )

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_ascii() == snapshot

    assert app.invoke(
        {"messages": [HumanMessage(content="what is weather in sf")]}
    ) == {
        "messages": [
            HumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "tool_call123",
                            "type": "function",
                            "function": {
                                "name": "search_api",
                                "arguments": '"query"',
                            },
                        }
                    ]
                },
            ),
            ToolMessage(content="result for query", tool_call_id="tool_call123"),
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "tool_call234",
                            "type": "function",
                            "function": {
                                "name": "search_api",
                                "arguments": '"another"',
                            },
                        },
                        {
                            "id": "tool_call567",
                            "type": "function",
                            "function": {
                                "name": "search_api",
                                "arguments": '"a third one"',
                            },
                        },
                    ]
                },
            ),
            ToolMessage(content="result for another", tool_call_id="tool_call234"),
            ToolMessage(content="result for a third one", tool_call_id="tool_call567"),
            AIMessage(content="answer"),
        ]
    }

    assert [
        *app.stream({"messages": [HumanMessage(content="what is weather in sf")]})
    ] == [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        additional_kwargs={
                            "tool_calls": [
                                {
                                    "id": "tool_call123",
                                    "type": "function",
                                    "function": {
                                        "name": "search_api",
                                        "arguments": '"query"',
                                    },
                                }
                            ]
                        },
                    )
                ]
            }
        },
        {
            "action": {
                "messages": [
                    ToolMessage(content="result for query", tool_call_id="tool_call123")
                ]
            }
        },
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        additional_kwargs={
                            "tool_calls": [
                                {
                                    "id": "tool_call234",
                                    "type": "function",
                                    "function": {
                                        "name": "search_api",
                                        "arguments": '"another"',
                                    },
                                },
                                {
                                    "id": "tool_call567",
                                    "type": "function",
                                    "function": {
                                        "name": "search_api",
                                        "arguments": '"a third one"',
                                    },
                                },
                            ]
                        },
                    )
                ]
            }
        },
        {
            "action": {
                "messages": [
                    ToolMessage(
                        content="result for another", tool_call_id="tool_call234"
                    ),
                    ToolMessage(
                        content="result for a third one", tool_call_id="tool_call567"
                    ),
                ]
            }
        },
        {"agent": {"messages": [AIMessage(content="answer")]}},
        {
            "__end__": {
                "messages": [
                    HumanMessage(content="what is weather in sf"),
                    AIMessage(
                        content="",
                        additional_kwargs={
                            "tool_calls": [
                                {
                                    "id": "tool_call123",
                                    "type": "function",
                                    "function": {
                                        "name": "search_api",
                                        "arguments": '"query"',
                                    },
                                }
                            ]
                        },
                    ),
                    ToolMessage(
                        content="result for query", tool_call_id="tool_call123"
                    ),
                    AIMessage(
                        content="",
                        additional_kwargs={
                            "tool_calls": [
                                {
                                    "id": "tool_call234",
                                    "type": "function",
                                    "function": {
                                        "name": "search_api",
                                        "arguments": '"another"',
                                    },
                                },
                                {
                                    "id": "tool_call567",
                                    "type": "function",
                                    "function": {
                                        "name": "search_api",
                                        "arguments": '"a third one"',
                                    },
                                },
                            ]
                        },
                    ),
                    ToolMessage(
                        content="result for another", tool_call_id="tool_call234"
                    ),
                    ToolMessage(
                        content="result for a third one", tool_call_id="tool_call567"
                    ),
                    AIMessage(content="answer"),
                ]
            }
        },
    ]


def test_prebuilt_chat(snapshot: SnapshotAssertion) -> None:
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

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_ascii() == snapshot

    assert app.invoke(
        {"messages": [HumanMessage(content="what is weather in sf")]}
    ) == {
        "messages": [
            HumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"query"'}
                },
            ),
            FunctionMessage(content="result for query", name="search_api"),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"another"'}
                },
            ),
            FunctionMessage(content="result for another", name="search_api"),
            AIMessage(content="answer"),
        ]
    }

    assert [
        *app.stream({"messages": [HumanMessage(content="what is weather in sf")]})
    ] == [
        {
            "agent": {
                "messages": [
                    AIMessage(
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
                    FunctionMessage(content="result for query", name="search_api")
                ]
            }
        },
        {
            "agent": {
                "messages": [
                    AIMessage(
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
                    FunctionMessage(content="result for another", name="search_api")
                ]
            }
        },
        {"agent": {"messages": [AIMessage(content="answer")]}},
        {
            "__end__": {
                "messages": [
                    HumanMessage(content="what is weather in sf"),
                    AIMessage(
                        content="",
                        additional_kwargs={
                            "function_call": {
                                "name": "search_api",
                                "arguments": '"query"',
                            }
                        },
                    ),
                    FunctionMessage(content="result for query", name="search_api"),
                    AIMessage(
                        content="",
                        additional_kwargs={
                            "function_call": {
                                "name": "search_api",
                                "arguments": '"another"',
                            }
                        },
                    ),
                    FunctionMessage(content="result for another", name="search_api"),
                    AIMessage(content="answer"),
                ]
            }
        },
    ]


def test_message_graph(
    snapshot: SnapshotAssertion, deterministic_uuids: MockerFixture
) -> None:
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

    def call_tool(messages):
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
        response = tool_executor.invoke(action)
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

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_ascii() == snapshot

    assert app.invoke(HumanMessage(content="what is weather in sf")) == [
        HumanMessage(
            content="what is weather in sf",
            id="00000000-0000-4000-8000-000000000002",  # adds missing ids
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {"name": "search_api", "arguments": '"query"'}
            },
            id="ai1",  # respects ids passed in
        ),
        FunctionMessage(
            content="result for query",
            name="search_api",
            id="00000000-0000-4000-8000-000000000014",
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {"name": "search_api", "arguments": '"another"'}
            },
            id="ai2",
        ),
        FunctionMessage(
            content="result for another",
            name="search_api",
            id="00000000-0000-4000-8000-000000000026",
        ),
        AIMessage(content="answer", id="ai3"),
    ]

    assert [*app.stream([HumanMessage(content="what is weather in sf")])] == [
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
                content="result for query",
                name="search_api",
                id="00000000-0000-4000-8000-000000000047",
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
                content="result for another",
                name="search_api",
                id="00000000-0000-4000-8000-000000000059",
            )
        },
        {"agent": AIMessage(content="answer", id="ai3")},
        {
            "__end__": [
                HumanMessage(
                    content="what is weather in sf",
                    id="00000000-0000-4000-8000-000000000035",
                ),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {"name": "search_api", "arguments": '"query"'}
                    },
                    id="ai1",
                ),
                FunctionMessage(
                    content="result for query",
                    name="search_api",
                    id="00000000-0000-4000-8000-000000000047",
                ),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "search_api",
                            "arguments": '"another"',
                        }
                    },
                    id="ai2",
                ),
                FunctionMessage(
                    content="result for another",
                    name="search_api",
                    id="00000000-0000-4000-8000-000000000059",
                ),
                AIMessage(content="answer", id="ai3"),
            ]
        },
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(), interrupt_after=["agent"]
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        for c in app_w_interrupt.stream(
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
        }
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            HumanMessage(
                content="what is weather in sf",
                id="00000000-0000-4000-8000-000000000068",
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"query"'}
                },
                id="ai1",
            ),
        ],
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.additional_kwargs["function_call"]["arguments"] = '"a different query"'
    app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            HumanMessage(
                content="what is weather in sf",
                id="00000000-0000-4000-8000-000000000068",
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
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "action": FunctionMessage(
                content="result for a different query",
                name="search_api",
                id="00000000-0000-4000-8000-000000000081",
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            HumanMessage(
                content="what is weather in sf",
                id="00000000-0000-4000-8000-000000000068",
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
                id="00000000-0000-4000-8000-000000000081",
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"another"'}
                },
                id="ai2",
            ),
        ],
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            HumanMessage(
                content="what is weather in sf",
                id="00000000-0000-4000-8000-000000000068",
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
                id="00000000-0000-4000-8000-000000000081",
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        next=("agent:edges",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "__end__": [
                HumanMessage(
                    content="what is weather in sf",
                    id="00000000-0000-4000-8000-000000000068",
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
                    id="00000000-0000-4000-8000-000000000081",
                ),
                AIMessage(content="answer", id="ai2"),
            ]
        }
    ]


def test_in_one_fan_out_out_one_graph_state() -> None:
    def sorted_add(x: list[str], y: list[str]) -> list[str]:
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
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

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "retriever_two": {"docs": ["doc3", "doc4"]},
            "retriever_one": {"docs": ["doc1", "doc2"]},
        },
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        {
            "__end__": {
                "query": "query: what is weather in sf",
                "answer": "doc1,doc2,doc3,doc4",
                "docs": ["doc1", "doc2", "doc3", "doc4"],
            }
        },
    ]


def test_in_one_fan_out_waiting_edge_graph_state() -> None:
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

    def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
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
    workflow.add_waiting_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}, debug=True) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {
            "analyzer_one": {"query": "analyzed: query: what is weather in sf"},
            "retriever_two": {"docs": ["doc3", "doc4"]},
        },
        {"('retriever_one', 'retriever_two'):qa": None},
        {"retriever_one": {"docs": ["doc1", "doc2"]}, "qa": {"answer": "doc3,doc4"}},
        {
            "('retriever_one', 'retriever_two'):qa": None,
            "__end__": {
                "query": "analyzed: query: what is weather in sf",
                "answer": "doc3,doc4",
                "docs": ["doc1", "doc2", "doc3", "doc4"],
            },
        },
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        {
            "__end__": {
                "query": "analyzed: query: what is weather in sf",
                "answer": "doc1,doc2,doc3,doc4",
                "docs": ["doc1", "doc2", "doc3", "doc4"],
            }
        },
    ]
