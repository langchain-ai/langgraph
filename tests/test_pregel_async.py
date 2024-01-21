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
    Self,
    TypedDict,
    Union,
)

import pytest
from langchain_core.runnables import RunnablePassthrough
from pytest_mock import MockerFixture

from langgraph.channels.base import InvalidUpdateError
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.context import Context
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, Graph, StateGraph
from langgraph.graph.message import MessageGraph
from langgraph.prebuilt.chat_agent_executor import create_function_calling_executor
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.pregel import Channel, GraphRecursionError, Pregel
from langgraph.pregel.reserved import ReservedChannels


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
    assert await app.ainvoke(2) == 3
    assert await app.ainvoke(2, output_keys=["output"]) == {"output": 3}

    assert await gapp.ainvoke(2) == 3


async def test_invoke_single_process_in_out_implicit_channels(
    mocker: MockerFixture,
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(nodes={"one": chain})

    assert app.input_schema.schema() == {"title": "LangGraphInput"}
    assert app.output_schema.schema() == {"title": "LangGraphOutput"}
    assert await app.ainvoke(2) == 3


async def test_invoke_single_process_in_write_kwargs(mocker: MockerFixture) -> None:
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
    assert await app.ainvoke(2) == {"output": 3, "fixed": 5, "output_plus_one": 4}


async def test_invoke_single_process_in_out_reserved_is_last(
    mocker: MockerFixture,
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: {**x, "input": x["input"] + 1})

    chain = (
        Channel.subscribe_to(["input"]).join([ReservedChannels.is_last_step])
        | add_one
        | Channel.write_to("output")
    )

    app = Pregel(nodes={"one": chain})

    assert app.input_schema.schema() == {"title": "LangGraphInput"}
    assert app.output_schema.schema() == {"title": "LangGraphOutput"}
    assert await app.ainvoke(2) == {"input": 3, "is_last_step": False}
    assert await app.ainvoke(2, {"recursion_limit": 1}) == {
        "input": 3,
        "is_last_step": True,
    }


async def test_invoke_single_process_in_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": chain},
        output=["output"],
    )

    assert app.input_schema.schema() == {"title": "LangGraphInput"}
    assert app.output_schema.schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {"output": {"title": "Output"}},
    }
    assert await app.ainvoke(2) == {"output": 3}


async def test_invoke_single_process_in_dict_out_dict(mocker: MockerFixture) -> None:
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
    assert await app.ainvoke({"input": 2}) == {"output": 3}


async def test_invoke_two_processes_in_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = Channel.subscribe_to("inbox") | add_one | Channel.write_to("output")

    app = Pregel(nodes={"one": one, "two": two})

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
                "output": 4,
            }
    assert step == 2

    step = 0
    async for values in app.astream(2):
        step += 1
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
        elif step == 3:
            assert values == {
                "__end__": 4,
            }
    assert step == 3

    step = 0
    async for values in gapp.astream(2):
        step += 1
        if step == 1:
            assert values == {
                "add_one": 3,
            }
            # modify value before running next step
            values["add_one"] = 5
        elif step == 2:
            # output is different now
            assert values == {
                "add_one_more": 6,
            }
        elif step == 3:
            assert values == {
                "__end__": 6,
            }
    assert step == 3


async def test_invoke_two_processes_in_dict_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = Channel.subscribe_to_each("inbox") | add_one | Channel.write_to("output")

    pubsub = Pregel(
        nodes={"one": one, "two": two},
        channels={"inbox": Topic(int)},
        input=["input", "inbox"],
    )

    # [12 + 1, 2 + 1 + 1]
    assert [
        c async for c in pubsub.astream({"input": 2, "inbox": 12}, output_keys="output")
    ] == [13, 4]
    assert [c async for c in pubsub.astream({"input": 2, "inbox": 12})] == [
        {"inbox": [3], "output": 13},
        {"output": 4},
    ]


async def test_batch_two_processes_in_out() -> None:
    async def add_one_with_delay(inp: int) -> int:
        await asyncio.sleep(inp / 10)
        return inp + 1

    one = Channel.subscribe_to("input") | add_one_with_delay | Channel.write_to("one")
    two = Channel.subscribe_to("one") | add_one_with_delay | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={"one": LastValue(int)},
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

    app = Pregel(nodes=nodes)

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

    app = Pregel(nodes=nodes)

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

    app = Pregel(nodes={"one": one, "two": two})

    with pytest.raises(InvalidUpdateError):
        # LastValue channels can only be updated once per iteration
        await app.ainvoke(2)


async def test_invoke_two_processes_two_in_two_out_valid(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = Channel.subscribe_to("input") | add_one | Channel.write_to("output")
    two = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={"output": Topic(int)},
    )

    # An Topic channel accumulates updates into a sequence
    assert await app.ainvoke(2) == [3, 3]


async def test_invoke_checkpoint(mocker: MockerFixture) -> None:
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

    memory = MemorySaver()

    app = Pregel(
        nodes={"one": one},
        channels={"total": BinaryOperatorAggregate(int, operator.add)},
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
        channels={"inbox": Topic(int)},
    )

    # Then invoke app
    # We get a single array result as chain_four waits for all publishers to finish
    # before operating on all elements published to topic_two as an array
    for _ in range(100):
        assert await app.ainvoke(2) == [13, 13]

    assert await asyncio.gather(*(app.ainvoke(2) for _ in range(100))) == [
        [13, 13] for _ in range(100)
    ]


async def test_invoke_join_then_call_other_pubsub(mocker: MockerFixture) -> None:
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
        channels={
            "inbox_one": Topic(int),
            "outbox_one": LastValue(int),
        },
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

    app = Pregel(nodes={"one": one, "two": two})

    # Then invoke pubsub
    assert [c async for c in app.astream(2)] == [
        {"between": 3, "output": 3},
        {"output": 4},
    ]


async def test_invoke_two_processes_no_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("between")
    two = Channel.subscribe_to("between") | add_one

    app = Pregel(nodes={"one": one, "two": two})

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
    two = Channel.subscribe_to_each("inbox") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "inbox": Topic(int),
            "ctx": Context(an_int, an_int_async, typ=int),
        },
        output=["inbox", "output"],
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
            assert chunk == {"output": 4}
        else:
            assert False, "Expected only two chunks"
    assert setup_sync.call_count == 0
    assert cleanup_sync.call_count == 0
    assert setup_async.call_count == 1, "Expected setup to be called once"
    assert cleanup_async.call_count == 1, "Expected cleanup to be called once"


async def test_conditional_graph() -> None:
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

    patches = [c async for c in app.astream_log({"input": "what is weather in sf"})]
    patch_paths = {op["path"] for log in patches for op in log.ops}

    # Check that agent (one of the nodes) has its output streamed to the logs
    assert "/logs/agent/streamed_output/-" in patch_paths


async def test_conditional_graph_state() -> None:
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


async def test_prebuilt_chat() -> None:
    from langchain.chat_models.fake import FakeMessagesListChatModel
    from langchain_community.tools import tool
    from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list) -> Self:
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
        c
        async for c in app.astream(
            {"messages": [HumanMessage(content="what is weather in sf")]}
        )
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


async def test_message_graph() -> None:
    from langchain.chat_models.fake import FakeMessagesListChatModel
    from langchain_community.tools import tool
    from langchain_core.agents import AgentAction
    from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list) -> Self:
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

    assert [
        c async for c in app.astream([HumanMessage(content="what is weather in sf")])
    ] == [
        {
            "agent": AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"query"'}
                },
            )
        },
        {"action": FunctionMessage(content="result for query", name="search_api")},
        {
            "agent": AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"another"'}
                },
            )
        },
        {"action": FunctionMessage(content="result for another", name="search_api")},
        {"agent": AIMessage(content="answer")},
        {
            "__end__": [
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
                        "function_call": {
                            "name": "search_api",
                            "arguments": '"another"',
                        }
                    },
                ),
                FunctionMessage(content="result for another", name="search_api"),
                AIMessage(content="answer"),
            ]
        },
    ]
