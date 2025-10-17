import asyncio
import operator
import re
import sys
from typing import (
    Annotated,
    Literal,
    cast,
)

import pytest
from langchain_core.messages import AnyMessage, ToolCall
from langchain_core.runnables import RunnableConfig, RunnablePick
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from pytest_mock import MockerFixture
from typing_extensions import TypedDict

from langgraph._internal._constants import PULL, PUSH
from langgraph.channels.last_value import LastValue
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.constants import END, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.types import PregelTask, Send, StateSnapshot, StreamWriter
from tests.any_int import AnyInt
from tests.any_str import AnyDict, AnyStr, UnsortedSequence
from tests.fake_chat import FakeChatModel
from tests.fake_tracer import FakeTracer
from tests.messages import (
    _AnyIdAIMessage,
    _AnyIdAIMessageChunk,
    _AnyIdHumanMessage,
    _AnyIdToolMessage,
)

pytestmark = pytest.mark.anyio


async def test_invoke_two_processes_in_out_interrupt(
    async_checkpointer: BaseCheckpointSaver, mocker: MockerFixture
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = NodeBuilder().subscribe_only("input").do(add_one).write_to("inbox")
    two = NodeBuilder().subscribe_only("inbox").do(add_one).write_to("output")
    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "inbox": LastValue(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
        checkpointer=async_checkpointer,
        interrupt_after_nodes=["one"],
    )
    thread1 = {"configurable": {"thread_id": "1"}}
    thread2 = {"configurable": {"thread_id": "2"}}

    # start execution, stop at inbox
    assert await app.ainvoke(2, thread1, durability="async") is None

    # inbox == 3
    checkpoint = await async_checkpointer.aget(thread1)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 3

    # resume execution, finish
    assert await app.ainvoke(None, thread1, durability="async") == 4

    # start execution again, stop at inbox
    assert await app.ainvoke(20, thread1, durability="async") is None

    # inbox == 21
    checkpoint = await async_checkpointer.aget(thread1)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 21

    # send a new value in, interrupting the previous execution
    assert await app.ainvoke(3, thread1, durability="async") is None
    assert await app.ainvoke(None, thread1, durability="async") == 5

    # start execution again, stopping at inbox
    assert await app.ainvoke(20, thread2, durability="async") is None

    # inbox == 21
    snapshot = await app.aget_state(thread2)
    assert snapshot.values["inbox"] == 21
    assert snapshot.next == ("two",)

    # update the state, resume
    await app.aupdate_state(thread2, 25, as_node="one")
    assert await app.ainvoke(None, thread2) == 26

    # no pending tasks
    snapshot = await app.aget_state(thread2)
    assert snapshot.next == ()

    # list history
    history = [c async for c in app.aget_state_history(thread1)]
    assert len(history) == 8
    assert history == [
        StateSnapshot(
            values={"inbox": 4, "output": 5, "input": 3},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 6,
            },
            created_at=AnyStr(),
            parent_config=history[1].config,
            interrupts=(),
        ),
        StateSnapshot(
            values={"inbox": 4, "output": 4, "input": 3},
            tasks=(PregelTask(AnyStr(), "two", (PULL, "two"), result={"output": 5}),),
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 5,
            },
            created_at=AnyStr(),
            parent_config=history[2].config,
            interrupts=(),
        ),
        StateSnapshot(
            values={"inbox": 21, "output": 4, "input": 3},
            tasks=(PregelTask(AnyStr(), "one", (PULL, "one"), result={"inbox": 4}),),
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "step": 4,
            },
            created_at=AnyStr(),
            parent_config=history[3].config,
            interrupts=(),
        ),
        StateSnapshot(
            values={"inbox": 21, "output": 4, "input": 20},
            tasks=(PregelTask(AnyStr(), "two", (PULL, "two")),),
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
            },
            created_at=AnyStr(),
            parent_config=history[4].config,
            interrupts=(),
        ),
        StateSnapshot(
            values={"inbox": 3, "output": 4, "input": 20},
            tasks=(PregelTask(AnyStr(), "one", (PULL, "one"), result={"inbox": 21}),),
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "step": 2,
            },
            created_at=AnyStr(),
            parent_config=history[5].config,
            interrupts=(),
        ),
        StateSnapshot(
            values={"inbox": 3, "output": 4, "input": 2},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config=history[6].config,
            interrupts=(),
        ),
        StateSnapshot(
            values={"inbox": 3, "input": 2},
            tasks=(PregelTask(AnyStr(), "two", (PULL, "two"), result={"output": 4}),),
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
            },
            created_at=AnyStr(),
            parent_config=history[7].config,
            interrupts=(),
        ),
        StateSnapshot(
            values={"input": 2},
            tasks=(PregelTask(AnyStr(), "one", (PULL, "one"), result={"inbox": 3}),),
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
            interrupts=(),
        ),
    ]

    # forking from any previous checkpoint should re-run nodes
    assert [
        c async for c in app.astream(None, history[0].config, stream_mode="updates")
    ] == []
    assert [
        c async for c in app.astream(None, history[1].config, stream_mode="updates")
    ] == [
        {"two": {"output": 5}},
    ]
    assert [
        c async for c in app.astream(None, history[2].config, stream_mode="updates")
    ] == [
        {"one": {"inbox": 4}},
        {"__interrupt__": ()},
    ]


async def test_fork_always_re_runs_nodes(
    async_checkpointer: BaseCheckpointSaver, mocker: MockerFixture
) -> None:
    add_one = mocker.Mock(side_effect=lambda _: 1)

    builder = StateGraph(Annotated[int, operator.add])
    builder.add_node("add_one", add_one)
    builder.add_edge(START, "add_one")
    builder.add_conditional_edges("add_one", lambda cnt: "add_one" if cnt < 6 else END)
    graph = builder.compile(checkpointer=async_checkpointer)

    thread1 = {"configurable": {"thread_id": "1"}}

    # start execution, stop at inbox
    assert [
        c
        async for c in graph.astream(
            1, thread1, stream_mode=["values", "updates"], durability="async"
        )
    ] == [
        ("values", 1),
        ("updates", {"add_one": 1}),
        ("values", 2),
        ("updates", {"add_one": 1}),
        ("values", 3),
        ("updates", {"add_one": 1}),
        ("values", 4),
        ("updates", {"add_one": 1}),
        ("values", 5),
        ("updates", {"add_one": 1}),
        ("values", 6),
    ]

    # list history
    history = [c async for c in graph.aget_state_history(thread1)]
    assert history == [
        StateSnapshot(
            values=6,
            next=(),
            tasks=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 5,
            },
            created_at=AnyStr(),
            parent_config=history[1].config,
            interrupts=(),
        ),
        StateSnapshot(
            values=5,
            tasks=(PregelTask(AnyStr(), "add_one", (PULL, "add_one"), result=1),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 4,
            },
            created_at=AnyStr(),
            parent_config=history[2].config,
            interrupts=(),
        ),
        StateSnapshot(
            values=4,
            tasks=(PregelTask(AnyStr(), "add_one", (PULL, "add_one"), result=1),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
            },
            created_at=AnyStr(),
            parent_config=history[3].config,
            interrupts=(),
        ),
        StateSnapshot(
            values=3,
            tasks=(PregelTask(AnyStr(), "add_one", (PULL, "add_one"), result=1),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 2,
            },
            created_at=AnyStr(),
            parent_config=history[4].config,
            interrupts=(),
        ),
        StateSnapshot(
            values=2,
            tasks=(PregelTask(AnyStr(), "add_one", (PULL, "add_one"), result=1),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config=history[5].config,
            interrupts=(),
        ),
        StateSnapshot(
            values=1,
            tasks=(PregelTask(AnyStr(), "add_one", (PULL, "add_one"), result=1),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
            },
            created_at=AnyStr(),
            parent_config=history[6].config,
            interrupts=(),
        ),
        StateSnapshot(
            values=0,
            tasks=(PregelTask(AnyStr(), "__start__", (PULL, "__start__"), result=1),),
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
            interrupts=(),
        ),
    ]

    # forking from any previous checkpoint should re-run nodes
    assert [
        c async for c in graph.astream(None, history[0].config, stream_mode="updates")
    ] == []
    assert [
        c async for c in graph.astream(None, history[1].config, stream_mode="updates")
    ] == [
        {"add_one": 1},
    ]
    assert [
        c async for c in graph.astream(None, history[2].config, stream_mode="updates")
    ] == [
        {"add_one": 1},
        {"add_one": 1},
    ]


async def test_conditional_graph_state(async_checkpointer: BaseCheckpointSaver) -> None:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import tool

    class AgentState(TypedDict):
        input: Annotated[str, UntrackedValue]
        agent_outcome: AgentAction | AgentFinish | None
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

    def agent_parser(input: str) -> dict[str, AgentAction | AgentFinish]:
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
        # execute the tool
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        return {"intermediate_steps": [[agent_action, observation]]}

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
            [
                AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:query",
                ),
                "result for query",
            ],
            [
                AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
                "result for another",
            ],
        ],
        "agent_outcome": AgentFinish(
            return_values={"answer": "answer"}, log="finish:answer"
        ),
    }

    assert [c async for c in app.astream({"input": "what is weather in sf"})] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:query",
                ),
            }
        },
        {
            "tools": {
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
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
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ],
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
    # Check that agent (one of the nodes) has its final output set in the logs
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
        checkpointer=async_checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config, durability="exit"
        )
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:query",
                ),
            }
        },
        {"__interrupt__": ()},
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:query",
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
        },
        parent_config=None,
        interrupts=(),
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
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
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
        {"__interrupt__": ()},
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
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                ]
            ],
        },
        tasks=(),
        next=(),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=async_checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config, durability="exit"
        )
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:query",
                ),
            }
        },
        {"__interrupt__": ()},
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
        },
        parent_config=None,
        interrupts=(),
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
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
        interrupts=(),
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
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
        {"__interrupt__": ()},
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
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                ]
            ],
        },
        tasks=(),
        next=(),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
        interrupts=(),
    )


async def test_prebuilt_tool_chat() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    model = FakeChatModel(
        messages=[
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
    )

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    app = create_react_agent(model, tools)

    assert await app.ainvoke(
        {"messages": [HumanMessage(content="what is weather in sf")]}
    ) == {
        "messages": [
            _AnyIdHumanMessage(content="what is weather in sf"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            _AnyIdToolMessage(
                content="result for query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            _AnyIdAIMessage(
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
            _AnyIdToolMessage(
                content="result for another",
                name="search_api",
                tool_call_id="tool_call234",
            ),
            _AnyIdToolMessage(
                content="result for a third one",
                name="search_api",
                tool_call_id="tool_call567",
                id=AnyStr(),
            ),
            _AnyIdAIMessage(content="answer"),
        ]
    }

    events = [
        c
        async for c in app.astream(
            {"messages": [HumanMessage(content="what is weather in sf")]},
            stream_mode="messages",
        )
    ]

    assert events[:3] == [
        (
            _AnyIdAIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "search_api",
                        "args": {"query": "query"},
                        "id": "tool_call123",
                        "type": "tool_call",
                    }
                ],
                tool_call_chunks=[
                    {
                        "name": "search_api",
                        "args": '{"query": "query"}',
                        "id": "tool_call123",
                        "index": None,
                        "type": "tool_call_chunk",
                    }
                ],
                chunk_position="last",
            ),
            {
                "langgraph_step": 1,
                "langgraph_node": "agent",
                "langgraph_triggers": ("branch:to:agent",),
                "langgraph_path": (PULL, "agent"),
                "langgraph_checkpoint_ns": AnyStr("agent:"),
                "checkpoint_ns": AnyStr("agent:"),
                "ls_provider": "fakechatmodel",
                "ls_model_type": "chat",
            },
        ),
        (
            _AnyIdToolMessage(
                content="result for query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            {
                "langgraph_step": 2,
                "langgraph_node": "tools",
                "langgraph_triggers": (PUSH,),
                "langgraph_path": (PUSH, AnyInt(), False),
                "langgraph_checkpoint_ns": AnyStr("tools:"),
            },
        ),
        (
            _AnyIdAIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "search_api",
                        "args": {"query": "another"},
                        "id": "tool_call234",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_api",
                        "args": {"query": "a third one"},
                        "id": "tool_call567",
                        "type": "tool_call",
                    },
                ],
                tool_call_chunks=[
                    {
                        "name": "search_api",
                        "args": '{"query": "another"}',
                        "id": "tool_call234",
                        "index": None,
                        "type": "tool_call_chunk",
                    },
                    {
                        "name": "search_api",
                        "args": '{"query": "a third one"}',
                        "id": "tool_call567",
                        "index": None,
                        "type": "tool_call_chunk",
                    },
                ],
                chunk_position="last",
            ),
            {
                "langgraph_step": 3,
                "langgraph_node": "agent",
                "langgraph_triggers": ("branch:to:agent",),
                "langgraph_path": (PULL, "agent"),
                "langgraph_checkpoint_ns": AnyStr("agent:"),
                "checkpoint_ns": AnyStr("agent:"),
                "ls_provider": "fakechatmodel",
                "ls_model_type": "chat",
            },
        ),
    ]

    assert events[3:5] == UnsortedSequence(
        (
            _AnyIdToolMessage(
                content="result for another",
                name="search_api",
                tool_call_id="tool_call234",
            ),
            {
                "langgraph_step": 4,
                "langgraph_node": "tools",
                "langgraph_triggers": (PUSH,),
                "langgraph_path": (PUSH, AnyInt(), False),
                "langgraph_checkpoint_ns": AnyStr("tools:"),
            },
        ),
        (
            _AnyIdToolMessage(
                content="result for a third one",
                name="search_api",
                tool_call_id="tool_call567",
            ),
            {
                "langgraph_step": 4,
                "langgraph_node": "tools",
                "langgraph_triggers": (PUSH,),
                "langgraph_path": (PUSH, AnyInt(), False),
                "langgraph_checkpoint_ns": AnyStr("tools:"),
            },
        ),
    )
    assert events[5:] == [
        (
            _AnyIdAIMessageChunk(
                content="answer",
                chunk_position="last",
            ),
            {
                "langgraph_step": 5,
                "langgraph_node": "agent",
                "langgraph_triggers": ("branch:to:agent",),
                "langgraph_path": (PULL, "agent"),
                "langgraph_checkpoint_ns": AnyStr("agent:"),
                "checkpoint_ns": AnyStr("agent:"),
                "ls_provider": "fakechatmodel",
                "ls_model_type": "chat",
            },
        ),
    ]

    stream_updates_events = [
        c
        async for c in app.astream(
            {"messages": [HumanMessage(content="what is weather in sf")]}
        )
    ]
    assert stream_updates_events[:3] == [
        {
            "agent": {
                "messages": [
                    _AnyIdAIMessage(
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
            "tools": {
                "messages": [
                    _AnyIdToolMessage(
                        content="result for query",
                        name="search_api",
                        tool_call_id="tool_call123",
                    )
                ]
            }
        },
        {
            "agent": {
                "messages": [
                    _AnyIdAIMessage(
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
    ]
    assert stream_updates_events[3:5] == UnsortedSequence(
        {
            "tools": {
                "messages": [
                    _AnyIdToolMessage(
                        content="result for another",
                        name="search_api",
                        tool_call_id="tool_call234",
                    ),
                ]
            }
        },
        {
            "tools": {
                "messages": [
                    _AnyIdToolMessage(
                        content="result for a third one",
                        name="search_api",
                        tool_call_id="tool_call567",
                    ),
                ]
            }
        },
    )
    assert stream_updates_events[5:] == [
        {"agent": {"messages": [_AnyIdAIMessage(content="answer")]}}
    ]


async def test_state_graph_packets(async_checkpointer: BaseCheckpointSaver) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolMessage,
    )
    from langchain_core.tools import tool

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]
    tools_by_name = {t.name: t for t in tools}

    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                id="ai1",
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
                id="ai2",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call234",
                        "name": "search_api",
                        "args": {"query": "another", "idx": 0},
                    },
                    {
                        "id": "tool_call567",
                        "name": "search_api",
                        "args": {"query": "a third one", "idx": 1},
                    },
                ],
            ),
            AIMessage(id="ai3", content="answer"),
        ]
    )

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        # Logic to decide whether to continue in the loop or exit
        if tool_calls := data["messages"][-1].tool_calls:
            return [Send("tools", tool_call) for tool_call in tool_calls]
        else:
            return END

    async def tools_node(input: ToolCall, config: RunnableConfig) -> AgentState:
        await asyncio.sleep(input["args"].get("idx", 0) / 10)
        output = await tools_by_name[input["name"]].ainvoke(input["args"], config)
        return {
            "messages": ToolMessage(
                content=output, name=input["name"], tool_call_id=input["id"]
            )
        }

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", {"messages": RunnablePick("messages") | model})
    workflow.add_node("tools", tools_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges("agent", should_continue)

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    assert await app.ainvoke(
        {"messages": HumanMessage(content="what is weather in sf")}
    ) == {
        "messages": [
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            _AnyIdToolMessage(
                content="result for query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            AIMessage(
                id="ai2",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call234",
                        "name": "search_api",
                        "args": {"query": "another", "idx": 0},
                    },
                    {
                        "id": "tool_call567",
                        "name": "search_api",
                        "args": {"query": "a third one", "idx": 1},
                    },
                ],
            ),
            _AnyIdToolMessage(
                content="result for another",
                name="search_api",
                tool_call_id="tool_call234",
            ),
            _AnyIdToolMessage(
                content="result for a third one",
                name="search_api",
                tool_call_id="tool_call567",
            ),
            AIMessage(content="answer", id="ai3"),
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
                "messages": AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                )
            },
        },
        {
            "tools": {
                "messages": _AnyIdToolMessage(
                    content="result for query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            }
        },
        {
            "agent": {
                "messages": AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                )
            }
        },
        {
            "tools": {
                "messages": _AnyIdToolMessage(
                    content="result for another",
                    name="search_api",
                    tool_call_id="tool_call234",
                )
            },
        },
        {
            "tools": {
                "messages": _AnyIdToolMessage(
                    content="result for a third one",
                    name="search_api",
                    tool_call_id="tool_call567",
                ),
            },
        },
        {"agent": {"messages": AIMessage(content="answer", id="ai3")}},
    ]

    # interrupt after agent

    app_w_interrupt = workflow.compile(
        checkpointer=async_checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"messages": HumanMessage(content="what is weather in sf")},
            config,
            durability="exit",
        )
    ] == [
        {
            "agent": {
                "messages": AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                )
            }
        },
        {"__interrupt__": ()},
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                ),
            ]
        },
        tasks=(PregelTask(AnyStr(), "tools", (PUSH, 0, False)),),
        next=("tools",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
        },
        parent_config=None,
        interrupts=(),
    )

    # modify ai message
    last_message = (await app_w_interrupt.aget_state(config)).values["messages"][-1]
    last_message.tool_calls[0]["args"]["query"] = "a different query"
    await app_w_interrupt.aupdate_state(config, {"messages": last_message})

    # message was replaced instead of appended
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
            ]
        },
        tasks=(PregelTask(AnyStr(), "tools", (PUSH, 0, False)),),
        next=("tools",),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "messages": _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            }
        },
        {
            "agent": {
                "messages": AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                )
            },
        },
        {"__interrupt__": ()},
    ]

    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                ),
                AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                ),
            ]
        },
        tasks=(
            PregelTask(AnyStr(), "tools", (PUSH, 0, False)),
            PregelTask(AnyStr(), "tools", (PUSH, 1, False)),
        ),
        next=("tools", "tools"),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 4,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )

    await app_w_interrupt.aupdate_state(
        config,
        {"messages": AIMessage(content="answer", id="ai2")},
    )

    # replaces message even if object identity is different, as long as id is the same
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                ),
                AIMessage(content="answer", id="ai2"),
            ]
        },
        tasks=(),
        next=(),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )

    # interrupt before tools

    app_w_interrupt = workflow.compile(
        checkpointer=async_checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    model.i = 0

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"messages": HumanMessage(content="what is weather in sf")},
            config,
            durability="exit",
        )
    ] == [
        {
            "agent": {
                "messages": AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                )
            }
        },
        {"__interrupt__": ()},
    ]
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                ),
            ]
        },
        tasks=(PregelTask(AnyStr(), "tools", (PUSH, 0, False)),),
        next=("tools",),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
        },
        parent_config=None,
        interrupts=(),
    )

    # modify ai message
    last_message = (await app_w_interrupt.aget_state(config)).values["messages"][-1]
    last_message.tool_calls[0]["args"]["query"] = "a different query"
    await app_w_interrupt.aupdate_state(config, {"messages": last_message})

    # message was replaced instead of appended
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
            ]
        },
        tasks=(PregelTask(AnyStr(), "tools", (PUSH, 0, False)),),
        next=("tools",),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "messages": _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            }
        },
        {
            "agent": {
                "messages": AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                )
            },
        },
        {"__interrupt__": ()},
    ]

    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                ),
                AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                ),
            ]
        },
        tasks=(
            PregelTask(AnyStr(), "tools", (PUSH, 0, False)),
            PregelTask(AnyStr(), "tools", (PUSH, 1, False)),
        ),
        next=("tools", "tools"),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 4,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )

    await app_w_interrupt.aupdate_state(
        config,
        {"messages": AIMessage(content="answer", id="ai2")},
    )

    # replaces message even if object identity is different, as long as id is the same
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                ),
                AIMessage(content="answer", id="ai2"),
            ]
        },
        tasks=(),
        next=(),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )


async def test_message_graph(async_checkpointer: BaseCheckpointSaver) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    class FakeFunctionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
            return self

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    model = FakeFunctionChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
            AIMessage(content="answer", id="ai3"),
        ]
    )

    # Define the function that determines whether to continue or not
    def should_continue(messages):
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    # Define a new graph
    workflow = StateGraph(state_schema=Annotated[list[AnyMessage], add_messages])  # type: ignore[arg-type]

    # Define the two nodes we will cycle between
    workflow.add_node("agent", model)
    workflow.add_node("tools", ToolNode(tools))

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
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    assert await app.ainvoke([HumanMessage(content="what is weather in sf")]) == [
        _AnyIdHumanMessage(
            content="what is weather in sf",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call123",
                    "name": "search_api",
                    "args": {"query": "query"},
                }
            ],
            id="ai1",  # respects ids passed in
        ),
        _AnyIdToolMessage(
            content="result for query",
            name="search_api",
            tool_call_id="tool_call123",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call456",
                    "name": "search_api",
                    "args": {"query": "another"},
                }
            ],
            id="ai2",
        ),
        _AnyIdToolMessage(
            content="result for another",
            name="search_api",
            tool_call_id="tool_call456",
        ),
        AIMessage(content="answer", id="ai3"),
    ]

    assert [
        c async for c in app.astream([HumanMessage(content="what is weather in sf")])
    ] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
        {
            "tools": [
                _AnyIdToolMessage(
                    content="result for query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
        {
            "tools": [
                _AnyIdToolMessage(
                    content="result for another",
                    name="search_api",
                    tool_call_id="tool_call456",
                )
            ]
        },
        {"agent": AIMessage(content="answer", id="ai3")},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=async_checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            HumanMessage(content="what is weather in sf"),
            config,
            durability="exit",
        )
    ] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
        {"__interrupt__": ()},
    ]

    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
        },
        parent_config=None,
        interrupts=(),
    )

    # modify ai message
    last_message = (await app_w_interrupt.aget_state(config)).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    await app_w_interrupt.aupdate_state(config, last_message)

    # message was replaced instead of appended
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": [
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
        {"__interrupt__": ()},
    ]

    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 4,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )

    await app_w_interrupt.aupdate_state(
        config,
        AIMessage(content="answer", id="ai2"),
    )

    # replaces message even if object identity is different, as long as id is the same
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        tasks=(),
        next=(),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
        },
        parent_config=(
            [c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )


async def test_in_one_fan_out_out_one_graph_state() -> None:
    def sorted_add(x: list[str], y: list[str]) -> list[str]:
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], operator.add]

    async def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    async def retriever_one(data: State) -> State:
        await asyncio.sleep(0.1)
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
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
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

    assert [
        c
        async for c in app.astream(
            {"query": "what is weather in sf"},
            stream_mode=["values", "updates", "debug"],
        )
    ] == [
        ("values", {"query": "what is weather in sf", "docs": []}),
        (
            "debug",
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "id": AnyStr(),
                    "name": "rewrite_query",
                    "input": {"query": "what is weather in sf", "docs": []},
                    "triggers": ("branch:to:rewrite_query",),
                },
            },
        ),
        ("updates", {"rewrite_query": {"query": "query: what is weather in sf"}}),
        (
            "debug",
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "id": AnyStr(),
                    "name": "rewrite_query",
                    "result": {
                        "query": "query: what is weather in sf",
                    },
                    "error": None,
                    "interrupts": [],
                },
            },
        ),
        ("values", {"query": "query: what is weather in sf", "docs": []}),
        (
            "debug",
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": AnyStr(),
                    "name": "retriever_one",
                    "input": {"query": "query: what is weather in sf", "docs": []},
                    "triggers": ("branch:to:retriever_one",),
                },
            },
        ),
        (
            "debug",
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": AnyStr(),
                    "name": "retriever_two",
                    "input": {"query": "query: what is weather in sf", "docs": []},
                    "triggers": ("branch:to:retriever_two",),
                },
            },
        ),
        (
            "updates",
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
        ),
        (
            "debug",
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": AnyStr(),
                    "name": "retriever_two",
                    "result": {
                        "docs": ["doc3", "doc4"],
                    },
                    "error": None,
                    "interrupts": [],
                },
            },
        ),
        (
            "updates",
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
        ),
        (
            "debug",
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": AnyStr(),
                    "name": "retriever_one",
                    "result": {
                        "docs": ["doc1", "doc2"],
                    },
                    "error": None,
                    "interrupts": [],
                },
            },
        ),
        (
            "values",
            {
                "query": "query: what is weather in sf",
                "docs": ["doc1", "doc2", "doc3", "doc4"],
            },
        ),
        (
            "debug",
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 3,
                "payload": {
                    "id": AnyStr(),
                    "name": "qa",
                    "input": {
                        "query": "query: what is weather in sf",
                        "docs": ["doc1", "doc2", "doc3", "doc4"],
                    },
                    "triggers": ("branch:to:qa",),
                },
            },
        ),
        ("updates", {"qa": {"answer": "doc1,doc2,doc3,doc4"}}),
        (
            "debug",
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 3,
                "payload": {
                    "id": AnyStr(),
                    "name": "qa",
                    "result": {
                        "answer": "doc1,doc2,doc3,doc4",
                    },
                    "error": None,
                    "interrupts": [],
                },
            },
        ),
        (
            "values",
            {
                "query": "query: what is weather in sf",
                "answer": "doc1,doc2,doc3,doc4",
                "docs": ["doc1", "doc2", "doc3", "doc4"],
            },
        ),
    ]


async def test_nested_graph_state(async_checkpointer: BaseCheckpointSaver) -> None:
    class InnerState(TypedDict):
        my_key: str
        my_other_key: str

    def inner_1(state: InnerState):
        return {
            "my_key": state["my_key"] + " here",
            "my_other_key": state["my_key"],
        }

    def inner_2(state: InnerState):
        return {
            "my_key": state["my_key"] + " and there",
            "my_other_key": state["my_key"],
        }

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: str
        other_parent_key: str

    def outer_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def outer_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("outer_1", outer_1)
    graph.add_node(
        "inner",
        inner.compile(interrupt_before=["inner_2"]),
    )
    graph.add_node("outer_2", outer_2)
    graph.set_entry_point("outer_1")
    graph.add_edge("outer_1", "inner")
    graph.add_edge("inner", "outer_2")
    graph.set_finish_point("outer_2")

    app = graph.compile(checkpointer=async_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({"my_key": "my value"}, config, durability="exit")
    # test state w/ nested subgraph state (right after interrupt)
    # first get_state without subgraph state
    expected = StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "inner",
                (PULL, "inner"),
                state={"configurable": {"thread_id": "1", "checkpoint_ns": AnyStr()}},
            ),
        ),
        next=("inner",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
        },
        created_at=AnyStr(),
        parent_config=None,
        interrupts=(),
    )
    assert await app.aget_state(config) == expected
    # now, get_state with subgraphs state
    assert await app.aget_state(config, subgraphs=True) == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "inner",
                (PULL, "inner"),
                state=StateSnapshot(
                    values={
                        "my_key": "hi my value here",
                        "my_other_key": "hi my value",
                    },
                    tasks=(
                        PregelTask(
                            AnyStr(),
                            "inner_2",
                            (PULL, "inner_2"),
                        ),
                    ),
                    next=("inner_2",),
                    config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("inner:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {"": AnyStr(), AnyStr("child:"): AnyStr()}
                            ),
                        }
                    },
                    metadata={
                        "parents": {
                            "": AnyStr(),
                        },
                        "source": "loop",
                        "step": 1,
                    },
                    created_at=AnyStr(),
                    parent_config=None,
                    interrupts=(),
                ),
            ),
        ),
        next=("inner",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
        },
        created_at=AnyStr(),
        parent_config=None,
        interrupts=(),
    )
    # get_state_history returns outer graph checkpoints
    assert [c async for c in app.aget_state_history(config)] == [expected]

    # get_state_history for a subgraph returns its checkpoints
    child_history = [
        c
        async for c in app.aget_state_history(
            (await app.aget_state(config)).tasks[0].state
        )
    ]
    expected_child_history = [
        StateSnapshot(
            values={"my_key": "hi my value here", "my_other_key": "hi my value"},
            next=("inner_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "loop",
                "step": 1,
                "parents": {"": AnyStr()},
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(PregelTask(AnyStr(), "inner_2", (PULL, "inner_2")),),
            interrupts=(),
        ),
    ]

    assert child_history == expected_child_history

    # resume
    await app.ainvoke(None, config, durability="exit")
    # test state w/ nested subgraph state (after resuming from interrupt)
    assert await app.aget_state(config) == StateSnapshot(
        values={"my_key": "hi my value here and there and back again"},
        tasks=(),
        next=(),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
        },
        created_at=AnyStr(),
        parent_config=(
            {
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            }
        ),
        interrupts=(),
    )
    # test full history at the end
    actual_history = [c async for c in app.aget_state_history(config)]
    expected_history = [
        StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
            },
            created_at=AnyStr(),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                }
            ),
            interrupts=(),
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "inner",
                    (PULL, "inner"),
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr(),
                        }
                    },
                    result=None,
                ),
            ),
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config=None,
            interrupts=(),
        ),
    ]

    assert actual_history == expected_history
    # test looking up parent state by checkpoint ID
    for actual_snapshot, expected_snapshot in zip(actual_history, expected_history):
        assert await app.aget_state(actual_snapshot.config) == expected_snapshot


async def test_doubly_nested_graph_state(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    class State(TypedDict):
        my_key: str

    class ChildState(TypedDict):
        my_key: str

    class GrandChildState(TypedDict):
        my_key: str

    def grandchild_1(state: ChildState):
        return {"my_key": state["my_key"] + " here"}

    def grandchild_2(state: ChildState):
        return {
            "my_key": state["my_key"] + " and there",
        }

    grandchild = StateGraph(GrandChildState)
    grandchild.add_node("grandchild_1", grandchild_1)
    grandchild.add_node("grandchild_2", grandchild_2)
    grandchild.add_edge("grandchild_1", "grandchild_2")
    grandchild.set_entry_point("grandchild_1")
    grandchild.set_finish_point("grandchild_2")

    child = StateGraph(ChildState)
    child.add_node(
        "child_1",
        grandchild.compile(interrupt_before=["grandchild_2"]),
    )
    child.set_entry_point("child_1")
    child.set_finish_point("child_1")

    def parent_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def parent_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("parent_1", parent_1)
    graph.add_node("child", child.compile())
    graph.add_node("parent_2", parent_2)
    graph.set_entry_point("parent_1")
    graph.add_edge("parent_1", "child")
    graph.add_edge("child", "parent_2")
    graph.set_finish_point("parent_2")

    app = graph.compile(checkpointer=async_checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert [
        c
        async for c in app.astream(
            {"my_key": "my value"}, config, subgraphs=True, durability="exit"
        )
    ] == [
        ((), {"parent_1": {"my_key": "hi my value"}}),
        (
            (AnyStr("child:"), AnyStr("child_1:")),
            {"grandchild_1": {"my_key": "hi my value here"}},
        ),
        ((), {"__interrupt__": ()}),
    ]
    # get state without subgraphs
    outer_state = await app.aget_state(config)
    assert outer_state == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "child",
                (PULL, "child"),
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("child"),
                    }
                },
            ),
        ),
        next=("child",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
        },
        created_at=AnyStr(),
        parent_config=None,
        interrupts=(),
    )
    child_state = await app.aget_state(outer_state.tasks[0].state)
    assert child_state == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "child_1",
                (PULL, "child_1"),
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr(),
                    }
                },
            ),
        ),
        next=("child_1",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": AnyStr("child:"),
                "checkpoint_id": AnyStr(),
                "checkpoint_map": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                    }
                ),
            }
        },
        metadata={
            "parents": {"": AnyStr()},
            "source": "loop",
            "step": 0,
        },
        created_at=AnyStr(),
        parent_config=None,
        interrupts=(),
    )
    grandchild_state = await app.aget_state(child_state.tasks[0].state)
    assert grandchild_state == StateSnapshot(
        values={"my_key": "hi my value here"},
        tasks=(
            PregelTask(
                AnyStr(),
                "grandchild_2",
                (PULL, "grandchild_2"),
            ),
        ),
        next=("grandchild_2",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": AnyStr(),
                "checkpoint_id": AnyStr(),
                "checkpoint_map": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                        AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                    }
                ),
            }
        },
        metadata={
            "parents": AnyDict(
                {
                    "": AnyStr(),
                    AnyStr("child:"): AnyStr(),
                }
            ),
            "source": "loop",
            "step": 1,
        },
        created_at=AnyStr(),
        parent_config=None,
        interrupts=(),
    )
    # get state with subgraphs
    assert await app.aget_state(config, subgraphs=True) == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "child",
                (PULL, "child"),
                state=StateSnapshot(
                    values={"my_key": "hi my value"},
                    tasks=(
                        PregelTask(
                            AnyStr(),
                            "child_1",
                            (PULL, "child_1"),
                            state=StateSnapshot(
                                values={"my_key": "hi my value here"},
                                tasks=(
                                    PregelTask(
                                        AnyStr(),
                                        "grandchild_2",
                                        (PULL, "grandchild_2"),
                                    ),
                                ),
                                next=("grandchild_2",),
                                config={
                                    "configurable": {
                                        "thread_id": "1",
                                        "checkpoint_ns": AnyStr(),
                                        "checkpoint_id": AnyStr(),
                                        "checkpoint_map": AnyDict(
                                            {
                                                "": AnyStr(),
                                                AnyStr("child:"): AnyStr(),
                                                AnyStr(
                                                    re.compile(r"child:.+|child1:")
                                                ): AnyStr(),
                                            }
                                        ),
                                    }
                                },
                                metadata={
                                    "parents": AnyDict(
                                        {
                                            "": AnyStr(),
                                            AnyStr("child:"): AnyStr(),
                                        }
                                    ),
                                    "source": "loop",
                                    "step": 1,
                                },
                                created_at=AnyStr(),
                                parent_config=None,
                                interrupts=(),
                            ),
                        ),
                    ),
                    next=("child_1",),
                    config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("child:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {"": AnyStr(), AnyStr("child:"): AnyStr()}
                            ),
                        }
                    },
                    metadata={
                        "parents": {"": AnyStr()},
                        "source": "loop",
                        "step": 0,
                    },
                    created_at=AnyStr(),
                    parent_config=None,
                    interrupts=(),
                ),
            ),
        ),
        next=("child",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
        },
        created_at=AnyStr(),
        parent_config=None,
        interrupts=(),
    )
    # resume
    assert [
        c async for c in app.astream(None, config, subgraphs=True, durability="exit")
    ] == [
        (
            (AnyStr("child:"), AnyStr("child_1:")),
            {"grandchild_2": {"my_key": "hi my value here and there"}},
        ),
        (
            (AnyStr("child:"),),
            {"child_1": {"my_key": "hi my value here and there"}},
        ),
        ((), {"child": {"my_key": "hi my value here and there"}}),
        ((), {"parent_2": {"my_key": "hi my value here and there and back again"}}),
    ]
    # get state with and without subgraphs
    assert (
        await app.aget_state(config)
        == await app.aget_state(config, subgraphs=True)
        == StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
            },
            created_at=AnyStr(),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                }
            ),
            interrupts=(),
        )
    )

    # get outer graph history
    outer_history = [c async for c in app.aget_state_history(config)]
    assert outer_history == [
        StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            interrupts=(),
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "child",
                    (PULL, "child"),
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("child"),
                        }
                    },
                ),
            ),
            next=("child",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config=None,
            interrupts=(),
        ),
    ]
    # get child graph history
    child_history = [
        c async for c in app.aget_state_history(outer_history[1].tasks[0].state)
    ]
    assert child_history == [
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("child_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "loop",
                "step": 0,
                "parents": {"": AnyStr()},
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="child_1",
                    path=(PULL, "child_1"),
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("child:"),
                        }
                    },
                    result=None,
                ),
            ),
            interrupts=(),
        ),
    ]
    # get grandchild graph history
    grandchild_history = [
        c async for c in app.aget_state_history(child_history[0].tasks[0].state)
    ]
    assert grandchild_history == [
        StateSnapshot(
            values={"my_key": "hi my value here"},
            next=("grandchild_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            metadata={
                "source": "loop",
                "step": 1,
                "parents": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                    }
                ),
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="grandchild_2",
                    path=(PULL, "grandchild_2"),
                    result=None,
                ),
            ),
            interrupts=(),
        ),
    ]


async def test_send_to_nested_graphs(async_checkpointer: BaseCheckpointSaver) -> None:
    class OverallState(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]

    async def continue_to_jokes(state: OverallState):
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

    class JokeState(TypedDict):
        subject: str

    async def edit(state: JokeState):
        subject = state["subject"]
        return {"subject": f"{subject} - hohoho"}

    # subgraph
    subgraph = StateGraph(JokeState, output_schema=OverallState)
    subgraph.add_node("edit", edit)
    subgraph.add_node(
        "generate", lambda state: {"jokes": [f"Joke about {state['subject']}"]}
    )
    subgraph.set_entry_point("edit")
    subgraph.add_edge("edit", "generate")
    subgraph.set_finish_point("generate")

    # parent graph
    builder = StateGraph(OverallState)
    builder.add_node(
        "generate_joke",
        subgraph.compile(interrupt_before=["generate"]),
    )
    builder.add_conditional_edges(START, continue_to_jokes)
    builder.add_edge("generate_joke", END)

    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    tracer = FakeTracer()

    # invoke and pause at nested interrupt
    assert await graph.ainvoke(
        {"subjects": ["cats", "dogs"]},
        config={**config, "callbacks": [tracer]},
    ) == {
        "subjects": ["cats", "dogs"],
        "jokes": [],
    }
    assert len(tracer.runs) == 1, "Should produce exactly 1 root run"

    # check state
    outer_state = await graph.aget_state(config)

    # update state of dogs joke graph
    await graph.aupdate_state(
        outer_state.tasks[1].state, {"subject": "turtles - hohoho"}
    )

    # continue past interrupt
    assert await graph.ainvoke(None, config=config) == {
        "subjects": ["cats", "dogs"],
        "jokes": ["Joke about cats - hohoho", "Joke about turtles - hohoho"],
    }


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
async def test_weather_subgraph(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, ToolCall
    from langchain_core.tools import tool

    from langgraph.graph import MessagesState

    # setup subgraph

    @tool
    def get_weather(city: str):
        """Get the weather for a specific city"""
        return f"I'ts sunny in {city}!"

    weather_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tool_call123",
                        name="get_weather",
                        args={"city": "San Francisco"},
                    )
                ],
            )
        ]
    )

    class SubGraphState(MessagesState):
        city: str

    def model_node(state: SubGraphState, writer: StreamWriter):
        writer(" very")
        result = weather_model.invoke(state["messages"])
        return {"city": cast(AIMessage, result).tool_calls[0]["args"]["city"]}

    def weather_node(state: SubGraphState, writer: StreamWriter):
        writer(" good")
        result = get_weather.invoke({"city": state["city"]})
        return {"messages": [{"role": "assistant", "content": result}]}

    subgraph = StateGraph(SubGraphState)
    subgraph.add_node(model_node)
    subgraph.add_node(weather_node)
    subgraph.add_edge(START, "model_node")
    subgraph.add_edge("model_node", "weather_node")
    subgraph.add_edge("weather_node", END)
    subgraph = subgraph.compile(interrupt_before=["weather_node"])

    # setup main graph

    class RouterState(MessagesState):
        route: Literal["weather", "other"]

    router_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tool_call123",
                        name="router",
                        args={"dest": "weather"},
                    )
                ],
            )
        ]
    )

    def router_node(state: RouterState, writer: StreamWriter):
        writer("I'm")
        system_message = "Classify the incoming query as either about weather or not."
        messages = [{"role": "system", "content": system_message}] + state["messages"]
        route = router_model.invoke(messages)
        return {"route": cast(AIMessage, route).tool_calls[0]["args"]["dest"]}

    def normal_llm_node(state: RouterState):
        return {"messages": [AIMessage("Hello!")]}

    def route_after_prediction(state: RouterState):
        if state["route"] == "weather":
            return "weather_graph"
        else:
            return "normal_llm_node"

    def weather_graph(state: RouterState):
        # this tests that all async checkpointers tested also implement sync methods
        # as the subgraph called with sync invoke will use sync checkpointer methods
        return subgraph.invoke(state)

    graph = StateGraph(RouterState)
    graph.add_node(router_node)
    graph.add_node(normal_llm_node)
    graph.add_node("weather_graph", weather_graph)
    graph.add_edge(START, "router_node")
    graph.add_conditional_edges(
        "router_node",
        route_after_prediction,
        path_map=["weather_graph", "normal_llm_node"],
    )
    graph.add_edge("normal_llm_node", END)
    graph.add_edge("weather_graph", END)

    def get_first_in_list():
        return [*graph.get_state_history(config, limit=1)][0]

    graph = graph.compile(checkpointer=async_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    thread2 = {"configurable": {"thread_id": "2"}}
    inputs = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}

    # run with custom output
    assert [
        c
        async for c in graph.astream(
            inputs, thread2, stream_mode="custom", subgraphs=True
        )
    ] == [
        ((), "I'm"),
        ((AnyStr("weather_graph:"),), " very"),
    ]
    assert [
        c
        async for c in graph.astream(
            None, thread2, stream_mode="custom", subgraphs=True
        )
    ] == [
        ((AnyStr("weather_graph:"),), " good"),
    ]

    # run until interrupt
    assert [
        c
        async for c in graph.astream(
            inputs,
            config=config,
            stream_mode="updates",
            subgraphs=True,
            durability="exit",
        )
    ] == [
        ((), {"router_node": {"route": "weather"}}),
        ((AnyStr("weather_graph:"),), {"model_node": {"city": "San Francisco"}}),
        ((), {"__interrupt__": ()}),
    ]

    # check current state
    state = await graph.aget_state(config)
    assert state == StateSnapshot(
        values={
            "messages": [_AnyIdHumanMessage(content="what's the weather in sf")],
            "route": "weather",
        },
        next=("weather_graph",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "step": 1,
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="weather_graph",
                path=(PULL, "weather_graph"),
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("weather_graph:"),
                    }
                },
            ),
        ),
        interrupts=(),
    )
    # confirm that list() delegates to alist() correctly
    assert await asyncio.to_thread(get_first_in_list) == state

    # update
    await graph.aupdate_state(state.tasks[0].state, {"city": "la"})

    # run after update
    assert [
        c
        async for c in graph.astream(
            None, config=config, stream_mode="updates", subgraphs=True
        )
    ] == [
        (
            (AnyStr("weather_graph:"),),
            {
                "weather_node": {
                    "messages": [{"role": "assistant", "content": "I'ts sunny in la!"}]
                }
            },
        ),
        (
            (),
            {
                "weather_graph": {
                    "messages": [
                        _AnyIdHumanMessage(content="what's the weather in sf"),
                        _AnyIdAIMessage(content="I'ts sunny in la!"),
                    ]
                }
            },
        ),
    ]

    # try updating acting as weather node
    config = {"configurable": {"thread_id": "14"}}
    inputs = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    assert [
        c
        async for c in graph.astream(
            inputs,
            config=config,
            stream_mode="updates",
            subgraphs=True,
            durability="exit",
        )
    ] == [
        ((), {"router_node": {"route": "weather"}}),
        ((AnyStr("weather_graph:"),), {"model_node": {"city": "San Francisco"}}),
        ((), {"__interrupt__": ()}),
    ]
    state = await graph.aget_state(config, subgraphs=True)
    assert state == StateSnapshot(
        values={
            "messages": [_AnyIdHumanMessage(content="what's the weather in sf")],
            "route": "weather",
        },
        next=("weather_graph",),
        config={
            "configurable": {
                "thread_id": "14",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "step": 1,
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="weather_graph",
                path=(PULL, "weather_graph"),
                state=StateSnapshot(
                    values={
                        "messages": [
                            _AnyIdHumanMessage(content="what's the weather in sf")
                        ],
                        "city": "San Francisco",
                    },
                    next=("weather_node",),
                    config={
                        "configurable": {
                            "thread_id": "14",
                            "checkpoint_ns": AnyStr("weather_graph:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {
                                    "": AnyStr(),
                                    AnyStr("weather_graph:"): AnyStr(),
                                }
                            ),
                        }
                    },
                    metadata={
                        "source": "loop",
                        "step": 1,
                        "parents": {"": AnyStr()},
                    },
                    created_at=AnyStr(),
                    parent_config=None,
                    interrupts=(),
                    tasks=(
                        PregelTask(
                            id=AnyStr(),
                            name="weather_node",
                            path=(PULL, "weather_node"),
                        ),
                    ),
                ),
            ),
        ),
        interrupts=(),
    )
    await graph.aupdate_state(
        state.tasks[0].state.config,
        {"messages": [{"role": "assistant", "content": "rainy"}]},
        as_node="weather_node",
    )
    state = await graph.aget_state(config, subgraphs=True)
    assert state == StateSnapshot(
        values={
            "messages": [_AnyIdHumanMessage(content="what's the weather in sf")],
            "route": "weather",
        },
        next=("weather_graph",),
        config={
            "configurable": {
                "thread_id": "14",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "step": 1,
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        interrupts=(),
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="weather_graph",
                path=(PULL, "weather_graph"),
                state=StateSnapshot(
                    values={
                        "messages": [
                            _AnyIdHumanMessage(content="what's the weather in sf"),
                            _AnyIdAIMessage(content="rainy"),
                        ],
                        "city": "San Francisco",
                    },
                    next=(),
                    config={
                        "configurable": {
                            "thread_id": "14",
                            "checkpoint_ns": AnyStr("weather_graph:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {
                                    "": AnyStr(),
                                    AnyStr("weather_graph:"): AnyStr(),
                                }
                            ),
                        }
                    },
                    metadata={
                        "step": 2,
                        "source": "update",
                        "parents": {"": AnyStr()},
                    },
                    created_at=AnyStr(),
                    parent_config=(
                        {
                            "configurable": {
                                "thread_id": "14",
                                "checkpoint_ns": AnyStr("weather_graph:"),
                                "checkpoint_id": AnyStr(),
                                "checkpoint_map": AnyDict(
                                    {
                                        "": AnyStr(),
                                        AnyStr("weather_graph:"): AnyStr(),
                                    }
                                ),
                            }
                        }
                    ),
                    tasks=(),
                    interrupts=(),
                ),
            ),
        ),
    )
    assert [
        c
        async for c in graph.astream(
            None, config=config, stream_mode="updates", subgraphs=True
        )
    ] == [
        (
            (),
            {
                "weather_graph": {
                    "messages": [
                        _AnyIdHumanMessage(content="what's the weather in sf"),
                        _AnyIdAIMessage(content="rainy"),
                    ]
                }
            },
        ),
    ]

    # run with custom output, without subgraph streaming, should omit subgraph chunks
    assert [
        c
        async for c in graph.astream(
            inputs, {"configurable": {"thread_id": "3"}}, stream_mode="custom"
        )
    ] == [
        "I'm",
    ]

    # run with messages output, with subgraph streaming, should inc subgraph messages
    assert [
        c
        async for c in graph.astream(
            inputs,
            {"configurable": {"thread_id": "4"}},
            stream_mode="messages",
            subgraphs=True,
        )
    ] == [
        (
            (),
            (
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tool_call123",
                            name="router",
                            args={"dest": "weather"},
                        )
                    ],
                ),
                {
                    "thread_id": "4",
                    "langgraph_step": 1,
                    "langgraph_node": "router_node",
                    "langgraph_triggers": ("branch:to:router_node",),
                    "langgraph_path": ("__pregel_pull", "router_node"),
                    "langgraph_checkpoint_ns": AnyStr("router_node:"),
                    "checkpoint_ns": AnyStr("router_node:"),
                    "ls_provider": "fakemessageslistchatmodel",
                    "ls_model_type": "chat",
                },
            ),
        ),
        (
            (AnyStr("weather_graph:"),),
            (
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tool_call123",
                            name="get_weather",
                            args={"city": "San Francisco"},
                        )
                    ],
                ),
                {
                    "thread_id": "4",
                    "langgraph_step": 1,
                    "langgraph_node": "model_node",
                    "langgraph_triggers": ("branch:to:model_node",),
                    "langgraph_path": ("__pregel_pull", "model_node"),
                    "langgraph_checkpoint_ns": AnyStr("weather_graph:"),
                    "checkpoint_ns": AnyStr("weather_graph:"),
                    "ls_provider": "fakemessageslistchatmodel",
                    "ls_model_type": "chat",
                },
            ),
        ),
    ]

    # run with messages output, without subgraph streaming, should exc subgraph messages
    assert [
        c
        async for c in graph.astream(
            inputs,
            {"configurable": {"thread_id": "5"}},
            stream_mode="messages",
        )
    ] == [
        (
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tool_call123",
                        name="router",
                        args={"dest": "weather"},
                    )
                ],
            ),
            {
                "thread_id": "5",
                "langgraph_step": 1,
                "langgraph_node": "router_node",
                "langgraph_triggers": ("branch:to:router_node",),
                "langgraph_path": ("__pregel_pull", "router_node"),
                "langgraph_checkpoint_ns": AnyStr("router_node:"),
                "checkpoint_ns": AnyStr("router_node:"),
                "ls_provider": "fakemessageslistchatmodel",
                "ls_model_type": "chat",
            },
        ),
    ]
