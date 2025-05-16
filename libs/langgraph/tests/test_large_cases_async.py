import asyncio
import operator
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    Union,
    cast,
)

import httpx
import pytest
from langchain_core.messages import ToolCall
from langchain_core.runnables import RunnableConfig, RunnablePick
from pydantic import BaseModel
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion
from typing_extensions import TypedDict

from langgraph.channels.context import Context
from langgraph.channels.last_value import LastValue
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, PULL, PUSH, START
from langgraph.graph.graph import Graph
from langgraph.graph.message import MessageGraph, add_messages
from langgraph.graph.state import StateGraph
from langgraph.managed.shared_value import SharedValue
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.pregel import Channel, Pregel
from langgraph.store.memory import InMemoryStore
from langgraph.types import PregelTask, Send, StateSnapshot, StreamWriter
from tests.any_int import AnyInt
from tests.any_str import AnyDict, AnyStr, UnsortedSequence
from tests.conftest import (
    ALL_CHECKPOINTERS_ASYNC,
    awith_checkpointer,
)
from tests.fake_chat import FakeChatModel
from tests.fake_tracer import FakeTracer
from tests.messages import (
    _AnyIdAIMessage,
    _AnyIdAIMessageChunk,
    _AnyIdHumanMessage,
    _AnyIdToolMessage,
)

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_invoke_two_processes_in_out_interrupt(
    checkpointer_name: str, mocker: MockerFixture
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = Channel.subscribe_to("inbox") | add_one | Channel.write_to("output")
    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app = Pregel(
            nodes={"one": one, "two": two},
            channels={
                "inbox": LastValue(int),
                "output": LastValue(int),
                "input": LastValue(int),
            },
            input_channels="input",
            output_channels="output",
            checkpointer=checkpointer,
            interrupt_after_nodes=["one"],
        )
        thread1 = {"configurable": {"thread_id": "1"}}
        thread2 = {"configurable": {"thread_id": "2"}}

        # start execution, stop at inbox
        assert await app.ainvoke(2, thread1) is None

        # inbox == 3
        checkpoint = await checkpointer.aget(thread1)
        assert checkpoint is not None
        assert checkpoint["channel_values"]["inbox"] == 3

        # resume execution, finish
        assert await app.ainvoke(None, thread1) == 4

        # start execution again, stop at inbox
        assert await app.ainvoke(20, thread1) is None

        # inbox == 21
        checkpoint = await checkpointer.aget(thread1)
        assert checkpoint is not None
        assert checkpoint["channel_values"]["inbox"] == 21

        # send a new value in, interrupting the previous execution
        assert await app.ainvoke(3, thread1) is None
        assert await app.ainvoke(None, thread1) == 5

        # start execution again, stopping at inbox
        assert await app.ainvoke(20, thread2) is None

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

        if "shallow" in checkpointer_name:
            return

        # list history
        history = [c async for c in app.aget_state_history(thread1)]
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
                    "writes": {"two": 5},
                    "thread_id": "1",
                },
                created_at=AnyStr(),
                parent_config=history[1].config,
                interrupts=(),
            ),
            StateSnapshot(
                values={"inbox": 4, "output": 4, "input": 3},
                tasks=(
                    PregelTask(AnyStr(), "two", (PULL, "two"), result={"output": 5}),
                ),
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
                    "writes": {"one": None},
                    "thread_id": "1",
                },
                created_at=AnyStr(),
                parent_config=history[2].config,
                interrupts=(),
            ),
            StateSnapshot(
                values={"inbox": 21, "output": 4, "input": 3},
                tasks=(
                    PregelTask(AnyStr(), "one", (PULL, "one"), result={"inbox": 4}),
                ),
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
                    "writes": {"input": 3},
                    "thread_id": "1",
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
                    "writes": {"one": None},
                    "thread_id": "1",
                },
                created_at=AnyStr(),
                parent_config=history[4].config,
                interrupts=(),
            ),
            StateSnapshot(
                values={"inbox": 3, "output": 4, "input": 20},
                tasks=(
                    PregelTask(AnyStr(), "one", (PULL, "one"), result={"inbox": 21}),
                ),
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
                    "writes": {"input": 20},
                    "thread_id": "1",
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
                    "writes": {"two": 4},
                    "thread_id": "1",
                },
                created_at=AnyStr(),
                parent_config=history[6].config,
                interrupts=(),
            ),
            StateSnapshot(
                values={"inbox": 3, "input": 2},
                tasks=(
                    PregelTask(AnyStr(), "two", (PULL, "two"), result={"output": 4}),
                ),
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
                    "writes": {"one": None},
                    "thread_id": "1",
                },
                created_at=AnyStr(),
                parent_config=history[7].config,
                interrupts=(),
            ),
            StateSnapshot(
                values={"input": 2},
                tasks=(
                    PregelTask(AnyStr(), "one", (PULL, "one"), result={"inbox": 3}),
                ),
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
                    "writes": {"input": 2},
                    "thread_id": "1",
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
        c async for c in graph.astream(1, thread1, stream_mode=["values", "updates"])
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
                "writes": {"add_one": 1},
                "thread_id": "1",
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
                "writes": {"add_one": 1},
                "thread_id": "1",
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
                "writes": {"add_one": 1},
                "thread_id": "1",
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
                "writes": {"add_one": 1},
                "thread_id": "1",
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
                "writes": {"add_one": 1},
                "thread_id": "1",
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
                "writes": None,
                "thread_id": "1",
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
                "writes": {"__start__": 1},
                "thread_id": "1",
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_conditional_graph(checkpointer_name: str) -> None:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.tools import tool

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
        data = data.copy()
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = await {t.name: t for t in tools}[agent_action.tool].ainvoke(
            agent_action.tool_input
        )
        if data.get("intermediate_steps") is None:
            data["intermediate_steps"] = []
        else:
            data["intermediate_steps"] = data["intermediate_steps"].copy()
        data["intermediate_steps"].append([agent_action, observation])
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
                "input": "what is weather in sf",
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
            }
        },
        {
            "agent": {
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
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                ]
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
        },
    ]

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        # test state get/update methods with interrupt_after

        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
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
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            }
        ]

        assert await app_w_interrupt.aget_state(config) == StateSnapshot(
            values={
                "agent": {
                    "input": "what is weather in sf",
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                },
            },
            tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
            next=("tools",),
            config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
            created_at=(
                await app_w_interrupt.checkpointer.aget_tuple(config)
            ).checkpoint["ts"],
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": {
                    "agent": {
                        "agent": {
                            "input": "what is weather in sf",
                            "agent_outcome": AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:query",
                            ),
                        }
                    }
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
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
                "step": 1,
                "writes": {
                    "agent": {
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "input": "what is weather in sf",
                    }
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
        )

        assert [c async for c in app_w_interrupt.astream(None, config)] == [
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
            {
                "tools": {
                    "input": "what is weather in sf",
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
                    "input": "what is weather in sf",
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
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
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
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        ]
                    ],
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    ),
                },
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
                "step": 4,
                "writes": {
                    "agent": {
                        "input": "what is weather in sf",
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
                        "agent_outcome": AgentFinish(
                            return_values={"answer": "a really nice answer"},
                            log="finish:a really nice answer",
                        ),
                    }
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
        )

        # test state get/update methods with interrupt_before

        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
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
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            }
        ]

        assert await app_w_interrupt.aget_state(config) == StateSnapshot(
            values={
                "agent": {
                    "input": "what is weather in sf",
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                },
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
                "step": 0,
                "writes": {
                    "agent": {
                        "agent": {
                            "input": "what is weather in sf",
                            "agent_outcome": AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:query",
                            ),
                        }
                    }
                },
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
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
                "step": 1,
                "writes": {
                    "agent": {
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "input": "what is weather in sf",
                    }
                },
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
        )

        assert [c async for c in app_w_interrupt.astream(None, config)] == [
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
            {
                "tools": {
                    "input": "what is weather in sf",
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
                    "input": "what is weather in sf",
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
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
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
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        ]
                    ],
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    ),
                },
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
                "step": 4,
                "writes": {
                    "agent": {
                        "input": "what is weather in sf",
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
                        "agent_outcome": AgentFinish(
                            return_values={"answer": "a really nice answer"},
                            log="finish:a really nice answer",
                        ),
                    }
                },
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
        )

        # test re-invoke to continue with interrupt_before

        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["tools"],
        )
        config = {"configurable": {"thread_id": "3"}}
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
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            }
        ]

        assert await app_w_interrupt.aget_state(config) == StateSnapshot(
            values={
                "agent": {
                    "input": "what is weather in sf",
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                },
            },
            tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
            next=("tools",),
            config={
                "configurable": {
                    "thread_id": "3",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": {
                    "agent": {
                        "agent": {
                            "input": "what is weather in sf",
                            "agent_outcome": AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:query",
                            ),
                        }
                    }
                },
                "thread_id": "3",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
        )

        assert [c async for c in app_w_interrupt.astream(None, config)] == [
            {
                "agent": {
                    "input": "what is weather in sf",
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                },
            },
            {
                "tools": {
                    "input": "what is weather in sf",
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
                    "input": "what is weather in sf",
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
                "agent": {
                    "input": "what is weather in sf",
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
                }
            },
            {
                "agent": {
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
            },
        ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_conditional_graph_state(
    mocker: MockerFixture, checkpointer_name: str
) -> None:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import tool

    setup = mocker.Mock()
    teardown = mocker.Mock()

    @asynccontextmanager
    async def assert_ctx_once() -> AsyncIterator[None]:
        assert setup.call_count == 0
        assert teardown.call_count == 0
        try:
            yield
        finally:
            assert setup.call_count == 1
            assert teardown.call_count == 1
            setup.reset_mock()
            teardown.reset_mock()

    class MyPydanticContextModel(BaseModel, arbitrary_types_allowed=True):
        session: httpx.AsyncClient
        something_else: str

    @asynccontextmanager
    async def make_context(
        config: RunnableConfig,
    ) -> AsyncIterator[MyPydanticContextModel]:
        assert isinstance(config, dict)
        setup()
        session = httpx.AsyncClient()
        try:
            yield MyPydanticContextModel(session=session, something_else="hello")
        finally:
            await session.aclose()
            teardown()

    class AgentState(TypedDict):
        input: Annotated[str, UntrackedValue]
        agent_outcome: Optional[Union[AgentAction, AgentFinish]]
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
        context: Annotated[MyPydanticContextModel, Context(make_context)]

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
        # check we have httpx session in AgentState
        assert isinstance(data["context"], MyPydanticContextModel)
        # execute the tool
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        return {"intermediate_steps": [[agent_action, observation]]}

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        # check we have httpx session in AgentState
        assert isinstance(data["context"], MyPydanticContextModel)
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

    async with assert_ctx_once():
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

    async with assert_ctx_once():
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

    async with assert_ctx_once():
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        # test state get/update methods with interrupt_after

        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
            interrupt_after=["agent"],
        )
        config = {"configurable": {"thread_id": "1"}}

        async with assert_ctx_once():
            assert [
                c
                async for c in app_w_interrupt.astream(
                    {"input": "what is weather in sf"}, config
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
                "writes": {
                    "agent": {
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                    }
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
        )

        async with assert_ctx_once():
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
                "writes": {
                    "agent": {
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        )
                    }
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
        )

        async with assert_ctx_once():
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

        async with assert_ctx_once():
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
                "writes": {
                    "agent": {
                        "agent_outcome": AgentFinish(
                            return_values={"answer": "a really nice answer"},
                            log="finish:a really nice answer",
                        )
                    }
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
        )

        # test state get/update methods with interrupt_before

        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
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
                "writes": {
                    "agent": {
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                    }
                },
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
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
                "writes": {
                    "agent": {
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        )
                    }
                },
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
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
                "writes": {
                    "agent": {
                        "agent_outcome": AgentFinish(
                            return_values={"answer": "a really nice answer"},
                            log="finish:a really nice answer",
                        )
                    }
                },
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_state_graph_packets(checkpointer_name: str) -> None:
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
        session: Annotated[httpx.AsyncClient, Context(httpx.AsyncClient)]

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
        assert isinstance(data["session"], httpx.AsyncClient)
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        # interrupt after agent

        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
            interrupt_after=["agent"],
        )
        config = {"configurable": {"thread_id": "1"}}

        assert [
            c
            async for c in app_w_interrupt.astream(
                {"messages": HumanMessage(content="what is weather in sf")}, config
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
                "writes": {
                    "agent": {
                        "messages": AIMessage(
                            content="",
                            id="ai1",
                            tool_calls=[
                                {
                                    "name": "search_api",
                                    "args": {"query": "query"},
                                    "id": "tool_call123",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    }
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
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
                "writes": {
                    "agent": {
                        "messages": AIMessage(
                            id="ai1",
                            content="",
                            tool_calls=[
                                {
                                    "id": "tool_call123",
                                    "name": "search_api",
                                    "args": {"query": "a different query"},
                                },
                            ],
                        )
                    }
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
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
                "writes": {
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
                        ),
                    },
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
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
                "writes": {
                    "agent": {
                        "messages": AIMessage(content="answer", id="ai2"),
                    }
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
        )

        # interrupt before tools

        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["tools"],
        )
        config = {"configurable": {"thread_id": "2"}}
        model.i = 0

        assert [
            c
            async for c in app_w_interrupt.astream(
                {"messages": HumanMessage(content="what is weather in sf")}, config
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
                "writes": {
                    "agent": {
                        "messages": AIMessage(
                            content="",
                            additional_kwargs={},
                            response_metadata={},
                            id="ai1",
                            tool_calls=[
                                {
                                    "name": "search_api",
                                    "args": {"query": "query"},
                                    "id": "tool_call123",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    }
                },
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
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
                "writes": {
                    "agent": {
                        "messages": AIMessage(
                            id="ai1",
                            content="",
                            tool_calls=[
                                {
                                    "id": "tool_call123",
                                    "name": "search_api",
                                    "args": {"query": "a different query"},
                                },
                            ],
                        )
                    }
                },
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
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
                "writes": {
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
                        ),
                    },
                },
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
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
                "writes": {
                    "agent": {
                        "messages": AIMessage(content="answer", id="ai2"),
                    }
                },
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
            interrupts=(),
        )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_message_graph(checkpointer_name: str) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

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
    workflow = MessageGraph()

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

    assert await app.ainvoke(HumanMessage(content="what is weather in sf")) == [
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
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
                "writes": {
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
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
            ),
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
                "writes": {
                    "agent": AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call123",
                                "name": "search_api",
                                "args": {"query": "a different query"},
                            }
                        ],
                        id="ai1",
                    )
                },
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
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
                "writes": {
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
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
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
                "writes": {"agent": AIMessage(content="answer", id="ai2")},
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [
                    c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
                ][-1].config
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
                    "result": [("query", "query: what is weather in sf")],
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
                    "result": [("docs", ["doc3", "doc4"])],
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
                    "result": [("docs", ["doc1", "doc2"])],
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
                    "result": [("answer", "doc1,doc2,doc3,doc4")],
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_start_branch_then(checkpointer_name: str) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str
        shared: Annotated[dict[str, dict[str, Any]], SharedValue.on("assistant_id")]
        other: Annotated[dict[str, dict[str, Any]], SharedValue.on("assistant_id")]

    def assert_shared_value(data: State, config: RunnableConfig) -> State:
        assert "shared" in data
        if thread_id := config["configurable"].get("thread_id"):
            if thread_id == "1":
                # this is the first thread, so should not see a value
                assert data["shared"] == {}
                return {"shared": {"1": {"hello": "world"}}, "other": {"2": {1: 2}}}
            elif thread_id == "2":
                # this should get value saved by thread 1
                assert data["shared"] == {"1": {"hello": "world"}}
            elif thread_id == "3":
                # this is a different assistant, so should not see previous value
                assert data["shared"] == {}
        return {}

    def tool_two_slow(data: State, config: RunnableConfig) -> State:
        return {"my_key": " slow", **assert_shared_value(data, config)}

    def tool_two_fast(data: State, config: RunnableConfig) -> State:
        return {"my_key": " fast", **assert_shared_value(data, config)}

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two_slow", tool_two_slow)
    tool_two_graph.add_node("tool_two_fast", tool_two_fast)
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        tool_two = tool_two_graph.compile(
            store=InMemoryStore(),
            checkpointer=checkpointer,
            interrupt_before=["tool_two_fast", "tool_two_slow"],
        )

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            await tool_two.ainvoke({"my_key": "value", "market": "DE"})

        thread1 = {"configurable": {"thread_id": "1", "assistant_id": "a"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}, thread1) == {
            "my_key": "value",
            "market": "DE",
        }
        if "shallow" not in checkpointer_name:
            assert [c.metadata async for c in tool_two.checkpointer.alist(thread1)] == [
                {
                    "parents": {},
                    "source": "loop",
                    "step": 0,
                    "writes": None,
                    "assistant_id": "a",
                    "thread_id": "1",
                },
                {
                    "parents": {},
                    "source": "input",
                    "step": -1,
                    "writes": {"__start__": {"my_key": "value", "market": "DE"}},
                    "assistant_id": "a",
                    "thread_id": "1",
                },
            ]

        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value", "market": "DE"},
            tasks=(PregelTask(AnyStr(), "tool_two_slow", (PULL, "tool_two_slow")),),
            next=("tool_two_slow",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": None,
                "assistant_id": "a",
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread1, debug=1) == {
            "my_key": "value slow",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value slow", "market": "DE"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
                "writes": {"tool_two_slow": {"my_key": " slow"}},
                "assistant_id": "a",
                "thread_id": "1",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )

        thread2 = {"configurable": {"thread_id": "2", "assistant_id": "a"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread2) == {
            "my_key": "value",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value", "market": "US"},
            tasks=(PregelTask(AnyStr(), "tool_two_fast", (PULL, "tool_two_fast")),),
            next=("tool_two_fast",),
            config={
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": None,
                "assistant_id": "a",
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread2, debug=1) == {
            "my_key": "value fast",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value fast", "market": "US"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
                "writes": {"tool_two_fast": {"my_key": " fast"}},
                "assistant_id": "a",
                "thread_id": "2",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )

        thread3 = {"configurable": {"thread_id": "3", "assistant_id": "b"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread3) == {
            "my_key": "value",
            "market": "US",
        }
        assert await tool_two.aget_state(thread3) == StateSnapshot(
            values={"my_key": "value", "market": "US"},
            tasks=(PregelTask(AnyStr(), "tool_two_fast", (PULL, "tool_two_fast")),),
            next=("tool_two_fast",),
            config={
                "configurable": {
                    "thread_id": "3",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": None,
                "assistant_id": "b",
                "thread_id": "3",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread3, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )
        # update state
        await tool_two.aupdate_state(thread3, {"my_key": "key"})  # appends to my_key
        assert await tool_two.aget_state(thread3) == StateSnapshot(
            values={"my_key": "valuekey", "market": "US"},
            tasks=(PregelTask(AnyStr(), "tool_two_fast", (PULL, "tool_two_fast")),),
            next=("tool_two_fast",),
            config={
                "configurable": {
                    "thread_id": "3",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "update",
                "step": 1,
                "writes": {START: {"my_key": "key"}},
                "assistant_id": "b",
                "thread_id": "3",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread3, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread3, debug=1) == {
            "my_key": "valuekey fast",
            "market": "US",
        }
        assert await tool_two.aget_state(thread3) == StateSnapshot(
            values={"my_key": "valuekey fast", "market": "US"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "3",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 2,
                "writes": {"tool_two_fast": {"my_key": " fast"}},
                "assistant_id": "b",
                "thread_id": "3",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread3, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_branch_then(checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        # test stream_mode=debug
        tool_two = tool_two_graph.compile(checkpointer=checkpointer)
        thread10 = {"configurable": {"thread_id": "10"}}
        assert [
            c
            async for c in tool_two.astream(
                {"my_key": "value", "market": "DE"}, thread10, stream_mode="debug"
            )
        ] == [
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": -1,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {"my_key": ""},
                    "metadata": {
                        "parents": {},
                        "source": "input",
                        "step": -1,
                        "writes": {"__start__": {"my_key": "value", "market": "DE"}},
                        "thread_id": "10",
                    },
                    "parent_config": None,
                    "next": ["__start__"],
                    "tasks": [
                        {
                            "id": AnyStr(),
                            "name": "__start__",
                            "interrupts": (),
                            "state": None,
                        }
                    ],
                },
            },
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": 0,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {
                        "my_key": "value",
                        "market": "DE",
                    },
                    "metadata": {
                        "parents": {},
                        "source": "loop",
                        "step": 0,
                        "writes": None,
                        "thread_id": "10",
                    },
                    "parent_config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "next": ["prepare"],
                    "tasks": [
                        {
                            "id": AnyStr(),
                            "name": "prepare",
                            "interrupts": (),
                            "state": None,
                        }
                    ],
                },
            },
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "id": AnyStr(),
                    "name": "prepare",
                    "input": {"my_key": "value", "market": "DE"},
                    "triggers": ("branch:to:prepare",),
                },
            },
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "id": AnyStr(),
                    "name": "prepare",
                    "result": [("my_key", " prepared")],
                    "error": None,
                    "interrupts": [],
                },
            },
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {
                        "my_key": "value prepared",
                        "market": "DE",
                    },
                    "metadata": {
                        "parents": {},
                        "source": "loop",
                        "step": 1,
                        "writes": {"prepare": {"my_key": " prepared"}},
                        "thread_id": "10",
                    },
                    "parent_config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "next": ["tool_two_slow"],
                    "tasks": [
                        {
                            "id": AnyStr(),
                            "name": "tool_two_slow",
                            "interrupts": (),
                            "state": None,
                        }
                    ],
                },
            },
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": AnyStr(),
                    "name": "tool_two_slow",
                    "input": {"my_key": "value prepared", "market": "DE"},
                    "triggers": ("branch:to:tool_two_slow",),
                },
            },
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": AnyStr(),
                    "name": "tool_two_slow",
                    "result": [("my_key", " slow")],
                    "error": None,
                    "interrupts": [],
                },
            },
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {
                        "my_key": "value prepared slow",
                        "market": "DE",
                    },
                    "metadata": {
                        "parents": {},
                        "source": "loop",
                        "step": 2,
                        "writes": {"tool_two_slow": {"my_key": " slow"}},
                        "thread_id": "10",
                    },
                    "parent_config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "next": ["finish"],
                    "tasks": [
                        {
                            "id": AnyStr(),
                            "name": "finish",
                            "interrupts": (),
                            "state": None,
                        }
                    ],
                },
            },
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 3,
                "payload": {
                    "id": AnyStr(),
                    "name": "finish",
                    "input": {"my_key": "value prepared slow", "market": "DE"},
                    "triggers": (
                        "branch:prepare:condition::then",
                        "branch:to:finish",
                    ),
                },
            },
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 3,
                "payload": {
                    "id": AnyStr(),
                    "name": "finish",
                    "result": [("my_key", " finished")],
                    "error": None,
                    "interrupts": [],
                },
            },
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": 3,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {
                        "my_key": "value prepared slow finished",
                        "market": "DE",
                    },
                    "metadata": {
                        "parents": {},
                        "source": "loop",
                        "step": 3,
                        "writes": {"finish": {"my_key": " finished"}},
                        "thread_id": "10",
                    },
                    "parent_config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "next": [],
                    "tasks": [],
                },
            },
        ]

        tool_two = tool_two_graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["tool_two_fast", "tool_two_slow"],
        )

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            await tool_two.ainvoke({"my_key": "value", "market": "DE"})

        thread1 = {"configurable": {"thread_id": "11"}}
        # stop when about to enter node
        assert [
            c
            async for c in tool_two.astream(
                {"my_key": "value", "market": "DE"}, thread1, stream_mode="debug"
            )
        ] == [
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": -1,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "11"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "11",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {"my_key": ""},
                    "metadata": {
                        "parents": {},
                        "source": "input",
                        "step": -1,
                        "writes": {"__start__": {"my_key": "value", "market": "DE"}},
                        "thread_id": "11",
                    },
                    "parent_config": None,
                    "next": ["__start__"],
                    "tasks": [
                        {
                            "id": AnyStr(),
                            "name": "__start__",
                            "interrupts": (),
                            "state": None,
                        }
                    ],
                },
            },
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": 0,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "11"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "11",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {
                        "my_key": "value",
                        "market": "DE",
                    },
                    "metadata": {
                        "parents": {},
                        "source": "loop",
                        "step": 0,
                        "writes": None,
                        "thread_id": "11",
                    },
                    "parent_config": {
                        "tags": [],
                        "metadata": {"thread_id": "11"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "11",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "next": ["prepare"],
                    "tasks": [
                        {
                            "id": AnyStr(),
                            "name": "prepare",
                            "interrupts": (),
                            "state": None,
                        }
                    ],
                },
            },
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "id": AnyStr(),
                    "name": "prepare",
                    "input": {"my_key": "value", "market": "DE"},
                    "triggers": ("branch:to:prepare",),
                },
            },
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "id": AnyStr(),
                    "name": "prepare",
                    "result": [("my_key", " prepared")],
                    "error": None,
                    "interrupts": [],
                },
            },
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "11"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "11",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {
                        "my_key": "value prepared",
                        "market": "DE",
                    },
                    "metadata": {
                        "parents": {},
                        "source": "loop",
                        "step": 1,
                        "writes": {"prepare": {"my_key": " prepared"}},
                        "thread_id": "11",
                    },
                    "parent_config": {
                        "tags": [],
                        "metadata": {"thread_id": "11"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "11",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "next": ["tool_two_slow"],
                    "tasks": [
                        {
                            "id": AnyStr(),
                            "name": "tool_two_slow",
                            "interrupts": (),
                            "state": None,
                        }
                    ],
                },
            },
        ]
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared", "market": "DE"},
            tasks=(PregelTask(AnyStr(), "tool_two_slow", (PULL, "tool_two_slow")),),
            next=("tool_two_slow",),
            config={
                "configurable": {
                    "thread_id": "11",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
                "writes": {"prepare": {"my_key": " prepared"}},
                "thread_id": "11",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread1, debug=1) == {
            "my_key": "value prepared slow finished",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared slow finished", "market": "DE"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "11",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
                "writes": {"finish": {"my_key": " finished"}},
                "thread_id": "11",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )

        thread2 = {"configurable": {"thread_id": "12"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread2) == {
            "my_key": "value prepared",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared", "market": "US"},
            tasks=(PregelTask(AnyStr(), "tool_two_fast", (PULL, "tool_two_fast")),),
            next=("tool_two_fast",),
            config={
                "configurable": {
                    "thread_id": "12",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
                "writes": {"prepare": {"my_key": " prepared"}},
                "thread_id": "12",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread2, debug=1) == {
            "my_key": "value prepared fast finished",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared fast finished", "market": "US"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "12",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
                "writes": {"finish": {"my_key": " finished"}},
                "thread_id": "12",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )

        tool_two = tool_two_graph.compile(
            checkpointer=checkpointer, interrupt_after=["prepare"]
        )

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            await tool_two.ainvoke({"my_key": "value", "market": "DE"})

        thread1 = {"configurable": {"thread_id": "21"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}, thread1) == {
            "my_key": "value prepared",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared", "market": "DE"},
            tasks=(PregelTask(AnyStr(), "tool_two_slow", (PULL, "tool_two_slow")),),
            next=("tool_two_slow",),
            config={
                "configurable": {
                    "thread_id": "21",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
                "writes": {"prepare": {"my_key": " prepared"}},
                "thread_id": "21",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread1, debug=1) == {
            "my_key": "value prepared slow finished",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared slow finished", "market": "DE"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "21",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
                "writes": {"finish": {"my_key": " finished"}},
                "thread_id": "21",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )

        thread2 = {"configurable": {"thread_id": "22"}}
        # stop when about to enter node
        assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread2) == {
            "my_key": "value prepared",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared", "market": "US"},
            tasks=(PregelTask(AnyStr(), "tool_two_fast", (PULL, "tool_two_fast")),),
            next=("tool_two_fast",),
            config={
                "configurable": {
                    "thread_id": "22",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
                "writes": {"prepare": {"my_key": " prepared"}},
                "thread_id": "22",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread2, debug=1) == {
            "my_key": "value prepared fast finished",
            "market": "US",
        }
        assert await tool_two.aget_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared fast finished", "market": "US"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "22",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
                "writes": {"finish": {"my_key": " finished"}},
                "thread_id": "22",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )

        thread3 = {"configurable": {"thread_id": "23"}}
        # update an empty thread before first run
        uconfig = await tool_two.aupdate_state(
            thread3, {"my_key": "key", "market": "DE"}
        )
        # check current state
        assert await tool_two.aget_state(thread3) == StateSnapshot(
            values={"my_key": "key", "market": "DE"},
            tasks=(PregelTask(AnyStr(), "prepare", (PULL, "prepare")),),
            next=("prepare",),
            config=uconfig,
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "update",
                "step": 0,
                "writes": {START: {"my_key": "key", "market": "DE"}},
                "thread_id": "23",
            },
            parent_config=None,
            interrupts=(),
        )
        # run from this point
        assert await tool_two.ainvoke(None, thread3) == {
            "my_key": "key prepared",
            "market": "DE",
        }
        # get state after first node
        assert await tool_two.aget_state(thread3) == StateSnapshot(
            values={"my_key": "key prepared", "market": "DE"},
            tasks=(PregelTask(AnyStr(), "tool_two_slow", (PULL, "tool_two_slow")),),
            next=("tool_two_slow",),
            config={
                "configurable": {
                    "thread_id": "23",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
                "writes": {"prepare": {"my_key": " prepared"}},
                "thread_id": "23",
            },
            parent_config=(None if "shallow" in checkpointer_name else uconfig),
            interrupts=(),
        )
        # resume, for same result as above
        assert await tool_two.ainvoke(None, thread3, debug=1) == {
            "my_key": "key prepared slow finished",
            "market": "DE",
        }
        assert await tool_two.aget_state(thread3) == StateSnapshot(
            values={"my_key": "key prepared slow finished", "market": "DE"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "23",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            created_at=AnyStr(),
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
                "writes": {"finish": {"my_key": " finished"}},
                "thread_id": "23",
            },
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else [c async for c in tool_two.checkpointer.alist(thread3, limit=2)][
                    -1
                ].config
            ),
            interrupts=(),
        )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_nested_graph_state(checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}
        await app.ainvoke({"my_key": "my value"}, config, debug=True)
        # test state w/ nested subgraph state (right after interrupt)
        # first get_state without subgraph state
        assert await app.aget_state(config) == StateSnapshot(
            values={"my_key": "hi my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "inner",
                    (PULL, "inner"),
                    state={
                        "configurable": {"thread_id": "1", "checkpoint_ns": AnyStr()}
                    },
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
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else {
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                }
            ),
            interrupts=(),
        )
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
                            "writes": {
                                "inner_1": {
                                    "my_key": "hi my value here",
                                    "my_other_key": "hi my value",
                                }
                            },
                            "step": 1,
                            "thread_id": "1",
                            "langgraph_node": "inner",
                            "langgraph_path": [PULL, "inner"],
                            "langgraph_step": 2,
                            "langgraph_triggers": ["branch:to:inner"],
                            "langgraph_checkpoint_ns": AnyStr("inner:"),
                        },
                        created_at=AnyStr(),
                        parent_config=(
                            None
                            if "shallow" in checkpointer_name
                            else {
                                "configurable": {
                                    "thread_id": "1",
                                    "checkpoint_ns": AnyStr("inner:"),
                                    "checkpoint_id": AnyStr(),
                                    "checkpoint_map": AnyDict(
                                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                                    ),
                                }
                            }
                        ),
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
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else {
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                }
            ),
            interrupts=(),
        )
        # get_state_history returns outer graph checkpoints
        history = [c async for c in app.aget_state_history(config)]
        expected_history = [
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
                                "checkpoint_ns": AnyStr("inner:"),
                            }
                        },
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
                    "writes": {"outer_1": {"my_key": "hi my value"}},
                    "step": 1,
                    "thread_id": "1",
                },
                created_at=AnyStr(),
                parent_config=(
                    None
                    if "shallow" in checkpointer_name
                    else {
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
                values={"my_key": "my value"},
                tasks=(
                    PregelTask(
                        AnyStr(),
                        "outer_1",
                        (PULL, "outer_1"),
                        result={"my_key": "hi my value"},
                    ),
                ),
                next=("outer_1",),
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
                    "writes": None,
                    "step": 0,
                    "thread_id": "1",
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
                values={},
                tasks=(
                    PregelTask(
                        AnyStr(),
                        "__start__",
                        (PULL, "__start__"),
                        result={"my_key": "my value"},
                    ),
                ),
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
                    "writes": {"__start__": {"my_key": "my value"}},
                    "step": -1,
                    "thread_id": "1",
                },
                created_at=AnyStr(),
                parent_config=None,
                interrupts=(),
            ),
        ]

        if "shallow" in checkpointer_name:
            expected_history = expected_history[:1]

        assert history == expected_history

        # get_state_history for a subgraph returns its checkpoints
        child_history = [
            c async for c in app.aget_state_history(history[0].tasks[0].state)
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
                    "writes": {
                        "inner_1": {
                            "my_key": "hi my value here",
                            "my_other_key": "hi my value",
                        }
                    },
                    "step": 1,
                    "parents": {"": AnyStr()},
                    "thread_id": "1",
                    "langgraph_node": "inner",
                    "langgraph_path": [PULL, "inner"],
                    "langgraph_step": 2,
                    "langgraph_triggers": ["branch:to:inner"],
                    "langgraph_checkpoint_ns": AnyStr("inner:"),
                },
                created_at=AnyStr(),
                parent_config=(
                    None
                    if "shallow" in checkpointer_name
                    else {
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("inner:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {"": AnyStr(), AnyStr("child:"): AnyStr()}
                            ),
                        }
                    }
                ),
                tasks=(PregelTask(AnyStr(), "inner_2", (PULL, "inner_2")),),
                interrupts=(),
            ),
            StateSnapshot(
                values={"my_key": "hi my value"},
                next=("inner_1",),
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
                    "writes": None,
                    "step": 0,
                    "parents": {"": AnyStr()},
                    "thread_id": "1",
                    "langgraph_node": "inner",
                    "langgraph_path": [PULL, "inner"],
                    "langgraph_step": 2,
                    "langgraph_triggers": ["branch:to:inner"],
                    "langgraph_checkpoint_ns": AnyStr("inner:"),
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("inner:"),
                        "checkpoint_id": AnyStr(),
                        "checkpoint_map": AnyDict(
                            {"": AnyStr(), AnyStr("child:"): AnyStr()}
                        ),
                    }
                },
                tasks=(
                    PregelTask(
                        AnyStr(),
                        "inner_1",
                        (PULL, "inner_1"),
                        result={
                            "my_key": "hi my value here",
                            "my_other_key": "hi my value",
                        },
                    ),
                ),
                interrupts=(),
            ),
            StateSnapshot(
                values={},
                next=("__start__",),
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
                    "source": "input",
                    "writes": {"__start__": {"my_key": "hi my value"}},
                    "step": -1,
                    "parents": {"": AnyStr()},
                    "thread_id": "1",
                    "langgraph_node": "inner",
                    "langgraph_path": [PULL, "inner"],
                    "langgraph_step": 2,
                    "langgraph_triggers": ["branch:to:inner"],
                    "langgraph_checkpoint_ns": AnyStr("inner:"),
                },
                created_at=AnyStr(),
                parent_config=None,
                tasks=(
                    PregelTask(
                        AnyStr(),
                        "__start__",
                        (PULL, "__start__"),
                        result={"my_key": "hi my value"},
                    ),
                ),
                interrupts=(),
            ),
        ]

        if "shallow" in checkpointer_name:
            expected_child_history = expected_child_history[:1]

        assert child_history == expected_child_history

        # resume
        await app.ainvoke(None, config, debug=True)
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
                "writes": {
                    "outer_2": {"my_key": "hi my value here and there and back again"}
                },
                "step": 3,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else {
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
                    "writes": {
                        "outer_2": {
                            "my_key": "hi my value here and there and back again"
                        }
                    },
                    "step": 3,
                    "thread_id": "1",
                },
                created_at=AnyStr(),
                parent_config=(
                    None
                    if "shallow" in checkpointer_name
                    else {
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
                values={"my_key": "hi my value here and there"},
                tasks=(
                    PregelTask(
                        AnyStr(),
                        "outer_2",
                        (PULL, "outer_2"),
                        result={"my_key": "hi my value here and there and back again"},
                    ),
                ),
                next=("outer_2",),
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
                    "writes": {"inner": {"my_key": "hi my value here and there"}},
                    "step": 2,
                    "thread_id": "1",
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
                        "inner",
                        (PULL, "inner"),
                        state={
                            "configurable": {
                                "thread_id": "1",
                                "checkpoint_ns": AnyStr(),
                            }
                        },
                        result={"my_key": "hi my value here and there"},
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
                    "writes": {"outer_1": {"my_key": "hi my value"}},
                    "step": 1,
                    "thread_id": "1",
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
                values={"my_key": "my value"},
                tasks=(
                    PregelTask(
                        AnyStr(),
                        "outer_1",
                        (PULL, "outer_1"),
                        result={"my_key": "hi my value"},
                    ),
                ),
                next=("outer_1",),
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
                    "writes": None,
                    "step": 0,
                    "thread_id": "1",
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
                values={},
                tasks=(
                    PregelTask(
                        AnyStr(),
                        "__start__",
                        (PULL, "__start__"),
                        result={"my_key": "my value"},
                    ),
                ),
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
                    "writes": {"__start__": {"my_key": "my value"}},
                    "step": -1,
                    "thread_id": "1",
                },
                created_at=AnyStr(),
                parent_config=None,
                interrupts=(),
            ),
        ]
        if "shallow" in checkpointer_name:
            expected_history = expected_history[:1]

        assert actual_history == expected_history
        # test looking up parent state by checkpoint ID
        for actual_snapshot, expected_snapshot in zip(actual_history, expected_history):
            assert await app.aget_state(actual_snapshot.config) == expected_snapshot


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_doubly_nested_graph_state(checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        # test invoke w/ nested interrupt
        config = {"configurable": {"thread_id": "1"}}
        assert [
            c async for c in app.astream({"my_key": "my value"}, config, subgraphs=True)
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
                "writes": {"parent_1": {"my_key": "hi my value"}},
                "step": 1,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else {
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                }
            ),
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
                "langgraph_checkpoint_ns": AnyStr("child:"),
                "langgraph_node": "child",
                "langgraph_path": ["__pregel_pull", "child"],
                "langgraph_step": 2,
                "langgraph_triggers": ["branch:to:child"],
                "parents": {"": AnyStr()},
                "source": "loop",
                "writes": None,
                "step": 0,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else {
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
                }
            ),
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
                "writes": {"grandchild_1": {"my_key": "hi my value here"}},
                "step": 1,
                "thread_id": "1",
                "langgraph_checkpoint_ns": AnyStr("child:"),
                "langgraph_node": "child_1",
                "langgraph_path": [PULL, AnyStr("child_1")],
                "langgraph_step": 1,
                "langgraph_triggers": [
                    "branch:to:child_1",
                ],
            },
            created_at=AnyStr(),
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else {
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
                }
            ),
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
                                        "writes": {
                                            "grandchild_1": {
                                                "my_key": "hi my value here"
                                            }
                                        },
                                        "step": 1,
                                        "thread_id": "1",
                                        "langgraph_checkpoint_ns": AnyStr("child:"),
                                        "langgraph_node": "child_1",
                                        "langgraph_path": [
                                            PULL,
                                            AnyStr("child_1"),
                                        ],
                                        "langgraph_step": 1,
                                        "langgraph_triggers": [
                                            "branch:to:child_1",
                                        ],
                                    },
                                    created_at=AnyStr(),
                                    parent_config=(
                                        None
                                        if "shallow" in checkpointer_name
                                        else {
                                            "configurable": {
                                                "thread_id": "1",
                                                "checkpoint_ns": AnyStr(),
                                                "checkpoint_id": AnyStr(),
                                                "checkpoint_map": AnyDict(
                                                    {
                                                        "": AnyStr(),
                                                        AnyStr("child:"): AnyStr(),
                                                        AnyStr(
                                                            re.compile(
                                                                r"child:.+|child1:"
                                                            )
                                                        ): AnyStr(),
                                                    }
                                                ),
                                            }
                                        }
                                    ),
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
                            "writes": None,
                            "step": 0,
                            "thread_id": "1",
                            "langgraph_node": "child",
                            "langgraph_path": [PULL, AnyStr("child")],
                            "langgraph_step": 2,
                            "langgraph_triggers": [
                                "branch:to:child",
                            ],
                            "langgraph_checkpoint_ns": AnyStr("child:"),
                        },
                        created_at=AnyStr(),
                        parent_config=(
                            None
                            if "shallow" in checkpointer_name
                            else {
                                "configurable": {
                                    "thread_id": "1",
                                    "checkpoint_ns": AnyStr("child:"),
                                    "checkpoint_id": AnyStr(),
                                    "checkpoint_map": AnyDict(
                                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                                    ),
                                }
                            }
                        ),
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
                "writes": {"parent_1": {"my_key": "hi my value"}},
                "step": 1,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else {
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                }
            ),
            interrupts=(),
        )
        # resume
        assert [c async for c in app.astream(None, config, subgraphs=True)] == [
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
                    "writes": {
                        "parent_2": {
                            "my_key": "hi my value here and there and back again"
                        }
                    },
                    "step": 3,
                    "thread_id": "1",
                },
                created_at=AnyStr(),
                parent_config=(
                    None
                    if "shallow" in checkpointer_name
                    else {
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

        if "shallow" in checkpointer_name:
            return

        # get outer graph history
        outer_history = [c async for c in app.aget_state_history(config)]
        assert (
            outer_history[0]
            == [
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
                        "writes": {
                            "parent_2": {
                                "my_key": "hi my value here and there and back again"
                            }
                        },
                        "step": 3,
                        "thread_id": "1",
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
                    values={"my_key": "hi my value here and there"},
                    next=("parent_2",),
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
                        "writes": {"child": {"my_key": "hi my value here and there"}},
                        "step": 2,
                        "thread_id": "1",
                    },
                    created_at=AnyStr(),
                    parent_config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        }
                    },
                    tasks=(
                        PregelTask(
                            id=AnyStr(), name="parent_2", path=(PULL, "parent_2")
                        ),
                    ),
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
                        "writes": {"parent_1": {"my_key": "hi my value"}},
                        "step": 1,
                        "thread_id": "1",
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
                    values={"my_key": "my value"},
                    next=("parent_1",),
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
                        "writes": None,
                        "step": 0,
                        "thread_id": "1",
                    },
                    created_at=AnyStr(),
                    parent_config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        }
                    },
                    tasks=(
                        PregelTask(
                            id=AnyStr(), name="parent_1", path=(PULL, "parent_1")
                        ),
                    ),
                    interrupts=(),
                ),
                StateSnapshot(
                    values={},
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
                        "writes": {"my_key": "my value"},
                        "step": -1,
                        "thread_id": "1",
                    },
                    created_at=AnyStr(),
                    parent_config=None,
                    tasks=(
                        PregelTask(
                            id=AnyStr(), name="__start__", path=(PULL, "__start__")
                        ),
                    ),
                    interrupts=(),
                ),
            ][0]
        )
        # get child graph history
        child_history = [
            c async for c in app.aget_state_history(outer_history[2].tasks[0].state)
        ]
        assert child_history == [
            StateSnapshot(
                values={"my_key": "hi my value here and there"},
                next=(),
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
                    "writes": {"child_1": {"my_key": "hi my value here and there"}},
                    "step": 1,
                    "parents": {"": AnyStr()},
                    "thread_id": "1",
                    "langgraph_node": "child",
                    "langgraph_path": [PULL, AnyStr("child")],
                    "langgraph_step": 2,
                    "langgraph_triggers": ["branch:to:child"],
                    "langgraph_checkpoint_ns": AnyStr("child:"),
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("child:"),
                        "checkpoint_id": AnyStr(),
                        "checkpoint_map": AnyDict(
                            {"": AnyStr(), AnyStr("child:"): AnyStr()}
                        ),
                    }
                },
                tasks=(),
                interrupts=(),
            ),
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
                    "writes": None,
                    "step": 0,
                    "parents": {"": AnyStr()},
                    "thread_id": "1",
                    "langgraph_node": "child",
                    "langgraph_path": [PULL, AnyStr("child")],
                    "langgraph_step": 2,
                    "langgraph_triggers": ["branch:to:child"],
                    "langgraph_checkpoint_ns": AnyStr("child:"),
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("child:"),
                        "checkpoint_id": AnyStr(),
                        "checkpoint_map": AnyDict(
                            {"": AnyStr(), AnyStr("child:"): AnyStr()}
                        ),
                    }
                },
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
                        result={"my_key": "hi my value here and there"},
                    ),
                ),
                interrupts=(),
            ),
            StateSnapshot(
                values={},
                next=("__start__",),
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
                    "source": "input",
                    "writes": {"__start__": {"my_key": "hi my value"}},
                    "step": -1,
                    "parents": {"": AnyStr()},
                    "thread_id": "1",
                    "langgraph_node": "child",
                    "langgraph_path": [PULL, AnyStr("child")],
                    "langgraph_step": 2,
                    "langgraph_triggers": ["branch:to:child"],
                    "langgraph_checkpoint_ns": AnyStr("child:"),
                },
                created_at=AnyStr(),
                parent_config=None,
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="__start__",
                        path=(PULL, "__start__"),
                        result={"my_key": "hi my value"},
                    ),
                ),
                interrupts=(),
            ),
        ]
        # get grandchild graph history
        grandchild_history = [
            c async for c in app.aget_state_history(child_history[1].tasks[0].state)
        ]
        assert grandchild_history == [
            StateSnapshot(
                values={"my_key": "hi my value here and there"},
                next=(),
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
                    "writes": {
                        "grandchild_2": {"my_key": "hi my value here and there"}
                    },
                    "step": 2,
                    "parents": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                        }
                    ),
                    "thread_id": "1",
                    "langgraph_checkpoint_ns": AnyStr("child:"),
                    "langgraph_node": "child_1",
                    "langgraph_path": [
                        PULL,
                        AnyStr("child_1"),
                    ],
                    "langgraph_step": 1,
                    "langgraph_triggers": [
                        "branch:to:child_1",
                    ],
                },
                created_at=AnyStr(),
                parent_config={
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
                tasks=(),
                interrupts=(),
            ),
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
                    "writes": {"grandchild_1": {"my_key": "hi my value here"}},
                    "step": 1,
                    "parents": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                        }
                    ),
                    "thread_id": "1",
                    "langgraph_checkpoint_ns": AnyStr("child:"),
                    "langgraph_node": "child_1",
                    "langgraph_path": [
                        PULL,
                        AnyStr("child_1"),
                    ],
                    "langgraph_step": 1,
                    "langgraph_triggers": [
                        "branch:to:child_1",
                    ],
                },
                created_at=AnyStr(),
                parent_config={
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
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="grandchild_2",
                        path=(PULL, "grandchild_2"),
                        result={"my_key": "hi my value here and there"},
                    ),
                ),
                interrupts=(),
            ),
            StateSnapshot(
                values={"my_key": "hi my value"},
                next=("grandchild_1",),
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
                    "writes": None,
                    "step": 0,
                    "parents": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                        }
                    ),
                    "thread_id": "1",
                    "langgraph_checkpoint_ns": AnyStr("child:"),
                    "langgraph_node": "child_1",
                    "langgraph_path": [
                        PULL,
                        AnyStr("child_1"),
                    ],
                    "langgraph_step": 1,
                    "langgraph_triggers": [
                        "branch:to:child_1",
                    ],
                },
                created_at=AnyStr(),
                parent_config={
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
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="grandchild_1",
                        path=(PULL, "grandchild_1"),
                        result={"my_key": "hi my value here"},
                    ),
                ),
                interrupts=(),
            ),
            StateSnapshot(
                values={},
                next=("__start__",),
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
                    "source": "input",
                    "writes": {"__start__": {"my_key": "hi my value"}},
                    "step": -1,
                    "parents": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                        }
                    ),
                    "thread_id": "1",
                    "langgraph_checkpoint_ns": AnyStr("child:"),
                    "langgraph_node": "child_1",
                    "langgraph_path": [
                        PULL,
                        AnyStr("child_1"),
                    ],
                    "langgraph_step": 1,
                    "langgraph_triggers": [
                        "branch:to:child_1",
                    ],
                },
                created_at=AnyStr(),
                parent_config=None,
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="__start__",
                        path=(PULL, "__start__"),
                        result={"my_key": "hi my value"},
                    ),
                ),
                interrupts=(),
            ),
        ]

        # replay grandchild checkpoint
        assert [
            c
            async for c in app.astream(
                None, grandchild_history[2].config, subgraphs=True
            )
        ] == [
            (
                (AnyStr("child:"), AnyStr("child_1:")),
                {"grandchild_1": {"my_key": "hi my value here"}},
            ),
            ((), {"__interrupt__": ()}),
        ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_send_to_nested_graphs(checkpointer_name: str) -> None:
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
    subgraph = StateGraph(JokeState, output=OverallState)
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
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
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_weather_subgraph(
    checkpointer_name: str, snapshot: SnapshotAssertion
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

    class Router(TypedDict):
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}
        thread2 = {"configurable": {"thread_id": "2"}}
        inputs = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}

        # run with custom output
        assert [
            c async for c in graph.astream(inputs, thread2, stream_mode="custom")
        ] == [
            "I'm",
            " very",
        ]
        assert [
            c async for c in graph.astream(None, thread2, stream_mode="custom")
        ] == [
            " good",
        ]

        # run until interrupt
        assert [
            c
            async for c in graph.astream(
                inputs, config=config, stream_mode="updates", subgraphs=True
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
                "writes": {"router_node": {"route": "weather"}},
                "step": 1,
                "parents": {},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else {
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                }
            ),
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
                        "messages": [
                            {"role": "assistant", "content": "I'ts sunny in la!"}
                        ]
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
                inputs, config=config, stream_mode="updates", subgraphs=True
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
                "writes": {"router_node": {"route": "weather"}},
                "step": 1,
                "parents": {},
                "thread_id": "14",
            },
            created_at=AnyStr(),
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else {
                    "configurable": {
                        "thread_id": "14",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                }
            ),
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
                            "writes": {"model_node": {"city": "San Francisco"}},
                            "step": 1,
                            "parents": {"": AnyStr()},
                            "thread_id": "14",
                            "langgraph_node": "weather_graph",
                            "langgraph_path": [PULL, "weather_graph"],
                            "langgraph_step": 2,
                            "langgraph_triggers": ["branch:to:weather_graph"],
                            "langgraph_checkpoint_ns": AnyStr("weather_graph:"),
                        },
                        created_at=AnyStr(),
                        parent_config=(
                            None
                            if "shallow" in checkpointer_name
                            else {
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
                "writes": {"router_node": {"route": "weather"}},
                "step": 1,
                "parents": {},
                "thread_id": "14",
            },
            created_at=AnyStr(),
            parent_config=(
                None
                if "shallow" in checkpointer_name
                else {
                    "configurable": {
                        "thread_id": "14",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                }
            ),
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
                            "writes": {
                                "weather_node": {
                                    "messages": [
                                        {"role": "assistant", "content": "rainy"}
                                    ]
                                }
                            },
                            "parents": {"": AnyStr()},
                            "thread_id": "14",
                            "checkpoint_id": AnyStr(),
                            "checkpoint_ns": AnyStr("weather_graph:"),
                            "langgraph_node": "weather_graph",
                            "langgraph_path": [PULL, "weather_graph"],
                            "langgraph_step": 2,
                            "langgraph_triggers": ["branch:to:weather_graph"],
                            "langgraph_checkpoint_ns": AnyStr("weather_graph:"),
                        },
                        created_at=AnyStr(),
                        parent_config=(
                            None
                            if "shallow" in checkpointer_name
                            else {
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
