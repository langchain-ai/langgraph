import json
import operator
import re
import time
from dataclasses import replace
from typing import Annotated, Any, Literal, cast

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolCall
from langchain_core.runnables import RunnableConfig, RunnableMap, RunnablePick
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion
from typing_extensions import TypedDict

from langgraph._internal._constants import PULL, PUSH
from langgraph.channels.last_value import LastValue
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState, add_messages
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.types import (
    Command,
    Durability,
    Interrupt,
    PregelTask,
    RetryPolicy,
    Send,
    StateSnapshot,
    StreamWriter,
    interrupt,
)
from tests.agents import AgentAction, AgentFinish
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


def test_invoke_two_processes_in_out_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
    mocker: MockerFixture,
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
        checkpointer=sync_checkpointer,
        interrupt_after_nodes=["one"],
    )
    thread1 = {"configurable": {"thread_id": "1"}}
    thread2 = {"configurable": {"thread_id": "2"}}

    # start execution, stop at inbox
    assert app.invoke(2, thread1, durability="async") is None

    # inbox == 3
    checkpoint = sync_checkpointer.get(thread1)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 3

    # resume execution, finish
    assert app.invoke(None, thread1, durability="async") == 4

    # start execution again, stop at inbox
    assert app.invoke(20, thread1, durability="async") is None

    # inbox == 21
    checkpoint = sync_checkpointer.get(thread1)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 21

    # send a new value in, interrupting the previous execution
    assert app.invoke(3, thread1, durability="async") is None
    assert app.invoke(None, thread1, durability="async") == 5

    # start execution again, stopping at inbox
    assert app.invoke(20, thread2, durability="async") is None

    # inbox == 21
    snapshot = app.get_state(thread2)
    assert snapshot.values["inbox"] == 21
    assert snapshot.next == ("two",)

    # update the state, resume
    app.update_state(thread2, 25, as_node="one")
    assert app.invoke(None, thread2) == 26

    # no pending tasks
    snapshot = app.get_state(thread2)
    assert snapshot.next == ()

    # list history
    history = [c for c in app.get_state_history(thread1)]
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

    # re-running from any previous checkpoint should re-run nodes
    assert [c for c in app.stream(None, history[0].config, stream_mode="updates")] == []
    assert [c for c in app.stream(None, history[1].config, stream_mode="updates")] == [
        {"two": {"output": 5}},
    ]
    assert [c for c in app.stream(None, history[2].config, stream_mode="updates")] == [
        {"one": {"inbox": 4}},
        {"__interrupt__": ()},
    ]


def test_fork_always_re_runs_nodes(
    sync_checkpointer: BaseCheckpointSaver, mocker: MockerFixture
) -> None:
    add_one = mocker.Mock(side_effect=lambda _: 1)

    builder = StateGraph(Annotated[int, operator.add])
    builder.add_node("add_one", add_one)
    builder.add_edge(START, "add_one")
    builder.add_conditional_edges("add_one", lambda cnt: "add_one" if cnt < 6 else END)
    graph = builder.compile(checkpointer=sync_checkpointer)

    thread1 = {"configurable": {"thread_id": "1"}}

    # start execution, stop at inbox
    assert [
        *graph.stream(1, thread1, stream_mode=["values", "updates"], durability="async")
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
    history = [c for c in graph.get_state_history(thread1)]
    assert history == [
        StateSnapshot(
            values=6,
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
        c for c in graph.stream(None, history[0].config, stream_mode="updates")
    ] == []
    assert [
        c for c in graph.stream(None, history[1].config, stream_mode="updates")
    ] == [
        {"add_one": 1},
    ]
    assert [
        c for c in graph.stream(None, history[2].config, stream_mode="updates")
    ] == [
        {"add_one": 1},
        {"add_one": 1},
    ]


def test_conditional_state_graph(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import tool

    class AgentState(TypedDict, total=False):
        input: Annotated[str, UntrackedValue]
        agent_outcome: AgentAction | AgentFinish | None
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    class ToolState(TypedDict, total=False):
        agent_outcome: AgentAction | AgentFinish

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
    def execute_tools(data: ToolState) -> dict:
        # check session in data
        assert "input" not in data
        assert "intermediate_steps" not in data
        # execute the tool
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        return {"intermediate_steps": [[agent_action, observation]]}

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        # check session in data
        # Logic to decide whether to continue in the loop or exit
        if isinstance(data["agent_outcome"], AgentFinish):
            return "exit"
        else:
            return "continue"

    # Define a new graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)
    workflow.add_node("tools", execute_tools, input_schema=ToolState)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    if isinstance(sync_checkpointer, InMemorySaver):
        assert json.dumps(app.get_input_jsonschema()) == snapshot
        assert json.dumps(app.get_output_jsonschema()) == snapshot
        assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke({"input": "what is weather in sf"}) == {
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

    assert [*app.stream({"input": "what is weather in sf"})] == [
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

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        for c in app_w_interrupt.stream(
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            )
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_before=["tools"],
        debug=True,
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c
        for c in app_w_interrupt.stream(
            {"input": "what is weather in sf"}, config, durability="exit"
        )
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            )
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    # test w interrupt before all
    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_before="*",
        debug=True,
    )
    config = {"configurable": {"thread_id": "3"}}
    llm.i = 0  # reset the llm

    assert [
        c
        for c in app_w_interrupt.stream(
            {"input": "what is weather in sf"}, config, durability="exit"
        )
    ] == [
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
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
        },
        parent_config=None,
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
                "thread_id": "3",
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
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
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
        },
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
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
            "step": 2,
        },
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    # test w interrupt after all
    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_after="*",
    )
    config = {"configurable": {"thread_id": "4"}}
    llm.i = 0  # reset the llm

    assert [
        c
        for c in app_w_interrupt.stream(
            {"input": "what is weather in sf"}, config, durability="exit"
        )
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
                "thread_id": "4",
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

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
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
        },
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
        config={
            "configurable": {
                "thread_id": "4",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "loop",
            "step": 2,
        },
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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


def test_prebuilt_tool_chat(snapshot: SnapshotAssertion) -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

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

    app = create_react_agent(model, tools)

    assert json.dumps(app.get_input_jsonschema()) == snapshot
    assert json.dumps(app.get_output_jsonschema()) == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke(
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
        for c in app.stream(
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

    assert app.invoke(
        {"messages": [HumanMessage(content="what is weather in sf")]},
        {"recursion_limit": 2},
        debug=True,
    ) == {
        "messages": [
            _AnyIdHumanMessage(content="what is weather in sf"),
            _AnyIdAIMessage(content="Sorry, need more steps to process this request."),
        ]
    }

    model.i = 0  # reset the model

    invoke_updates_events = app.invoke(
        {"messages": [HumanMessage(content="what is weather in sf")]},
        stream_mode="updates",
    )

    stream_updates_events = [
        *app.stream({"messages": [HumanMessage(content="what is weather in sf")]})
    ]

    for output in (invoke_updates_events, stream_updates_events):
        assert output[:3] == [
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
        assert output[3:5] == UnsortedSequence(
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
        assert output[5:] == [
            {"agent": {"messages": [_AnyIdAIMessage(content="answer")]}}
        ]


def test_state_graph_packets(
    sync_checkpointer: BaseCheckpointSaver, mocker: MockerFixture
) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolCall,
        ToolMessage,
    )
    from langchain_core.tools import tool

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        time.sleep(0.1)
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

    def agent(data: AgentState) -> AgentState:
        return {
            "messages": model.invoke(data["messages"]),
            "something_extra": "hi there",
        }

    # Define decision-making logic
    def should_continue(data: dict) -> str:
        assert data["something_extra"] == "hi there", (
            "nodes can pass extra data to their cond edges, which isn't saved in state"
        )
        # Logic to decide whether to continue in the loop or exit
        if tool_calls := data["messages"][-1].tool_calls:
            return [Send("tools", tool_call) for tool_call in tool_calls]
        else:
            return END

    def tools_node(input: ToolCall, config: RunnableConfig) -> AgentState:
        time.sleep(input["args"].get("idx", 0) / 10)
        output = tools_by_name[input["name"]].invoke(input["args"], config)
        return {
            "messages": ToolMessage(
                content=output, name=input["name"], tool_call_id=input["id"]
            )
        }

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", agent)
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

    assert app.invoke({"messages": HumanMessage(content="what is weather in sf")}) == {
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
        for c in app.stream(
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
        checkpointer=sync_checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        for c in app_w_interrupt.stream(
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
    last_message = (app_w_interrupt.get_state(config)).values["messages"][-1]
    last_message.tool_calls[0]["args"]["query"] = "a different query"
    app_w_interrupt.update_state(
        config, {"messages": last_message, "something_extra": "hi there"}
    )

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
            [*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
            "step": 4,
        },
        parent_config=(
            [*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config
        ),
        interrupts=(),
    )

    app_w_interrupt.update_state(
        config,
        {
            "messages": AIMessage(content="answer", id="ai2"),
            "something_extra": "hi there",
        },
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
            [*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config
        ),
        interrupts=(),
    )

    # interrupt before tools

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    model.i = 0

    assert [
        c
        for c in app_w_interrupt.stream(
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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

    # modify ai message
    last_message = (app_w_interrupt.get_state(config)).values["messages"][-1]
    last_message.tool_calls[0]["args"]["query"] = "a different query"
    app_w_interrupt.update_state(
        config, {"messages": last_message, "something_extra": "hi there"}
    )

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
        },
        parent_config=(
            [*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
            "step": 4,
        },
        parent_config=(
            [*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config
        ),
        interrupts=(),
    )

    app_w_interrupt.update_state(
        config,
        {
            "messages": AIMessage(content="answer", id="ai2"),
            "something_extra": "hi there",
        },
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
        parent_config=(
            [*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config
        ),
        interrupts=(),
    )


def test_message_graph(
    snapshot: SnapshotAssertion,
    deterministic_uuids: MockerFixture,
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    from copy import deepcopy

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool

    class FakeFunctionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = deepcopy(self.responses[self.i])
            if self.i < len(self.responses) - 1:
                self.i += 1
            else:
                self.i = 0
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])

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

    if isinstance(sync_checkpointer, InMemorySaver):
        assert json.dumps(app.get_input_jsonschema()) == snapshot
        assert json.dumps(app.get_output_jsonschema()) == snapshot
        assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke([HumanMessage(content="what is weather in sf")]) == [
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

    assert [*app.stream([HumanMessage(content="what is weather in sf")])] == [
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
        checkpointer=sync_checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        for c in app_w_interrupt.stream(
            ("human", "what is weather in sf"), config, durability="exit"
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    next_config = app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
        config=next_config,
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
        },
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
            "step": 4,
        },
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),  # replace existing message
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    model.i = 0  # reset the llm

    assert [
        c
        for c in app_w_interrupt.stream(
            "what is weather in sf", config, durability="exit"
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
            "step": 4,
        },
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
        ],
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
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    # add an extra message as if it came from "tools" node
    app_w_interrupt.update_state(config, ("ai", "an extra message"), as_node="tools")

    # extra message is coerced BaseMessge and appended
    # now the next node is "agent" per the graph edges
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
            _AnyIdAIMessage(content="an extra message"),
        ],
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
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
            "step": 6,
        },
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )


def test_root_graph(
    deterministic_uuids: MockerFixture,
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    from copy import deepcopy

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool

    class FakeFunctionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = deepcopy(self.responses[self.i])
            if self.i < len(self.responses) - 1:
                self.i += 1
            else:
                self.i = 0
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])

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

    class State(TypedDict):
        __root__: Annotated[list[BaseMessage], add_messages]

    # Define a new graph
    workflow = StateGraph(State)

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

    assert app.invoke(HumanMessage(content="what is weather in sf")) == [
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

    assert [*app.stream([HumanMessage(content="what is weather in sf")])] == [
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
                ToolMessage(
                    content="result for query",
                    name="search_api",
                    tool_call_id="tool_call123",
                    id="00000000-0000-4000-8000-000000000024",
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
                ToolMessage(
                    content="result for another",
                    name="search_api",
                    tool_call_id="tool_call456",
                    id="00000000-0000-4000-8000-000000000030",
                )
            ]
        },
        {"agent": AIMessage(content="answer", id="ai3")},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        for c in app_w_interrupt.stream(
            ("human", "what is weather in sf"), config, durability="exit"
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    next_config = app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
        config=next_config,
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
        },
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
                id=AnyStr(),
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
            "step": 4,
        },
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),  # replace existing message
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
        ],
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
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    model.i = 0  # reset the llm

    assert [
        c
        for c in app_w_interrupt.stream(
            "what is weather in sf", config, durability="exit"
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
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

    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
                id=AnyStr(),
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
            "step": 4,
        },
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    # add an extra message as if it came from "tools" node
    app_w_interrupt.update_state(config, ("ai", "an extra message"), as_node="tools")

    # extra message is coerced BaseMessge and appended
    # now the next node is "agent" per the graph edges
    assert app_w_interrupt.get_state(config) == StateSnapshot(
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
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
            _AnyIdAIMessage(content="an extra message"),
        ],
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
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
            "step": 6,
        },
        parent_config=(
            list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
        ),
        interrupts=(),
    )

    # create new graph with one more state key, reuse previous thread history

    def simple_add(left, right):
        if not isinstance(right, list):
            right = [right]
        return left + right

    class MoreState(TypedDict):
        __root__: Annotated[list[BaseMessage], simple_add]
        something_else: str

    # Define a new graph
    new_workflow = StateGraph(MoreState)
    new_workflow.add_node(
        "agent", RunnableMap(__root__=RunnablePick("__root__") | model)
    )
    new_workflow.add_node(
        "tools", RunnableMap(__root__=RunnablePick("__root__") | ToolNode(tools))
    )
    new_workflow.set_entry_point("agent")
    new_workflow.add_conditional_edges(
        "agent",
        RunnablePick("__root__") | should_continue,
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )
    new_workflow.add_edge("tools", "agent")
    new_app = new_workflow.compile(checkpointer=sync_checkpointer)
    model.i = 0  # reset the llm

    # previous state is converted to new schema
    assert new_app.get_state(config) == StateSnapshot(
        values={
            "__root__": [
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
                _AnyIdAIMessage(content="an extra message"),
            ]
        },
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
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
            "step": 6,
        },
        parent_config=(list(new_app.checkpointer.list(config, limit=2))[-1].config),
        interrupts=(),
    )

    # new input is merged to old state
    assert new_app.invoke(
        {
            "__root__": [HumanMessage(content="what is weather in la")],
            "something_else": "value",
        },
        config,
        interrupt_before=["agent"],
    ) == {
        "__root__": [
            HumanMessage(
                content="what is weather in sf",
                id="00000000-0000-4000-8000-000000000051",
            ),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "name": "search_api",
                        "args": {"query": "a different query"},
                        "id": "tool_call123",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            AIMessage(content="answer", id="ai2"),
            AIMessage(
                content="an extra message", id="00000000-0000-4000-8000-000000000066"
            ),
            HumanMessage(content="what is weather in la"),
        ],
        "something_else": "value",
    }


def test_in_one_fan_out_out_one_graph_state() -> None:
    def sorted_add(x: list[str], y: list[str]) -> list[str]:
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    def retriever_one(data: State) -> State:
        # timer ensures stream output order is stable
        # also, it confirms that the update order is not dependent on finishing order
        # instead being defined by the order of the nodes/edges in the graph definition
        # ie. stable between invocations
        time.sleep(0.1)
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
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    assert [*app.stream({"query": "what is weather in sf"}, stream_mode="values")] == [
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
        *app.stream(
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


def test_dynamic_interrupt(sync_checkpointer: BaseCheckpointSaver) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_node_count = 0

    def tool_two_node(s: State) -> State:
        nonlocal tool_two_node_count
        tool_two_node_count += 1
        if s["market"] == "DE":
            answer = interrupt("Just because...")
        else:
            answer = " all good"
        return {"my_key": answer}

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two", tool_two_node, retry_policy=RetryPolicy())
    tool_two_graph.add_edge(START, "tool_two")
    tool_two = tool_two_graph.compile()

    tracer = FakeTracer()
    assert tool_two.invoke(
        {"my_key": "value", "market": "DE"}, {"callbacks": [tracer]}
    ) == {
        "my_key": "value",
        "market": "DE",
        "__interrupt__": [Interrupt(value="Just because...", id=AnyStr())],
    }
    assert tool_two_node_count == 1, "interrupts aren't retried"
    assert len(tracer.runs) == 1
    run = tracer.runs[0]
    assert run.end_time is not None
    assert run.error is None
    assert run.outputs == {"market": "DE", "my_key": "value"}

    assert tool_two.invoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value all good",
        "market": "US",
    }

    tool_two = tool_two_graph.compile(checkpointer=sync_checkpointer)

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        tool_two.invoke({"my_key": "value", "market": "DE"})

    # flow: interrupt -> resume with answer
    thread2 = {"configurable": {"thread_id": "2"}}
    # stop when about to enter node
    assert [
        c for c in tool_two.stream({"my_key": "value ⛰️", "market": "DE"}, thread2)
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    id=AnyStr(),
                ),
            )
        },
    ]
    # resume with answer
    assert [c for c in tool_two.stream(Command(resume=" my answer"), thread2)] == [
        {"tool_two": {"my_key": " my answer"}},
    ]

    # flow: interrupt -> clear tasks
    thread1 = {"configurable": {"thread_id": "1"}}
    # stop when about to enter node
    assert tool_two.invoke(
        {"my_key": "value ⛰️", "market": "DE"}, thread1, durability="exit"
    ) == {
        "my_key": "value ⛰️",
        "market": "DE",
        "__interrupt__": [Interrupt(value="Just because...", id=AnyStr())],
    }

    assert [c.metadata for c in tool_two.checkpointer.list(thread1)] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
        },
    ]

    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️", "market": "DE"},
        next=("tool_two",),
        tasks=(
            PregelTask(
                AnyStr(),
                "tool_two",
                (PULL, "tool_two"),
                interrupts=(
                    Interrupt(
                        value="Just because...",
                        id=AnyStr(),
                    ),
                ),
            ),
        ),
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
            "step": 0,
        },
        parent_config=None,
        interrupts=(
            Interrupt(
                value="Just because...",
                id=AnyStr(),
            ),
        ),
    )
    # clear the interrupt and next tasks
    tool_two.update_state(thread1, None, as_node=END)
    # interrupt and next tasks are cleared
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️", "market": "DE"},
        next=(),
        tasks=(),
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
        },
        parent_config=(list(tool_two.checkpointer.list(thread1, limit=2))[-1].config),
        interrupts=(),
    )


def test_partial_pending_checkpoint(sync_checkpointer: BaseCheckpointSaver) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    def tool_one(s: State) -> State:
        return {"my_key": " one"}

    tool_two_node_count = 0

    def tool_two_node(s: State) -> State:
        nonlocal tool_two_node_count
        tool_two_node_count += 1
        if s["market"] == "DE":
            time.sleep(0.1)
            answer = interrupt("Just because...")
        else:
            answer = " all good"
        return {"my_key": answer}

    def start(state: State) -> list[Send | str]:
        return ["tool_two", Send("tool_one", state)]

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two", tool_two_node, retry_policy=RetryPolicy())
    tool_two_graph.add_node("tool_one", tool_one)
    tool_two_graph.set_conditional_entry_point(start)
    tool_two = tool_two_graph.compile()

    tracer = FakeTracer()
    assert tool_two.invoke(
        {"my_key": "value", "market": "DE"}, {"callbacks": [tracer]}
    ) == {
        "my_key": "value one",
        "market": "DE",
        "__interrupt__": [Interrupt(value="Just because...", id=AnyStr())],
    }
    assert tool_two_node_count == 1, "interrupts aren't retried"
    assert len(tracer.runs) == 1
    run = tracer.runs[0]
    assert run.end_time is not None
    assert run.error is None
    assert run.outputs == {"market": "DE", "my_key": "value one"}

    assert tool_two.invoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value all good one",
        "market": "US",
    }

    tool_two = tool_two_graph.compile(checkpointer=sync_checkpointer)

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        tool_two.invoke({"my_key": "value", "market": "DE"})

    # flow: interrupt -> resume with answer
    thread2 = {"configurable": {"thread_id": "2"}}
    # stop when about to enter node
    assert [
        c for c in tool_two.stream({"my_key": "value ⛰️", "market": "DE"}, thread2)
    ] == [
        {
            "tool_one": {"my_key": " one"},
        },
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    id=AnyStr(),
                ),
            )
        },
    ]
    # resume with answer
    assert [c for c in tool_two.stream(Command(resume=" my answer"), thread2)] == [
        {"tool_one": {"my_key": " one"}, "__metadata__": {"cached": True}},
        {"tool_two": {"my_key": " my answer"}},
    ]

    # flow: interrupt -> clear tasks
    thread1 = {"configurable": {"thread_id": "1"}}
    # stop when about to enter node
    assert tool_two.invoke(
        {"my_key": "value ⛰️", "market": "DE"}, thread1, durability="exit"
    ) == {
        "my_key": "value ⛰️ one",
        "market": "DE",
        "__interrupt__": [Interrupt(value="Just because...", id=AnyStr())],
    }
    assert [c.metadata for c in tool_two.checkpointer.list(thread1)] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
        },
    ]

    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️ one", "market": "DE"},
        next=("tool_two",),
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="tool_one",
                path=("__pregel_push", 0, False),
                result={"my_key": " one"},
            ),
            PregelTask(
                AnyStr(),
                "tool_two",
                (PULL, "tool_two"),
                interrupts=(
                    Interrupt(
                        value="Just because...",
                        id=AnyStr(),
                    ),
                ),
            ),
        ),
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
            "step": 0,
        },
        parent_config=None,
        interrupts=(
            Interrupt(
                value="Just because...",
                id=AnyStr(),
            ),
        ),
    )

    # clear the interrupt and next tasks
    tool_two.update_state(thread1, None, as_node=END)

    # interrupt and unresolved tasks are cleared, finished tasks are kept
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️ one", "market": "DE"},
        next=(),
        tasks=(),
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
        },
        parent_config=([*tool_two.checkpointer.list(thread1, limit=2)][-1].config),
        interrupts=(),
    )


def test_dynamic_interrupt_subgraph(sync_checkpointer: BaseCheckpointSaver) -> None:
    class SubgraphState(TypedDict):
        my_key: str
        market: str

    tool_two_node_count = 0

    def tool_two_node(s: SubgraphState) -> SubgraphState:
        nonlocal tool_two_node_count
        tool_two_node_count += 1
        if s["market"] == "DE":
            answer = interrupt("Just because...")
        else:
            answer = " all good"
        return {"my_key": answer}

    subgraph = StateGraph(SubgraphState)
    subgraph.add_node("do", tool_two_node, retry_policy=RetryPolicy())
    subgraph.add_edge(START, "do")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two", subgraph.compile())
    tool_two_graph.add_edge(START, "tool_two")
    tool_two = tool_two_graph.compile()

    tracer = FakeTracer()
    assert tool_two.invoke(
        {"my_key": "value", "market": "DE"}, {"callbacks": [tracer]}
    ) == {
        "my_key": "value",
        "market": "DE",
        "__interrupt__": [
            Interrupt(
                value="Just because...",
                id=AnyStr(),
            )
        ],
    }
    assert tool_two_node_count == 1, "interrupts aren't retried"
    assert len(tracer.runs) == 1
    run = tracer.runs[0]
    assert run.end_time is not None
    assert run.error is None
    assert run.outputs == {"market": "DE", "my_key": "value"}

    assert tool_two.invoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value all good",
        "market": "US",
    }

    tool_two = tool_two_graph.compile(checkpointer=sync_checkpointer)

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        tool_two.invoke({"my_key": "value", "market": "DE"})

    # flow: interrupt -> resume with answer
    thread2 = {"configurable": {"thread_id": "2"}}
    # stop when about to enter node
    assert [
        c for c in tool_two.stream({"my_key": "value ⛰️", "market": "DE"}, thread2)
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    id=AnyStr(),
                ),
            )
        },
    ]
    # resume with answer
    assert [c for c in tool_two.stream(Command(resume=" my answer"), thread2)] == [
        {"tool_two": {"my_key": " my answer", "market": "DE"}},
    ]

    # flow: interrupt -> clear tasks
    thread1 = {"configurable": {"thread_id": "1"}}
    # stop when about to enter node
    assert tool_two.invoke(
        {"my_key": "value ⛰️", "market": "DE"}, thread1, durability="exit"
    ) == {
        "my_key": "value ⛰️",
        "market": "DE",
        "__interrupt__": [
            Interrupt(
                value="Just because...",
                id=AnyStr(),
            )
        ],
    }

    assert [
        c.metadata
        for c in tool_two.checkpointer.list(
            {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
        )
    ] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
        },
    ]

    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️", "market": "DE"},
        next=("tool_two",),
        tasks=(
            PregelTask(
                AnyStr(),
                "tool_two",
                (PULL, "tool_two"),
                interrupts=(
                    Interrupt(
                        value="Just because...",
                        id=AnyStr(),
                    ),
                ),
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("tool_two:"),
                    }
                },
            ),
        ),
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
        },
        parent_config=None,
        interrupts=(
            Interrupt(
                value="Just because...",
                id=AnyStr(),
            ),
        ),
    )
    # clear the interrupt and next tasks
    tool_two.update_state(thread1, None, as_node=END)
    # interrupt and next tasks are cleared
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️", "market": "DE"},
        next=(),
        tasks=(),
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
            "source": "update",
            "step": 1,
        },
        parent_config=(
            list(
                tool_two.checkpointer.list(
                    {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}, limit=2
                )
            )[-1].config
        ),
        interrupts=(),
    )


def test_send_dedupe_on_resume(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    class InterruptOnce:
        ticks: int = 0

        def __call__(self, state):
            self.ticks += 1
            if self.ticks == 1:
                interrupt("Bahh")
            return ["|".join(("flaky", str(state)))]

    class Node:
        def __init__(self, name: str):
            self.name = name
            self.ticks = 0
            setattr(self, "__name__", name)

        def __call__(self, state):
            time.sleep(0)
            # sleep makes it more likely to trigger edge case where 1st task
            # finishes before 2nd is registered in futures dict
            self.ticks += 1
            update = (
                [self.name]
                if isinstance(state, list)
                else ["|".join((self.name, str(state)))]
            )
            if isinstance(state, Command):
                return replace(state, update=update)
            else:
                return update

    def send_for_fun(state):
        return [
            Send("2", Command(goto=Send("2", 3))),
            Send("2", Command(goto=Send("flaky", 4))),
            "3.1",
        ]

    def route_to_three(state) -> Literal["3"]:
        return "3"

    builder = StateGraph(Annotated[list, operator.add])
    builder.add_node(Node("1"))
    builder.add_node(Node("2"))
    builder.add_node(Node("3"))
    builder.add_node(Node("3.1"))
    builder.add_node("flaky", InterruptOnce())
    builder.add_edge(START, "1")
    builder.add_conditional_edges("1", send_for_fun)
    builder.add_conditional_edges("2", route_to_three)

    graph = builder.compile(checkpointer=sync_checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}
    assert graph.invoke(["0"], thread1, durability=durability) == {
        "__interrupt__": [
            Interrupt(
                value="Bahh",
                id=AnyStr(),
            ),
        ],
    }
    assert builder.nodes["2"].runnable.func.ticks == 3
    assert builder.nodes["flaky"].runnable.func.ticks == 1
    # check state
    state = graph.get_state(thread1)
    assert state.next == ("flaky",)
    # check history
    history = [c for c in graph.get_state_history(thread1)]
    assert len(history) == (4 if durability != "exit" else 1)

    # resume execution
    assert graph.invoke(None, thread1, durability=durability) == [
        "0",
        "1",
        "3.1",
        "2|Command(goto=Send(node='2', arg=3))",
        "2|Command(goto=Send(node='flaky', arg=4))",
        "3",
        "2|3",
        "flaky|4",
        "3",
    ]
    # node "2" doesn't get called again, as we recover writes saved before
    assert builder.nodes["2"].runnable.func.ticks == 3
    # node "flaky" gets called again, as it was interrupted
    assert builder.nodes["flaky"].runnable.func.ticks == 2
    # check state
    state = graph.get_state(thread1)
    assert state.next == ()
    # check history
    history = [c for c in graph.get_state_history(thread1)]
    assert len(history) == (6 if durability != "exit" else 2)
    expected_history = [
        StateSnapshot(
            values=[
                "0",
                "1",
                "3.1",
                "2|Command(goto=Send(node='2', arg=3))",
                "2|Command(goto=Send(node='flaky', arg=4))",
                "3",
                "2|3",
                "flaky|4",
                "3",
            ],
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "step": 4,
                "parents": {},
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(),
            interrupts=(),
        ),
        StateSnapshot(
            values=[
                "0",
                "1",
                "3.1",
                "2|Command(goto=Send(node='2', arg=3))",
                "2|Command(goto=Send(node='flaky', arg=4))",
                "3",
                "2|3",
                "flaky|4",
            ],
            next=("3",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "step": 3,
                "parents": {},
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
                    id=AnyStr(),
                    name="3",
                    path=("__pregel_pull", "3"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["3"],
                ),
            ),
            interrupts=(),
        ),
        StateSnapshot(
            values=[
                "0",
                "1",
                "3.1",
                "2|Command(goto=Send(node='2', arg=3))",
                "2|Command(goto=Send(node='flaky', arg=4))",
            ],
            next=("2", "flaky", "3"),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "step": 2,
                "parents": {},
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
                    id=AnyStr(),
                    name="2",
                    path=("__pregel_push", 0, False),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["2|3"],
                ),
                PregelTask(
                    id=AnyStr(),
                    name="flaky",
                    path=("__pregel_push", 1, False),
                    error=None,
                    interrupts=(Interrupt(value="Bahh", id=AnyStr()),),
                    state=None,
                    result=["flaky|4"] if durability != "exit" else None,
                ),
                PregelTask(
                    id=AnyStr(),
                    name="3",
                    path=("__pregel_pull", "3"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["3"],
                ),
            ),
            interrupts=(Interrupt(value="Bahh", id=AnyStr()),),
        ),
        StateSnapshot(
            values=["0", "1"],
            next=("2", "2", "3.1"),
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
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="2",
                    path=("__pregel_push", 0, False),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["2|Command(goto=Send(node='2', arg=3))"],
                ),
                PregelTask(
                    id=AnyStr(),
                    name="2",
                    path=("__pregel_push", 1, False),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["2|Command(goto=Send(node='flaky', arg=4))"],
                ),
                PregelTask(
                    id=AnyStr(),
                    name="3.1",
                    path=("__pregel_pull", "3.1"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["3.1"],
                ),
            ),
            interrupts=(),
        ),
        StateSnapshot(
            values=["0"],
            next=("1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "step": 0,
                "parents": {},
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
                    id=AnyStr(),
                    name="1",
                    path=("__pregel_pull", "1"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["1"],
                ),
            ),
            interrupts=(),
        ),
        StateSnapshot(
            values=[],
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "step": -1,
                "parents": {},
            },
            created_at=AnyStr(),
            parent_config=None,
            interrupts=(),
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="__start__",
                    path=("__pregel_pull", "__start__"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["0"],
                ),
            ),
        ),
    ]
    if durability != "exit":
        assert history == expected_history
    else:
        assert history[0] == expected_history[0]._replace(
            parent_config=history[1].config
        )
        assert history[1] == expected_history[2]._replace(parent_config=None)


def test_nested_graph_state(sync_checkpointer: BaseCheckpointSaver) -> None:
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

    app = graph.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    app.invoke({"my_key": "my value"}, config, durability="exit")
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
    assert app.get_state(config) == expected
    assert list(app.get_state_history(config)) == [expected]
    # now, get_state with subgraphs state
    assert app.get_state(config, subgraphs=True) == StateSnapshot(
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

    # get_state_history for a subgraph returns its checkpoints
    child_history = [*app.get_state_history(app.get_state(config).tasks[0].state)]
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
            interrupts=(),
            tasks=(PregelTask(AnyStr(), "inner_2", (PULL, "inner_2")),),
        ),
    ]
    assert child_history == expected_child_history

    # resume
    app.invoke(None, config, durability="exit")
    # test state w/ nested subgraph state (after resuming from interrupt)
    assert app.get_state(config) == StateSnapshot(
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
    actual_history = list(app.get_state_history(config))
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
                        "configurable": {"thread_id": "1", "checkpoint_ns": AnyStr()}
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
        assert app.get_state(actual_snapshot.config) == expected_snapshot


def test_doubly_nested_graph_state(
    sync_checkpointer: BaseCheckpointSaver,
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

    app = graph.compile(checkpointer=sync_checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert [
        c
        for c in app.stream(
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
    outer_state = app.get_state(config)
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
    child_state = app.get_state(outer_state.tasks[0].state)
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
    grandchild_state = app.get_state(child_state.tasks[0].state)
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
    assert app.get_state(config, subgraphs=True) == StateSnapshot(
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
    # # resume
    assert [c for c in app.stream(None, config, subgraphs=True, durability="exit")] == [
        (
            (AnyStr("child:"), AnyStr("child_1:")),
            {"grandchild_2": {"my_key": "hi my value here and there"}},
        ),
        ((AnyStr("child:"),), {"child_1": {"my_key": "hi my value here and there"}}),
        ((), {"child": {"my_key": "hi my value here and there"}}),
        ((), {"parent_2": {"my_key": "hi my value here and there and back again"}}),
    ]
    # get state with and without subgraphs
    assert (
        app.get_state(config)
        == app.get_state(config, subgraphs=True)
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
    outer_history = list(app.get_state_history(config))
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
                    result=None,
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
    child_history = list(app.get_state_history(outer_history[1].tasks[0].state))
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
    grandchild_history = list(app.get_state_history(child_history[0].tasks[0].state))
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


def test_send_to_nested_graphs(sync_checkpointer: BaseCheckpointSaver) -> None:
    class OverallState(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]

    def continue_to_jokes(state: OverallState):
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

    class JokeState(TypedDict):
        subject: str

    def edit(state: JokeState):
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

    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    tracer = FakeTracer()

    # invoke and pause at nested interrupt
    assert graph.invoke(
        {"subjects": ["cats", "dogs"]}, config={**config, "callbacks": [tracer]}
    ) == {
        "subjects": ["cats", "dogs"],
        "jokes": [],
    }
    assert len(tracer.runs) == 1, "Should produce exactly 1 root run"

    # check state
    outer_state = graph.get_state(config)

    # update state of dogs joke graph
    graph.update_state(outer_state.tasks[1].state, {"subject": "turtles - hohoho"})

    # continue past interrupt
    assert sorted(
        graph.stream(None, config=config),
        key=lambda d: d["generate_joke"]["jokes"][0],
    ) == [
        {"generate_joke": {"jokes": ["Joke about cats - hohoho"]}},
        {"generate_joke": {"jokes": ["Joke about turtles - hohoho"]}},
    ]


def test_send_react_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

    ai_message = AIMessage(
        "",
        id="ai1",
        tool_calls=[ToolCall(name="foo", args={"hi": [1, 2, 3]}, id=AnyStr())],
    )

    def agent(state):
        return {"messages": ai_message}

    def route(state):
        if isinstance(state["messages"][-1], AIMessage):
            return [
                Send(call["name"], call) for call in state["messages"][-1].tool_calls
            ]

    foo_called = 0

    def foo(call: ToolCall):
        nonlocal foo_called
        foo_called += 1
        return {"messages": ToolMessage(str(call["args"]), tool_call_id=call["id"])}

    builder = StateGraph(MessagesState)
    builder.add_node(agent)
    builder.add_node(foo)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route)
    graph = builder.compile()

    assert graph.invoke({"messages": [HumanMessage("hello")]}) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="{'hi': [1, 2, 3]}",
                tool_call_id=AnyStr(),
            ),
        ]
    }
    assert foo_called == 1

    # simple interrupt-resume flow
    foo_called = 0
    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "1"}}
    assert graph.invoke({"messages": [HumanMessage("hello")]}, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert foo_called == 0
    assert graph.invoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="{'hi': [1, 2, 3]}",
                tool_call_id=AnyStr(),
            ),
        ]
    }
    assert foo_called == 1

    # interrupt-update-resume flow
    foo_called = 0
    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "2"}}
    assert graph.invoke(
        {"messages": [HumanMessage("hello")]}, thread1, durability="exit"
    ) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert foo_called == 0

    # get state should show the pending task
    state = graph.get_state(thread1)
    assert state == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        },
        next=("foo",),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 1,
            "source": "loop",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        interrupts=(),
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", 0, False),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
    )

    # remove the tool call, clearing the pending task
    graph.update_state(
        thread1, {"messages": AIMessage("Bye now", id=ai_message.id, tool_calls=[])}
    )

    # tool call no longer in pending tasks
    assert graph.get_state(thread1) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="Bye now",
                    tool_calls=[],
                ),
            ]
        },
        next=(),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 2,
            "source": "update",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=(
            {
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            }
        ),
        interrupts=(),
        tasks=(),
    )

    # tool call not executed
    assert graph.invoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(content="Bye now"),
        ]
    }
    assert foo_called == 0

    # interrupt-update-resume flow, creating new Send in update call
    foo_called = 0
    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "3"}}
    assert graph.invoke(
        {"messages": [HumanMessage("hello")]}, thread1, durability="exit"
    ) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert foo_called == 0

    # get state should show the pending task
    state = graph.get_state(thread1)
    assert state == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        },
        next=("foo",),
        config={
            "configurable": {
                "thread_id": "3",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 1,
            "source": "loop",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        interrupts=(),
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", 0, False),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
    )

    # replace the tool call, should clear previous send, create new one
    graph.update_state(
        thread1,
        {
            "messages": AIMessage(
                "",
                id=ai_message.id,
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [4, 5, 6]},
                        "id": "tool1",
                        "type": "tool_call",
                    }
                ],
            )
        },
    )

    # prev tool call no longer in pending tasks, new tool call is
    assert graph.get_state(thread1) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [4, 5, 6]},
                            "id": "tool1",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        },
        next=("foo",),
        config={
            "configurable": {
                "thread_id": "3",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 2,
            "source": "update",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=(
            {
                "configurable": {
                    "thread_id": "3",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            }
        ),
        interrupts=(),
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", 0, False),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
    )

    # prev tool call not executed, new tool call is
    assert graph.invoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            AIMessage(
                "",
                id="ai1",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [4, 5, 6]},
                        "id": "tool1",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(content="{'hi': [4, 5, 6]}", tool_call_id="tool1"),
        ]
    }
    assert foo_called == 1


def test_send_react_interrupt_control(
    sync_checkpointer: BaseCheckpointSaver, snapshot: SnapshotAssertion
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

    ai_message = AIMessage(
        "",
        id="ai1",
        tool_calls=[ToolCall(name="foo", args={"hi": [1, 2, 3]}, id=AnyStr())],
    )

    def agent(state) -> Command[Literal["foo"]]:
        return Command(
            update={"messages": ai_message},
            goto=[Send(call["name"], call) for call in ai_message.tool_calls],
        )

    foo_called = 0

    def foo(call: ToolCall):
        nonlocal foo_called
        foo_called += 1
        return {"messages": ToolMessage(str(call["args"]), tool_call_id=call["id"])}

    builder = StateGraph(MessagesState)
    builder.add_node(agent)
    builder.add_node(foo)
    builder.add_edge(START, "agent")
    graph = builder.compile()

    if isinstance(sync_checkpointer, InMemorySaver):
        assert graph.get_graph().draw_mermaid() == snapshot

    assert graph.invoke({"messages": [HumanMessage("hello")]}) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="{'hi': [1, 2, 3]}",
                tool_call_id=AnyStr(),
            ),
        ]
    }
    assert foo_called == 1

    # simple interrupt-resume flow
    foo_called = 0
    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "1"}}
    assert graph.invoke({"messages": [HumanMessage("hello")]}, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert foo_called == 0
    assert graph.invoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="{'hi': [1, 2, 3]}",
                tool_call_id=AnyStr(),
            ),
        ]
    }
    assert foo_called == 1

    # interrupt-update-resume flow
    foo_called = 0
    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "2"}}
    assert graph.invoke(
        {"messages": [HumanMessage("hello")]}, thread1, durability="exit"
    ) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert foo_called == 0

    # get state should show the pending task
    state = graph.get_state(thread1)
    assert state == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        },
        next=("foo",),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 1,
            "source": "loop",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", 0, False),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
        interrupts=(),
    )

    # remove the tool call, clearing the pending task
    graph.update_state(
        thread1, {"messages": AIMessage("Bye now", id=ai_message.id, tool_calls=[])}
    )

    # tool call no longer in pending tasks
    assert graph.get_state(thread1) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="Bye now",
                    tool_calls=[],
                ),
            ]
        },
        next=(),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 2,
            "source": "update",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=(
            {
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            }
        ),
        interrupts=(),
        tasks=(),
    )

    # tool call not executed
    assert graph.invoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(content="Bye now"),
        ]
    }
    assert foo_called == 0

    # interrupt-update-resume flow, creating new Send in update call

    # TODO add here test with invoke(Command())


def test_weather_subgraph(
    sync_checkpointer: BaseCheckpointSaver, snapshot: SnapshotAssertion
) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )

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
    graph = graph.compile(checkpointer=sync_checkpointer)

    if isinstance(sync_checkpointer, InMemorySaver):
        assert graph.get_graph(xray=1).draw_mermaid() == snapshot

    config = {"configurable": {"thread_id": "1"}}
    thread2 = {"configurable": {"thread_id": "2"}}
    inputs = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}

    # run with custom output
    assert [
        c for c in graph.stream(inputs, thread2, stream_mode="custom", subgraphs=True)
    ] == [
        ((), "I'm"),
        ((AnyStr("weather_graph:"),), " very"),
    ]
    assert [
        c for c in graph.stream(None, thread2, stream_mode="custom", subgraphs=True)
    ] == [
        ((AnyStr("weather_graph:"),), " good"),
    ]

    # run until interrupt
    assert [
        c
        for c in graph.stream(
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
    state = graph.get_state(config)
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

    # update
    graph.update_state(state.tasks[0].state, {"city": "la"})

    # run after update
    assert [
        c
        for c in graph.stream(
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
        for c in graph.stream(
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
    state = graph.get_state(config, subgraphs=True)
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
                    tasks=(
                        PregelTask(
                            id=AnyStr(),
                            name="weather_node",
                            path=(PULL, "weather_node"),
                        ),
                    ),
                    interrupts=(),
                ),
            ),
        ),
        interrupts=(),
    )
    graph.update_state(
        state.tasks[0].state.config,
        {"messages": [{"role": "assistant", "content": "rainy"}]},
        as_node="weather_node",
    )
    state = graph.get_state(config, subgraphs=True)
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
                    interrupts=(),
                    tasks=(),
                ),
            ),
        ),
        interrupts=(),
    )
    assert [
        c
        for c in graph.stream(
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
        for c in graph.stream(
            inputs, {"configurable": {"thread_id": "3"}}, stream_mode="custom"
        )
    ] == [
        "I'm",
    ]

    # run with messages output, with subgraph streaming, should inc subgraph messages
    assert [
        c
        for c in graph.stream(
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
        for c in graph.stream(
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


def test_subgraph_to_end_does_not_warn() -> None:
    """Regression test for https://github.com/langchain-ai/langgraph/issues/5572."""

    class State(TypedDict):
        x: str

    def update_x(state: State):
        return Command(goto=END, update={"x": state["x"] + "!"})

    # Subgraph
    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node("update_x", update_x)
    subgraph_builder.add_edge(START, "update_x")
    subgraph_builder.add_edge("update_x", END)
    subgraph = subgraph_builder.compile()

    # Parent graph
    builder = StateGraph(State)
    builder.add_node("subgraph_node", subgraph)
    builder.add_edge(START, "subgraph_node")
    builder.add_edge("subgraph_node", END)
    graph = builder.compile()

    response = graph.invoke({"x": "hello"})
    assert response == {"x": "hello!"}
