"""Tests for keyed subgraph instances (`configurable["subgraph_key"]`).

A key gives a subgraph invocation an addressable checkpoint namespace
(`<parent frames>|:key`) instead of one derived from the invoking task id or
the call-order counter. Parallel invocations with distinct keys never share
state, and a later invocation with the same key continues the same history.
"""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import ToolNode, ToolRuntime
from typing_extensions import TypedDict

from langgraph.graph import START, MessagesState, StateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command, interrupt

pytestmark = pytest.mark.anyio


class ParentState(TypedDict):
    topic: str
    result: str


def keyed(key: str) -> RunnableConfig:
    return {"configurable": {"subgraph_key": key}}


def texts(result: dict) -> list[str]:
    return [m.text for m in result["messages"]]


def make_agent(
    name: str, *, stateful: bool = False, interrupt_on: str | None = None
) -> Pregel:
    def reply(state: MessagesState) -> dict:
        q = state["messages"][-1].text
        if interrupt_on and interrupt_on in q:
            ans = interrupt(f"{name} needs approval for {q}")
            return {"messages": [AIMessage(content=f"{name}({ans}): {q}")]}
        return {"messages": [AIMessage(content=f"{name}: {q}")]}

    builder = (
        StateGraph(MessagesState).add_node("reply", reply).add_edge(START, "reply")
    )
    return builder.compile(checkpointer=True) if stateful else builder.compile()


def make_parent(node, saver: BaseCheckpointSaver) -> Pregel:
    return (
        StateGraph(ParentState)
        .add_node("call", node)
        .add_edge(START, "call")
        .compile(checkpointer=saver)
    )


def namespaces(saver: BaseCheckpointSaver, config: RunnableConfig) -> list[str]:
    return sorted(
        {
            c.config["configurable"]["checkpoint_ns"]
            for c in saver.list(
                {"configurable": {"thread_id": config["configurable"]["thread_id"]}}
            )
        }
    )


@pytest.mark.parametrize("stateful", [False, True])
def test_parallel_keyed_calls_then_continue_one(
    sync_checkpointer: BaseCheckpointSaver, stateful: bool
) -> None:
    """Two parallel calls to the same subgraph with distinct keys get distinct
    namespaces; a later call with one of the keys continues that history.
    Identical behaviour for per-invocation and checkpointer=True subgraphs."""
    fruit = make_agent("fruit", stateful=stateful)
    turn = 0

    def call(state: ParentState, config: RunnableConfig) -> dict:
        nonlocal turn
        turn += 1
        if turn == 1:
            with get_executor_for_config(config) as ex:
                futs = {
                    k: ex.submit(
                        fruit.invoke, {"messages": [HumanMessage(content=q)]}, keyed(k)
                    )
                    for k, q in (("tc_a", "apples"), ("tc_b", "bananas"))
                }
                outs = {k: texts(f.result()) for k, f in futs.items()}
            return {"result": repr(outs)}
        r = fruit.invoke(
            {"messages": [HumanMessage(content=state["topic"])]}, keyed("tc_b")
        )
        return {"result": repr(texts(r))}

    parent = make_parent(call, sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    out = parent.invoke({"topic": "", "result": ""}, config)
    assert out["result"] == repr(
        {"tc_a": ["apples", "fruit: apples"], "tc_b": ["bananas", "fruit: bananas"]}
    )
    out = parent.invoke({"topic": "more bananas", "result": ""}, config)
    assert out["result"] == repr(
        ["bananas", "fruit: bananas", "more bananas", "fruit: more bananas"]
    )
    assert namespaces(sync_checkpointer, config) == ["", "call|:tc_a", "call|:tc_b"]
    # keyed instances are addressable through the parent by explicit namespace
    child = parent.get_state(
        {"configurable": {**config["configurable"], "checkpoint_ns": "call|:tc_b"}}
    )
    assert child.config["configurable"]["checkpoint_ns"] == "call|:tc_b"
    assert texts(child.values) == [
        "bananas",
        "fruit: bananas",
        "more bananas",
        "fruit: more bananas",
    ]


def test_keys_make_parallel_resume_deterministic(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Without keys, two subgraphs invoked in parallel inside one node are told
    apart by a call-order counter, which is racy: if the arrival order flips
    on resume, each subgraph loads the other's checkpoint. Keys fix this."""
    a = make_agent("A")
    b = make_agent("B", interrupt_on="hello-B")
    attempts = 0

    def call(state: ParentState, config: RunnableConfig) -> dict:
        nonlocal attempts
        attempts += 1
        first_is_a = attempts == 1

        def run(graph: Pregel, key: str, text: str, delay: float) -> dict:
            time.sleep(delay)
            return graph.invoke({"messages": [HumanMessage(content=text)]}, keyed(key))

        with get_executor_for_config(config) as ex:
            fa = ex.submit(run, a, "tc_a", "hello-A", 0.0 if first_is_a else 0.2)
            fb = ex.submit(run, b, "tc_b", "hello-B", 0.2 if first_is_a else 0.0)
            ra, rb = fa.result(), fb.result()
        return {"result": f"{texts(ra)}|{texts(rb)}"}

    parent = make_parent(call, sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    out = parent.invoke({"topic": "", "result": ""}, config)
    assert [i.value for i in out["__interrupt__"]] == ["B needs approval for hello-B"]
    out = parent.invoke(Command(resume="yes"), config)
    assert out["result"] == "['hello-A', 'A: hello-A']|['hello-B', 'B(yes): hello-B']"


def test_stateful_subgraph_new_turn_inside_resumed_task(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """A checkpointer=True subgraph invoked for a *new turn* from a task that is
    being resumed (parent interrupted before the call) must apply its input
    instead of inheriting the parent's resume flag and returning stale state."""
    fruit = make_agent("fruit", stateful=True)

    def call(state: ParentState) -> dict:
        ok = interrupt(f"ask about {state['topic']}?")
        r = fruit.invoke({"messages": [HumanMessage(content=state["topic"])]})
        return {"result": f"({ok}) {texts(r)}"}

    parent = make_parent(call, sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    parent.invoke({"topic": "apples", "result": ""}, config)
    assert parent.invoke(Command(resume="yes"), config)["result"] == (
        "(yes) ['apples', 'fruit: apples']"
    )
    parent.invoke({"topic": "bananas", "result": ""}, config)
    assert parent.invoke(Command(resume="yes"), config)["result"] == (
        "(yes) ['apples', 'fruit: apples', 'bananas', 'fruit: bananas']"
    )


def test_stateful_subgraph_turn_not_replayed_on_resume(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """The opposite case: subgraph call *then* interrupt in the same task. On
    resume the task re-runs; the subgraph must resume (return its completed
    turn) rather than apply the input again and duplicate the turn."""
    fruit = make_agent("fruit", stateful=True)

    def call(state: ParentState) -> dict:
        r = fruit.invoke({"messages": [HumanMessage(content=state["topic"])]})
        ok = interrupt("continue?")
        return {"result": f"({ok}) {texts(r)}"}

    parent = make_parent(call, sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    parent.invoke({"topic": "apples", "result": ""}, config)
    parent.invoke(Command(resume="yes"), config)
    parent.invoke({"topic": "bananas", "result": ""}, config)
    assert parent.invoke(Command(resume="yes"), config)["result"] == (
        "(yes) ['apples', 'fruit: apples', 'bananas', 'fruit: bananas']"
    )


def test_interrupt_inside_keyed_child_then_continue(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    fruit = make_agent("fruit", interrupt_on="durian")

    def call(state: ParentState) -> dict:
        r = fruit.invoke(
            {"messages": [HumanMessage(content=state["topic"])]}, keyed("session-1")
        )
        return {"result": repr(texts(r))}

    parent = make_parent(call, sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    out = parent.invoke({"topic": "durian", "result": ""}, config)
    assert [i.value for i in out["__interrupt__"]] == [
        "fruit needs approval for durian"
    ]
    assert parent.invoke(Command(resume="ok"), config)["result"] == repr(
        ["durian", "fruit(ok): durian"]
    )
    assert parent.invoke({"topic": "mango", "result": ""}, config)["result"] == repr(
        ["durian", "fruit(ok): durian", "mango", "fruit: mango"]
    )


def test_same_key_in_parallel_is_rejected(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    fruit = make_agent("fruit")

    def slow_reply(state: MessagesState) -> dict:
        time.sleep(0.3)
        return {"messages": [AIMessage(content="ok")]}

    fruit = (
        StateGraph(MessagesState)
        .add_node("reply", slow_reply)
        .add_edge(START, "reply")
        .compile()
    )

    def call(state: ParentState, config: RunnableConfig) -> dict:
        with get_executor_for_config(config) as ex:
            futs = [
                ex.submit(
                    fruit.invoke,
                    {"messages": [HumanMessage(content=q)]},
                    keyed("shared"),
                )
                for q in ("a", "b")
            ]
            return {"result": repr([texts(f.result()) for f in futs])}

    parent = make_parent(call, sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    with pytest.raises(RuntimeError, match="already being executed"):
        parent.invoke({"topic": "", "result": ""}, config)


def test_keyed_child_persists_under_exit_durability(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    fruit = make_agent("fruit")

    def call(state: ParentState) -> dict:
        r = fruit.invoke(
            {"messages": [HumanMessage(content=state["topic"])]}, keyed("k")
        )
        return {"result": repr(texts(r))}

    parent = make_parent(call, sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    parent.invoke({"topic": "kiwi", "result": ""}, config, durability="exit")
    out = parent.invoke({"topic": "lime", "result": ""}, config, durability="exit")
    assert out["result"] == repr(["kiwi", "fruit: kiwi", "lime", "fruit: lime"])


def test_parent_command_from_keyed_child(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    def child_node(state: MessagesState) -> Command:
        return Command(graph=Command.PARENT, update={"result": "from-child"})

    child = (
        StateGraph(MessagesState)
        .add_node("n", child_node)
        .add_edge(START, "n")
        .compile()
    )

    def call(state: ParentState) -> dict:
        child.invoke({"messages": []}, keyed("k1"))
        return {"result": "unreachable"}

    parent = make_parent(call, sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    assert parent.invoke({"topic": "", "result": ""}, config)["result"] == "from-child"


def test_three_level_keyed_nesting(sync_checkpointer: BaseCheckpointSaver) -> None:
    leaf = make_agent("leaf")

    def mid_node(state: MessagesState) -> dict:
        r = leaf.invoke({"messages": [HumanMessage(content="deep")]}, keyed("L"))
        return {"messages": [AIMessage(content="mid saw " + r["messages"][-1].text)]}

    mid = (
        StateGraph(MessagesState)
        .add_node("mid_node", mid_node)
        .add_edge(START, "mid_node")
        .compile()
    )

    def top_node(state: ParentState) -> dict:
        r = mid.invoke({"messages": [HumanMessage(content="hi")]}, keyed("M"))
        return {"result": r["messages"][-1].text}

    top = (
        StateGraph(ParentState)
        .add_node("top_node", top_node)
        .add_edge(START, "top_node")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    chunks = list(
        top.stream(
            {"topic": "", "result": ""}, config, subgraphs=True, stream_mode="updates"
        )
    )
    assert [c[0] for c in chunks] == [
        ("top_node", ":M", "mid_node", ":L"),
        ("top_node", ":M"),
        (),
    ]
    assert namespaces(sync_checkpointer, config) == [
        "",
        "top_node|:M",
        "top_node|:M|mid_node|:L",
    ]
    thread = config["configurable"]
    assert texts(
        top.get_state(
            {"configurable": {**thread, "checkpoint_ns": "top_node|:M"}}
        ).values
    ) == [
        "hi",
        "mid saw leaf: deep",
    ]
    assert texts(
        top.get_state(
            {"configurable": {**thread, "checkpoint_ns": "top_node|:M|mid_node|:L"}}
        ).values
    ) == ["deep", "leaf: deep"]


def test_tool_node_keyed_by_tool_call_id(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """The harness pattern: a tool keys the subagent by its tool_call_id, and a
    second tool continues a conversation by passing that id back."""
    expert = make_agent("expert", interrupt_on="risky")

    @tool
    def ask_expert(question: str, runtime: ToolRuntime) -> str:
        """Ask the expert."""
        r = expert.invoke(
            {"messages": [HumanMessage(content=question)]}, keyed(runtime.tool_call_id)
        )
        return r["messages"][-1].text

    @tool
    def continue_expert(agent_id: str, question: str) -> str:
        """Continue an earlier expert conversation."""
        r = expert.invoke(
            {"messages": [HumanMessage(content=question)]}, keyed(agent_id)
        )
        return " / ".join(texts(r))

    graph = (
        StateGraph(MessagesState)
        .add_node("tools", ToolNode([ask_expert, continue_expert]))
        .add_edge(START, "tools")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ask_expert",
                "args": {"question": "a risky plan"},
                "id": "call_1",
            },
            {"name": "ask_expert", "args": {"question": "a safe plan"}, "id": "call_2"},
        ],
    )
    out = graph.invoke({"messages": [ai]}, config)
    assert [i.value for i in out["__interrupt__"]] == [
        "expert needs approval for a risky plan"
    ]
    out = graph.invoke(Command(resume="approved"), config)
    assert [
        (m.tool_call_id, m.content)
        for m in out["messages"]
        if isinstance(m, ToolMessage)
    ] == [
        ("call_1", "expert(approved): a risky plan"),
        ("call_2", "expert: a safe plan"),
    ]
    assert namespaces(sync_checkpointer, config) == [
        "",
        "tools|:call_1",
        "tools|:call_2",
    ]
    ai2 = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "continue_expert",
                "args": {"agent_id": "call_1", "question": "and now?"},
                "id": "call_3",
            }
        ],
    )
    out = graph.invoke({"messages": [ai2]}, config)
    assert out["messages"][-1].content == (
        "a risky plan / expert(approved): a risky plan / and now? / expert: and now?"
    )


async def test_async_parallel_keyed_and_time_travel(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    fruit = make_agent("fruit")
    turn = 0

    async def call(state: ParentState) -> dict:
        nonlocal turn
        turn += 1
        if turn == 1:
            outs = await asyncio.gather(
                fruit.ainvoke(
                    {"messages": [HumanMessage(content="apples")]}, keyed("a")
                ),
                fruit.ainvoke(
                    {"messages": [HumanMessage(content="bananas")]}, keyed("b")
                ),
            )
            return {"result": repr([texts(o) for o in outs])}
        r = await fruit.ainvoke(
            {"messages": [HumanMessage(content=state["topic"])]}, keyed("a")
        )
        return {"result": repr(texts(r))}

    parent = make_parent(call, async_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    out = await parent.ainvoke({"topic": "", "result": ""}, config)
    assert out["result"] == repr(
        [["apples", "fruit: apples"], ["bananas", "fruit: bananas"]]
    )
    out = await parent.ainvoke({"topic": "more apples", "result": ""}, config)
    assert out["result"] == repr(
        ["apples", "fruit: apples", "more apples", "fruit: more apples"]
    )

    # update_state on a keyed child by explicit namespace
    child_cfg = {"configurable": {**config["configurable"], "checkpoint_ns": "call|:a"}}
    await parent.aupdate_state(child_cfg, {"messages": [AIMessage(content="(note)")]})
    assert texts((await parent.aget_state(child_cfg)).values)[-1] == "(note)"

    # time travel: replaying the parent from the first input checkpoint must
    # fork the keyed child from its pre-turn-1 state, not continue its latest
    history = [s async for s in parent.aget_state_history(config)]
    first_input = next(
        s for s in reversed(history) if s.metadata.get("source") == "input"
    )
    turn = 0
    out = await parent.ainvoke(None, first_input.config)
    assert out["result"] == repr(
        [["apples", "fruit: apples"], ["bananas", "fruit: bananas"]]
    )


def test_key_validation(sync_checkpointer: BaseCheckpointSaver) -> None:
    fruit = make_agent("fruit")

    def call(state: ParentState) -> dict:
        fruit.invoke({"messages": [HumanMessage(content="x")]}, keyed("a|b"))
        return {"result": ""}

    parent = make_parent(call, sync_checkpointer)
    with pytest.raises(ValueError, match="reserved character"):
        parent.invoke(
            {"topic": "", "result": ""}, {"configurable": {"thread_id": str(uuid4())}}
        )


def test_unkeyed_behaviour_unchanged(sync_checkpointer: BaseCheckpointSaver) -> None:
    """Without a key, in-node subgraph invocations keep today's namespaces:
    per-invocation `call:<task_id>` (+ `|<n>` call-order counter) and
    checkpointer=True `call` (+ `|<n>`)."""
    fruit = make_agent("fruit", stateful=True)
    veggie = make_agent("veggie")

    def call(state: ParentState) -> dict:
        fruit.invoke({"messages": [HumanMessage(content="a")]})
        veggie.invoke({"messages": [HumanMessage(content="b")]})
        return {"result": ""}

    parent = make_parent(call, sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    parent.invoke({"topic": "", "result": ""}, config)
    ns = namespaces(sync_checkpointer, config)
    assert ns[0] == "" and ns[1] == "call"
    assert ns[2].startswith("call:") and ns[2].endswith("|1")
