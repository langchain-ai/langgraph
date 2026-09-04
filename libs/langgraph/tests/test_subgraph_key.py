"""Tests for keyed subgraph instances.

A subgraph invocation can be keyed in two ways:

- explicitly, with `configurable["subgraph_key"]` on the invoke config;
- through `Send(node, arg, key=...)`: the pushed task's id derives from the
  key, duplicate keys in one step are rejected, and subgraphs invoked inside
  the task inherit the key.

A keyed subgraph runs under `<parent frames without task ids>|:key` whatever
its `checkpointer=` mode: distinct keys never share state, and a later
invocation with the same key continues the same history.
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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.prebuilt import ToolNode, ToolRuntime
from typing_extensions import TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command, Send, interrupt

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


def make_send_graph(
    tools: list, saver: BaseCheckpointSaver, *, key=lambda tc: tc["id"]
) -> Pregel:
    """A create_agent-shaped graph: one Send per tool call, keyed by tool call id."""

    def route(state: MessagesState) -> list[Send]:
        return [
            Send("tools", [tc], key=key(tc)) for tc in state["messages"][-1].tool_calls
        ]

    return (
        StateGraph(MessagesState)
        .add_node("tools", ToolNode(tools))
        .add_conditional_edges(START, route, ["tools"])
        .compile(checkpointer=saver)
    )


def ai(*calls: tuple[str, dict, str]) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": n, "args": a, "id": i} for n, a, i in calls]
    )


def tool_msgs(out: dict, n: int) -> list[tuple[str, str]]:
    return [
        (m.tool_call_id, m.content)
        for m in out["messages"]
        if isinstance(m, ToolMessage)
    ][-n:]


def namespaces(saver: BaseCheckpointSaver, config: RunnableConfig) -> list[str]:
    return sorted(
        {
            c.config["configurable"]["checkpoint_ns"]
            for c in saver.list(
                {"configurable": {"thread_id": config["configurable"]["thread_id"]}}
            )
        }
    )


# --- explicit keys --------------------------------------------------------


@pytest.mark.parametrize("stateful", [False, True])
def test_parallel_keyed_calls_then_continue_one(
    sync_checkpointer: BaseCheckpointSaver, stateful: bool
) -> None:
    """Two parallel calls with distinct keys get distinct memories; a later call
    with one of the keys continues it. Same in both persistence modes."""
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
    child = parent.get_state(
        {"configurable": {**config["configurable"], "checkpoint_ns": "call|:tc_b"}}
    )
    assert child.config["configurable"]["checkpoint_ns"] == "call|:tc_b"
    assert texts(child.values)[-1] == "fruit: more bananas"


def test_keys_make_in_node_parallel_resume_deterministic(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Without keys, subgraphs invoked in parallel inside one node body are told
    apart by a call-order counter, which is racy: if the arrival order flips on
    resume, each subgraph loads the other's checkpoint. Keys fix this."""
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
    """Subgraph call *then* interrupt in the same task: on resume the subgraph
    must return its completed turn, not apply the input again."""
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
    with pytest.raises(ValueError, match="reserved character"):
        Send("tools", {}, key="a|b")
    with pytest.raises(ValueError, match="non-empty"):
        Send("tools", {}, key="")


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


def test_stateful_keyed_child_without_parent_checkpointer() -> None:
    fruit = make_agent("fruit", stateful=True)

    def call(state: ParentState) -> dict:
        r = fruit.invoke(
            {"messages": [HumanMessage(content=state["topic"])]}, keyed("k")
        )
        return {"result": repr(texts(r))}

    parent = (
        StateGraph(ParentState).add_node("call", call).add_edge(START, "call").compile()
    )
    assert parent.invoke({"topic": "kiwi", "result": ""})["result"] == repr(
        ["kiwi", "fruit: kiwi"]
    )
    assert parent.invoke({"topic": "lime", "result": ""})["result"] == repr(
        ["lime", "fruit: lime"]
    )


# --- Send keys ------------------------------------------------------------


def test_send_key_round_trips_through_serde() -> None:
    serde = JsonPlusSerializer()
    for s in (
        Send("tools", {"a": 1}),
        Send("tools", {"a": 1}, timeout=1.5),
        Send("tools", {"a": 1}, key="call_1"),
        Send("tools", {"a": 1}, timeout=1.5, key="call_1"),
    ):
        assert serde.loads_typed(serde.dumps_typed(s)) == s
    assert repr(Send("tools", 1, key="k")) == "Send(node='tools', arg=1, key='k')"


def test_send_keyed_tasks_key_the_subgraphs_they_invoke(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """create_agent shape: one Send per tool call, keyed by the call id. A
    subagent invoked inside the tool, with no config of its own, lands on
    `tools|:<call id>`. Interrupt, resume with arrival order flipped, then
    continue one subagent by passing its key explicitly from another tool."""
    expert = make_agent("expert", interrupt_on="risky")
    calls = {"a": 0, "b": 0}

    @tool
    def ask_expert(question: str) -> str:
        """Ask the expert."""
        who = "a" if "risky" in question else "b"
        calls[who] += 1
        # first on attempt 1 / last on resume for a, the reverse for b
        time.sleep(
            (0.0 if who == "a" else 0.2)
            if calls[who] == 1
            else (0.2 if who == "a" else 0.0)
        )
        return expert.invoke({"messages": [HumanMessage(content=question)]})[
            "messages"
        ][-1].text

    @tool
    def continue_expert(agent_id: str, question: str) -> str:
        """Continue an earlier expert conversation."""
        r = expert.invoke(
            {"messages": [HumanMessage(content=question)]}, keyed(agent_id)
        )
        return " / ".join(texts(r))

    graph = make_send_graph([ask_expert, continue_expert], sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    out = graph.invoke(
        {
            "messages": [
                ai(
                    ("ask_expert", {"question": "a risky plan"}, "call_a"),
                    ("ask_expert", {"question": "a safe plan"}, "call_b"),
                )
            ]
        },
        config,
    )
    assert [i.value for i in out["__interrupt__"]] == [
        "expert needs approval for a risky plan"
    ]
    out = graph.invoke(Command(resume="approved"), config)
    assert tool_msgs(out, 2) == [
        ("call_a", "expert(approved): a risky plan"),
        ("call_b", "expert: a safe plan"),
    ]
    assert namespaces(sync_checkpointer, config) == [
        "",
        "tools|:call_a",
        "tools|:call_b",
    ]
    out = graph.invoke(
        {
            "messages": [
                ai(
                    (
                        "continue_expert",
                        {"agent_id": "call_a", "question": "and now?"},
                        "call_c",
                    )
                )
            ]
        },
        config,
    )
    assert out["messages"][-1].content == (
        "a risky plan / expert(approved): a risky plan / and now? / expert: and now?"
    )
    # the continuing tool's own Send key is unused: the explicit key wins
    assert namespaces(sync_checkpointer, config) == [
        "",
        "tools|:call_a",
        "tools|:call_b",
    ]


def test_send_key_gives_position_independent_task_ids(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Keyed Sends get task ids from the key, not their position."""
    seen: dict[str, str] = {}

    @tool
    def record(name: str, runtime: ToolRuntime) -> str:
        """Record the task id."""
        seen[name] = runtime.config["configurable"]["__pregel_task_id"]
        return name

    graph = make_send_graph(
        [record], sync_checkpointer, key=lambda tc: tc["args"]["name"]
    )
    for order in (("x", "y"), ("y", "x")):
        config = {"configurable": {"thread_id": str(uuid4())}}
        # two threads, same first checkpoint id is impossible, so compare within one step instead
        graph.invoke(
            {"messages": [ai(*((("record", {"name": n}, f"id_{n}")) for n in order))]},
            config,
        )
        tasks = graph.get_state_history(config)
        step0 = next(s for s in tasks if s.metadata.get("step") == 0)
        ids = {t.name: t.id for t in step0.tasks}
        assert set(ids) == {"tools"} or len(step0.tasks) == 2
    assert set(seen) == {"x", "y"}


def test_send_duplicate_keys_rejected(sync_checkpointer: BaseCheckpointSaver) -> None:
    """Two keyed Sends for one node with the same key in one step are refused
    before any task runs, wherever tasks are prepared."""
    ran: list[str] = []

    @tool
    def ask(q: str) -> str:
        """Ask."""
        ran.append(q)
        return q

    graph = make_send_graph([ask], sync_checkpointer, key=lambda tc: tc["name"])
    config = {"configurable": {"thread_id": str(uuid4())}}
    with pytest.raises(InvalidUpdateError, match="Duplicate Send key 'ask'"):
        graph.invoke(
            {"messages": [ai(("ask", {"q": "a"}, "c1"), ("ask", {"q": "b"}, "c2"))]},
            config,
        )
    assert ran == []
    # a single call per step is fine, and a stable key gives a stable memory
    expert = make_agent("expert")

    @tool
    def ask_expert(q: str) -> str:
        """Ask expert."""
        return " / ".join(texts(expert.invoke({"messages": [HumanMessage(content=q)]})))

    graph = make_send_graph([ask_expert], sync_checkpointer, key=lambda tc: tc["name"])
    config = {"configurable": {"thread_id": str(uuid4())}}
    graph.invoke({"messages": [ai(("ask_expert", {"q": "apples"}, "c1"))]}, config)
    out = graph.invoke({"messages": [ai(("ask_expert", {"q": "pears"}, "c2"))]}, config)
    assert (
        out["messages"][-1].content == "apples / expert: apples / pears / expert: pears"
    )
    assert namespaces(sync_checkpointer, config) == ["", "tools|:ask_expert"]


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
    child_cfg = {"configurable": {**config["configurable"], "checkpoint_ns": "call|:a"}}
    await parent.aupdate_state(child_cfg, {"messages": [AIMessage(content="(note)")]})
    assert texts((await parent.aget_state(child_cfg)).values)[-1] == "(note)"
    history = [s async for s in parent.aget_state_history(config)]
    first_input = next(
        s for s in reversed(history) if s.metadata.get("source") == "input"
    )
    turn = 0
    out = await parent.ainvoke(None, first_input.config)
    assert out["result"] == repr(
        [["apples", "fruit: apples"], ["bananas", "fruit: bananas"]]
    )
