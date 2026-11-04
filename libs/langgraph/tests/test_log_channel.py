"""Transcript log prototype: append-only `Log` / `MessagesLog` channels.

Every scenario runs against `InMemorySaver` (native log storage: segments,
step rows, range reads) and `SqliteSaver` (no native support: the default
`read_log` walks the parent chain), and both must read the same log.
"""

import operator
import sqlite3
import warnings
from collections.abc import Iterator
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver, LogPage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from langgraph.channels import Log, LogRef, MessagesLog
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import Command, Overwrite, Send, interrupt
from langgraph.warnings import LangGraphDeprecatedSinceV13

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["memory", "sqlite"])
def saver(request: pytest.FixtureRequest) -> Iterator[BaseCheckpointSaver]:
    if request.param == "memory":
        yield InMemorySaver()
    else:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        yield SqliteSaver(conn)
        conn.close()


def contents(page: LogPage) -> list[Any]:
    return [e.content if hasattr(e, "content") else e for _, e in page.entries]


class ChatState(TypedDict):
    messages: Annotated[list, MessagesLog()]


def chat_graph(saver: BaseCheckpointSaver, **compile_kw: Any):
    def model(state: ChatState) -> dict:
        return {"messages": [AIMessage(f"reply to {state['messages'][-1].content}")]}

    builder = StateGraph(ChatState)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    return builder.compile(checkpointer=saver, **compile_kw)


def test_appends_reads_and_state_marker(saver: BaseCheckpointSaver) -> None:
    graph = chat_graph(saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": [HumanMessage("a")]}, config)
    out = graph.invoke({"messages": [HumanMessage("b")]}, config)

    # nodes and invoke() see the list
    assert [m.content for m in out["messages"]] == [
        "a",
        "reply to a",
        "b",
        "reply to b",
    ]
    # state reads return the marker, not the transcript
    assert graph.get_state(config).values == {"messages": LogRef(4)}
    # one read operation, same arguments everywhere
    page = graph.read_log(config, "messages")
    assert page.length == 4
    assert contents(page) == ["a", "reply to a", "b", "reply to b"]
    assert [p for p, _ in page.entries] == [0, 1, 2, 3]
    assert contents(graph.read_log(config, "messages", limit=2)) == ["b", "reply to b"]
    assert contents(graph.read_log(config, "messages", before=2, limit=1)) == [
        "reply to a"
    ]
    assert contents(graph.read_log(config, "messages", after=1, limit=1)) == ["b"]
    # the saver-level read returns JSON
    raw = saver.read_log(config, "messages")
    assert raw.entries[0][1]["type"] == "human"
    # every checkpoint points at its own leaf
    lengths = [
        (s.metadata["source"], graph.read_log(s.config, "messages").length)
        for s in graph.get_state_history(config)
    ]
    assert lengths == [
        ("loop", 4),
        ("loop", 3),
        ("input", 2),
        ("loop", 2),
        ("loop", 1),
        ("input", 0),
    ]


async def test_async_appends_and_reads() -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as fallback:
        for saver in (InMemorySaver(), fallback):
            graph = chat_graph(saver)
            config = {"configurable": {"thread_id": "1"}}
            await graph.ainvoke({"messages": [HumanMessage("a")]}, config)
            await graph.ainvoke({"messages": [HumanMessage("b")]}, config)
            state = await graph.aget_state(config)
            assert state.values == {"messages": LogRef(4)}
            page = await graph.aread_log(config, "messages")
            assert contents(page) == ["a", "reply to a", "b", "reply to b"]


def test_log_entries_are_not_stored_in_checkpoints() -> None:
    saver = InMemorySaver()
    graph = chat_graph(saver)
    config = {"configurable": {"thread_id": "1"}}
    for i in range(20):
        graph.invoke({"messages": [HumanMessage(f"m{i}")]}, config)
    assert not [k for k in saver.blobs if k[2] == "messages"]
    log = saver.log.logs[("1", "", "messages")]
    stored = sum(len(rows) for rows in log.entries.values())
    # each entry stored once: linear in the transcript
    assert stored == 40 == graph.read_log(config, "messages").length
    assert len(log.segments) == 1


def test_parallel_tasks_keep_one_order(saver: BaseCheckpointSaver) -> None:
    class State(TypedDict):
        events: Annotated[list, Log()]
        seen: Annotated[list, operator.add]

    def fan_out(state: State) -> list[Send]:
        return [Send("worker", {"i": i}) for i in range(5)]

    def worker(arg: dict) -> dict:
        return {"events": [{"worker": arg["i"]}, {"worker": arg["i"], "n": 2}]}

    def after(state: State) -> dict:
        return {"seen": [list(state["events"])]}

    builder = StateGraph(State)
    builder.add_node("worker", worker)
    builder.add_node("after", after)
    builder.add_conditional_edges(START, fan_out, ["worker"])
    builder.add_edge("worker", "after")
    builder.add_edge("after", END)
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "1"}}
    out = graph.invoke({"events": [], "seen": []}, config)
    in_run = out["seen"][-1]
    assert len(in_run) == 10
    # read back after the run: the same order the run's nodes saw
    assert contents(graph.read_log(config, "events")) == in_run


def test_interrupt_and_resume_reuse_finished_writes(
    saver: BaseCheckpointSaver,
) -> None:
    calls: list[str] = []

    class State(TypedDict):
        events: Annotated[list, Log()]

    def fast(state: State) -> dict:
        calls.append("fast")
        return {"events": [{"from": "fast"}]}

    def asks(state: State) -> dict:
        answer = interrupt("question")
        calls.append("asks")
        return {"events": [{"from": "asks", "answer": answer}]}

    builder = StateGraph(State)
    builder.add_node("fast", fast)
    builder.add_node("asks", asks)
    builder.add_edge(START, "fast")
    builder.add_edge(START, "asks")
    builder.add_edge(["fast", "asks"], END)
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"events": [{"from": "input"}]}, config)
    # fast finished; its entry is pending at the head, not committed
    page = saver.read_log(config, "events")
    assert page.length == 1
    assert page.pending == [{"from": "fast"}]

    out = graph.invoke(Command(resume="yes"), config)
    assert calls == ["fast", "asks"]  # fast was not re-run
    # the reused write and the new one commit as one step, in path order,
    # and the log reads back exactly what the run's state held
    assert out["events"] == [
        {"from": "input"},
        {"from": "asks", "answer": "yes"},
        {"from": "fast"},
    ]
    assert contents(graph.read_log(config, "events")) == out["events"]
    assert graph.read_log(config, "events").pending == []


def test_time_travel_forks_without_touching_the_original(
    saver: BaseCheckpointSaver,
) -> None:
    graph = chat_graph(saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": [HumanMessage("a")]}, config)
    graph.invoke({"messages": [HumanMessage("b")]}, config)
    original = graph.get_state(config).config
    # the checkpoint before the second reply ("b" added, model not run)
    before_reply = next(
        s
        for s in graph.get_state_history(config)
        if s.next == ("model",) and graph.read_log(s.config, "messages").length == 3
    )
    graph.invoke(None, before_reply.config)  # replay: a new branch
    head = graph.get_state(config).config
    assert (
        head["configurable"]["checkpoint_id"]
        != original["configurable"]["checkpoint_id"]
    )
    assert contents(graph.read_log(head, "messages")) == [
        "a",
        "reply to a",
        "b",
        "reply to b",
    ]
    assert contents(graph.read_log(original, "messages")) == [
        "a",
        "reply to a",
        "b",
        "reply to b",
    ]
    a = graph.read_log(head, "messages").entries
    b = graph.read_log(original, "messages").entries
    # same prefix, different last message (a new run of the model node)
    assert [m.id for _, m in a[:3]] == [m.id for _, m in b[:3]]
    assert a[3][1].id != b[3][1].id
    if isinstance(saver, InMemorySaver):
        assert len(saver.log.logs[("1", "", "messages")].segments) == 2


def test_update_state_fork_starts_a_branch(saver: BaseCheckpointSaver) -> None:
    graph = chat_graph(saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": [HumanMessage("a")]}, config)
    graph.invoke({"messages": [HumanMessage("b")]}, config)
    original = graph.get_state(config).config
    first_turn = next(
        s
        for s in graph.get_state_history(config)
        if graph.read_log(s.config, "messages").length == 2
        and s.metadata["source"] == "loop"
    )
    # edit: fork from the end of the first turn with a different question
    forked = graph.update_state(
        first_turn.config, {"messages": [HumanMessage("c")]}, as_node="model"
    )
    assert contents(graph.read_log(forked, "messages")) == ["a", "reply to a", "c"]
    graph.invoke(Command(goto="model"), forked)
    head = graph.get_state(config).config
    assert contents(graph.read_log(head, "messages")) == [
        "a",
        "reply to a",
        "c",
        "reply to c",
    ]
    assert contents(graph.read_log(original, "messages")) == [
        "a",
        "reply to a",
        "b",
        "reply to b",
    ]


def test_subgraph_has_its_own_log(saver: BaseCheckpointSaver) -> None:
    class Sub(TypedDict):
        notes: Annotated[list, Log()]

    def note(state: Sub) -> dict:
        return {"notes": [{"n": len(state.get("notes", []))}]}

    sub_builder = StateGraph(Sub)
    sub_builder.add_node("note", note)
    sub_builder.add_edge(START, "note")
    sub = sub_builder.compile(checkpointer=True)

    class Parent(TypedDict):
        messages: Annotated[list, MessagesLog()]

    def call_sub(state: Parent) -> dict:
        sub.invoke({"notes": [{"from": "parent"}]})
        return {"messages": [AIMessage("done")]}

    builder = StateGraph(Parent)
    builder.add_node("call_sub", call_sub)
    builder.add_edge(START, "call_sub")
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": [HumanMessage("go")]}, config)
    graph.invoke({"messages": [HumanMessage("again")]}, config)
    assert contents(graph.read_log(config, "messages")) == [
        "go",
        "done",
        "again",
        "done",
    ]
    sub_config = {"configurable": {"thread_id": "1", "checkpoint_ns": "call_sub"}}
    assert contents(saver.read_log(sub_config, "notes")) == [
        {"from": "parent"},
        {"n": 1},
        {"from": "parent"},
        {"n": 3},
    ]


def test_exit_durability_commits_inline(saver: BaseCheckpointSaver) -> None:
    graph = chat_graph(saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": [HumanMessage("a")]}, config, durability="exit")
    graph.invoke({"messages": [HumanMessage("b")]}, config, durability="exit")
    assert contents(graph.read_log(config, "messages")) == [
        "a",
        "reply to a",
        "b",
        "reply to b",
    ]


def test_prune_keeps_the_heads_transcript() -> None:
    saver = InMemorySaver()
    graph = chat_graph(saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": [HumanMessage("a")]}, config)
    graph.invoke({"messages": [HumanMessage("b")]}, config)
    original = graph.get_state(config).config
    before_reply = next(
        s
        for s in graph.get_state_history(config)
        if s.next == ("model",) and graph.read_log(s.config, "messages").length == 3
    )
    graph.invoke(None, before_reply.config)
    log = saver.log.logs[("1", "", "messages")]
    assert len(log.segments) == 2

    saver.prune(["1"], strategy="keep_latest")
    assert len(saver.storage["1"][""]) == 1
    # the head's full transcript survives without its ancestors' checkpoints
    assert contents(graph.read_log(config, "messages")) == [
        "a",
        "reply to a",
        "b",
        "reply to b",
    ]
    # the abandoned branch is collected
    log = saver.log.logs[("1", "", "messages")]
    assert sum(len(r) for r in log.entries.values()) == 4
    assert len(log.steps) == 4
    assert graph.read_log(original, "messages").length == 0  # its checkpoint is gone

    # the thread keeps working after prune
    graph.invoke({"messages": [HumanMessage("c")]}, config)
    assert graph.read_log(config, "messages").length == 6

    saver.prune(["1"], strategy="delete")
    assert ("1", "", "messages") not in saver.log.logs


def test_rollback_collects_the_runs_entries() -> None:
    saver = InMemorySaver()
    graph = chat_graph(saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke(
        {"messages": [HumanMessage("a")]}, {**config, "metadata": {"run_id": "r1"}}
    )
    graph.invoke(
        {"messages": [HumanMessage("b")]}, {**config, "metadata": {"run_id": "r2"}}
    )
    saver.delete_for_runs(["r2"])
    assert contents(graph.read_log(config, "messages")) == ["a", "reply to a"]
    log = saver.log.logs[("1", "", "messages")]
    assert sum(len(r) for r in log.entries.values()) == 2
    # the tip moved back: the next run continues the same segment
    graph.invoke({"messages": [HumanMessage("c")]}, config)
    assert contents(graph.read_log(config, "messages")) == [
        "a",
        "reply to a",
        "c",
        "reply to c",
    ]
    assert len(saver.log.logs[("1", "", "messages")].segments) == 1


def test_copy_and_delete_thread() -> None:
    saver = InMemorySaver()
    graph = chat_graph(saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": [HumanMessage("a")]}, config)
    saver.copy_thread("1", "2")
    copy_config = {"configurable": {"thread_id": "2"}}
    graph.invoke({"messages": [HumanMessage("b")]}, copy_config)
    assert contents(graph.read_log(config, "messages")) == ["a", "reply to a"]
    assert contents(graph.read_log(copy_config, "messages")) == [
        "a",
        "reply to a",
        "b",
        "reply to b",
    ]
    saver.delete_thread("1")
    assert ("1", "", "messages") not in saver.log.logs
    assert graph.read_log(copy_config, "messages").length == 4


def test_log_takes_only_json(saver: BaseCheckpointSaver) -> None:
    class State(TypedDict):
        events: Annotated[list, Log()]

    def bad(state: State) -> dict:
        return {"events": [{"when": object()}]}

    builder = StateGraph(State)
    builder.add_node("bad", bad)
    builder.add_edge(START, "bad")
    graph = builder.compile(checkpointer=saver)
    with pytest.raises(InvalidUpdateError, match="not JSON"):
        graph.invoke({"events": [{"ok": 1}]}, {"configurable": {"thread_id": "1"}})
    with pytest.raises(InvalidUpdateError, match="not JSON"):
        graph.invoke({"events": [float("nan")]}, {"configurable": {"thread_id": "2"}})


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda s: [RemoveMessage(id=s["messages"][0].id)], id="remove"),
        pytest.param(lambda s: Overwrite([AIMessage("x")]), id="overwrite"),
        pytest.param(lambda s: [s["messages"][0]], id="replace-by-id"),
        pytest.param(lambda s: s["messages"] + [AIMessage("x")], id="whole-list"),
    ],
)
def test_messages_log_rejects_edits(saver: BaseCheckpointSaver, write: Any) -> None:
    def edits(state: ChatState) -> dict:
        return {"messages": write(state)}

    builder = StateGraph(ChatState)
    builder.add_node("edits", edits)
    builder.add_edge(START, "edits")
    graph = builder.compile(checkpointer=saver)
    with pytest.raises(InvalidUpdateError):
        graph.invoke(
            {"messages": [HumanMessage("a")]}, {"configurable": {"thread_id": "1"}}
        )


def test_fallback_and_native_read_the_same_log() -> None:
    def run(saver: BaseCheckpointSaver) -> list[list[Any]]:
        graph = chat_graph(saver)
        config = {"configurable": {"thread_id": "1"}}
        for q in "abc":
            graph.invoke({"messages": [HumanMessage(q)]}, config)
        return [
            contents(graph.read_log(s.config, "messages"))
            for s in graph.get_state_history(config)
        ]

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        assert run(InMemorySaver()) == run(SqliteSaver(conn))
    finally:
        conn.close()


def test_resume_at_a_superseded_checkpoint_branches(
    saver: BaseCheckpointSaver,
) -> None:
    """A resume at a checkpoint the thread moved past re-runs finished tasks on
    that same checkpoint, without a fork checkpoint. The log puts the re-run
    in a new segment and leaves the original branch alone."""
    calls: list[str] = []

    class State(TypedDict):
        events: Annotated[list, Log()]

    def fast(state: State) -> dict:
        calls.append("fast")
        return {"events": [{"from": "fast", "run": len(calls)}]}

    def asks(state: State) -> dict:
        answer = interrupt("question")
        return {"events": [{"from": "asks", "answer": answer}]}

    builder = StateGraph(State)
    builder.add_node("fast", fast)
    builder.add_node("asks", asks)
    builder.add_edge(START, "fast")
    builder.add_edge(START, "asks")
    builder.add_edge(["fast", "asks"], END)
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"events": [{"from": "input"}]}, config)
    interrupted = graph.get_state(config).config
    first = graph.invoke(Command(resume="yes"), config)["events"]
    original_head = graph.get_state(config).config

    second = graph.invoke(Command(resume="no"), interrupted)["events"]
    assert calls == ["fast", "fast"]  # the finished task re-ran
    new_head = graph.get_state(config).config
    assert contents(graph.read_log(new_head, "events")) == second
    assert contents(graph.read_log(original_head, "events")) == first
    assert first != second
    if isinstance(saver, InMemorySaver):
        assert len(saver.log.logs[("1", "", "events")].segments) == 2


def test_tool_calling_agent_runs_on_a_messages_log(
    saver: BaseCheckpointSaver,
) -> None:
    """Prebuilt `ToolNode` and `tools_condition` work unchanged: they read the
    list and append messages."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def model(state: ChatState) -> dict:
        last = state["messages"][-1]
        if isinstance(last, HumanMessage):
            call = {
                "name": "add",
                "args": {"a": 2, "b": 3},
                "id": f"call-{len(state['messages'])}",
            }
            return {"messages": [AIMessage("", tool_calls=[call])]}
        return {"messages": [AIMessage(f"the answer is {last.content}")]}

    builder = StateGraph(ChatState)
    builder.add_node("model", model)
    builder.add_node("tools", ToolNode([add]))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "1"}}
    out = graph.invoke({"messages": [HumanMessage("2+3?")]}, config)
    assert [type(m).__name__ for m in out["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    page = graph.read_log(config, "messages")
    assert [m.id for _, m in page.entries] == [m.id for m in out["messages"]]
    assert page.entries[-1][1].content == "the answer is 5"


def test_input_after_update_state_reads_the_inline_step(
    saver: BaseCheckpointSaver,
) -> None:
    """`update_state` commits its entries inline; the input checkpoint after
    it keeps the version with a stripped marker, and reads must still find
    the entries on the checkpoint that committed them."""
    graph = chat_graph(saver)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": [HumanMessage("a")]}, config)
    graph.update_state(config, {"messages": [HumanMessage("note")]})
    graph.invoke({"messages": [HumanMessage("b")]}, config)
    assert contents(graph.read_log(config, "messages")) == [
        "a",
        "reply to a",
        "note",
        "b",
        "reply to b",
    ]


# -- existing `add_messages` threads ------------------------------------------


class OldChatState(TypedDict):
    messages: Annotated[list, add_messages]


def _reply(state: dict) -> dict:
    return {"messages": [AIMessage(f"reply to {state['messages'][-1].content}")]}


def _chat(schema: type, saver: BaseCheckpointSaver, **nodes: Any):
    builder = StateGraph(schema)
    builder.add_node("model", nodes.get("model", _reply))
    builder.add_edge(START, "model")
    if "review" in nodes:
        builder.add_node("review", nodes["review"])
        builder.add_edge("model", "review")
        builder.add_edge("review", END)
    else:
        builder.add_edge("model", END)
    return builder.compile(checkpointer=saver)


def test_add_messages_thread_switches_to_messages_log(
    saver: BaseCheckpointSaver,
) -> None:
    """A thread written with `add_messages` keeps working on `MessagesLog`:
    the stored list is the log's first entries."""
    config = {"configurable": {"thread_id": "1"}}
    old = _chat(OldChatState, saver)
    old.invoke({"messages": [HumanMessage("a", id="h1")]}, config)
    old.invoke({"messages": [HumanMessage("b", id="h2")]}, config)
    before_switch = old.get_state(config).config

    new = _chat(ChatState, saver)
    out = new.invoke({"messages": [HumanMessage("c", id="h3")]}, config)
    assert [m.content for m in out["messages"]] == [
        "a",
        "reply to a",
        "b",
        "reply to b",
        "c",
        "reply to c",
    ]
    assert new.get_state(config).values == {"messages": LogRef(6)}
    page = new.read_log(config, "messages")
    assert [p for p, _ in page.entries] == list(range(6))
    assert contents(page) == [m.content for m in out["messages"]]
    assert [m.id for _, m in page.entries][:1] == ["h1"]
    # checkpoints from before the switch still read as the list they stored
    assert contents(new.read_log(before_switch, "messages")) == [
        "a",
        "reply to a",
        "b",
        "reply to b",
    ]
    assert len(new.get_state(before_switch).values["messages"]) == 4
    # the next turn appends after the seed
    new.invoke({"messages": [HumanMessage("d")]}, config)
    assert new.read_log(config, "messages").length == 8
    if isinstance(saver, InMemorySaver):
        log = saver.log.logs[("1", "", "messages")]
        assert sum(len(rows) for rows in log.entries.values()) == 8


def test_switch_with_an_interrupted_add_messages_run(
    saver: BaseCheckpointSaver,
) -> None:
    """The first checkpoint after the switch can come from the loop (a
    resume), not graph input; the seed is committed with it."""

    def review(state: dict) -> dict:
        answer = interrupt("approve?")
        return {"messages": [AIMessage(f"review: {answer}")]}

    config = {"configurable": {"thread_id": "1"}}
    old = _chat(OldChatState, saver, review=review)
    old.invoke({"messages": [HumanMessage("a")]}, config)
    assert old.get_state(config).next == ("review",)

    new = _chat(ChatState, saver, review=review)
    out = new.invoke(Command(resume="ok"), config)
    assert [m.content for m in out["messages"]] == ["a", "reply to a", "review: ok"]
    assert contents(new.read_log(config, "messages")) == [
        "a",
        "reply to a",
        "review: ok",
    ]


def test_switch_then_update_state(saver: BaseCheckpointSaver) -> None:
    config = {"configurable": {"thread_id": "1"}}
    old = _chat(OldChatState, saver)
    old.invoke({"messages": [HumanMessage("a")]}, config)
    new = _chat(ChatState, saver)
    new.update_state(config, {"messages": [HumanMessage("note")]})
    assert contents(new.read_log(config, "messages")) == ["a", "reply to a", "note"]
    out = new.invoke({"messages": [HumanMessage("b")]}, config)
    assert [m.content for m in out["messages"]][-2:] == ["b", "reply to b"]
    assert new.read_log(config, "messages").length == 5


# -- deprecation of message edit ops on `add_messages` -------------------------


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda s: [RemoveMessage(id=s["messages"][0].id)], id="remove"),
        pytest.param(
            lambda s: [RemoveMessage(id=REMOVE_ALL_MESSAGES), AIMessage("x")],
            id="remove-all",
        ),
        pytest.param(
            lambda s: [HumanMessage("edited", id=s["messages"][0].id)],
            id="replace-by-id",
        ),
        pytest.param(lambda s: Overwrite([AIMessage("x")]), id="overwrite"),
    ],
)
def test_add_messages_edit_ops_are_deprecated(edit: Any) -> None:
    def node(state: dict) -> dict:
        return {"messages": edit(state)}

    graph = _chat(OldChatState, InMemorySaver(), model=node)
    with pytest.warns(LangGraphDeprecatedSinceV13, match="MessagesLog"):
        graph.invoke(
            {"messages": [HumanMessage("a")]}, {"configurable": {"thread_id": "1"}}
        )


def test_add_messages_appends_are_not_deprecated() -> None:
    graph = _chat(OldChatState, InMemorySaver())
    with warnings.catch_warnings():
        warnings.simplefilter("error", LangGraphDeprecatedSinceV13)
        graph.invoke(
            {"messages": [HumanMessage("a")]}, {"configurable": {"thread_id": "1"}}
        )
