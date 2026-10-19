"""Strict-mode deserialization of control types (Send / Command / Overwrite).

Regression tests for the control-type allowlist gate. Before the fix,
``_check_allowed`` consulted ``SAFE_MSGPACK_TYPES`` before any configured
allowlist, so ``("langgraph.types", "Send")`` was reconstructed under
``LANGGRAPH_STRICT_MSGPACK=true`` with no explicit opt-in. A forged
``writes`` row (NULL_TASK_ID, ``__pregel_tasks``, a legitimately serialized
``Send`` blob) therefore scheduled an edge-unreachable node with an
attacker-chosen argument on resume, defeating the containment strict mode
promises.

After the fix, control types deserialize only when explicitly present in the
configured allowlist: an operator-supplied ``allowed_msgpack_modules``, or the
allowlist a graph derives at compile time when its own nodes or routers emit
control types.
"""

from __future__ import annotations

import operator
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import ormsgpack
import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde import _msgpack as _lg_msgpack
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

from langgraph._internal import _serde
from langgraph._internal._constants import NULL_TASK_ID
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Overwrite, Send, interrupt

CONTROL_KEYS = [
    ("langgraph.types", "Send"),
    ("langgraph.types", "Command"),
    ("langgraph.types", "Overwrite"),
]

FORGED_ARG = {"directive": "exfiltrate", "token": "attacker-controlled"}


class ForgedState(TypedDict):
    out: str
    admin_out: str


class ControlState(TypedDict):
    out: str


class MapState(TypedDict):
    xs: list[str]
    out: Annotated[list, operator.add]


class CmdState(TypedDict):
    name: str
    name2: str


class MsgState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _strict_serde() -> JsonPlusSerializer:
    # Explicit None rather than relying on the env flag: the flag is read at
    # serializer construction, and BaseCheckpointSaver's constructor default
    # is a shared instance created at import time (tests.conftest_checkpointer
    # takes the same precaution for the strict-mode CI job).
    return JsonPlusSerializer(allowed_msgpack_modules=None)


@pytest.fixture(autouse=True)
def _strict_msgpack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in this module as if LANGGRAPH_STRICT_MSGPACK=true."""
    monkeypatch.setattr(_lg_msgpack, "STRICT_MSGPACK_ENABLED", True)
    monkeypatch.setattr(_serde, "STRICT_MSGPACK_ENABLED", True)


# --- (a) the forged Send row is rejected under strict mode ---


def test_forged_null_task_send_write_is_rejected(tmp_path: Path) -> None:
    """A forged pending-write Send to an edge-unreachable node must not run.

    Mirrors the reported exploit: the attacker holds checkpoint-store write
    access and INSERTs one ``writes`` row (NULL_TASK_ID, ``__pregel_tasks``,
    legitimately serialized ``Send("admin", arg)``). Before the fix this row
    was reconstructed under strict mode and the privileged node executed with
    the attacker argument on resume; see
    test_forged_send_blob_is_a_legitimate_send_payload for proof that the
    bytes themselves are a well-formed ``Send``.
    """
    sentinel = tmp_path / "admin_ran"

    def ask(state: ForgedState) -> dict:
        answer = interrupt("approve?")
        return {"out": f"action taken with {answer}"}

    def admin(payload: dict) -> dict:
        sentinel.write_text(repr(payload))
        return {"admin_out": "admin ran"}

    conn = sqlite3.connect(tmp_path / "forged.db", check_same_thread=False)
    try:
        saver = SqliteSaver(conn, serde=_strict_serde())
        builder = StateGraph(ForgedState)
        builder.add_node("ask", ask)
        builder.add_node("admin", admin)  # compiled in, but NO edge reaches it
        builder.add_edge(START, "ask")
        builder.add_edge("ask", END)
        graph = builder.compile(checkpointer=saver)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        list(graph.stream({"x": 1}, config, stream_mode="updates"))
        assert graph.get_state(config).next == ("ask",)

        forge_pending_write(
            db_path=tmp_path / "forged.db",
            thread_id=thread_id,
            channel="__pregel_tasks",
            blob=JsonPlusSerializer().dumps_typed(Send("admin", FORGED_ARG)),
        )

        events = list(graph.stream(Command(resume="y"), config, stream_mode="updates"))
        ran_nodes = [node for event in events for node in event]

        assert "admin" not in ran_nodes
        assert not sentinel.exists()
        # the resume itself completes normally; only the forged packet drops
        assert graph.get_state(config).values["out"] == "action taken with y"
    finally:
        conn.close()


def test_forged_os_system_ext_stays_blocked(tmp_path: Path) -> None:
    """The strict-mode constructor gate for non-safe types still holds."""
    sentinel = tmp_path / "os_ran"

    def ask(state: ControlState) -> dict:
        answer = interrupt("approve?")
        return {"out": f"action taken with {answer}"}

    ctrl_blob = ormsgpack.packb(
        {"p": None},
        default=lambda _o: ormsgpack.Ext(
            0, ormsgpack.packb(("os", "system", f"touch {sentinel}"))
        ),
        option=ormsgpack.OPT_NON_STR_KEYS,
    )

    conn = sqlite3.connect(tmp_path / "ctrl.db", check_same_thread=False)
    try:
        saver = SqliteSaver(conn, serde=_strict_serde())
        builder = StateGraph(ControlState)
        builder.add_node("ask", ask)
        builder.add_edge(START, "ask")
        builder.add_edge("ask", END)
        graph = builder.compile(checkpointer=saver)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        list(graph.stream({"x": 1}, config, stream_mode="updates"))
        assert graph.get_state(config).next == ("ask",)

        forge_pending_write(
            db_path=tmp_path / "ctrl.db",
            thread_id=thread_id,
            channel="__interrupt__",
            blob=("msgpack", ctrl_blob),
        )

        snapshot = graph.get_state(config)  # deserializes the poisoned row

        assert not sentinel.exists()
        assert snapshot.next == ("ask",)
    finally:
        conn.close()


def test_forged_send_blob_is_a_legitimate_send_payload() -> None:
    """The forged row's bytes are a well-formed Send, which is the point.

    With ``Send`` allowlisted (what pre-fix semantics did unconditionally via
    SAFE_MSGPACK_TYPES), the exact blob written by the forged row
    deserializes to a live ``Send`` for the unreachable node with the
    attacker argument. This documents both that the blob needs no exotic
    encoding and that the fix blocks a payload the graph would otherwise
    execute.
    """
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("langgraph.types", "Send")])
    blob = JsonPlusSerializer().dumps_typed(Send("admin", FORGED_ARG))

    revived = serde.loads_typed(blob)

    assert isinstance(revived, Send)
    assert revived.node == "admin"
    assert revived.arg == FORGED_ARG


# --- (b) opted-in users are unaffected ---


@pytest.mark.parametrize(
    ("obj", "cls"),
    [
        (Send("worker", {"x": 1}), Send),
        (Command(goto="worker", update={"x": 1}), Command),
        (Overwrite([1, 2]), Overwrite),
    ],
    ids=["send", "command", "overwrite"],
)
def test_control_types_round_trip_with_explicit_allowlist(
    obj: object, cls: type
) -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=CONTROL_KEYS)

    revived = serde.loads_typed(serde.dumps_typed(obj))

    assert isinstance(revived, cls)
    assert revived == obj


def test_control_types_round_trip_with_derived_allowlist() -> None:
    """The compile-time derived allowlist (with_msgpack_allowlist) admits
    control types for graphs that emit them."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    allowlisted = serde.with_msgpack_allowlist([("langgraph.types", "Send")])

    revived = allowlisted.loads_typed(allowlisted.dumps_typed(Send("worker", {"x": 1})))

    assert revived == Send("worker", {"x": 1})


def test_map_reduce_sends_still_work() -> None:
    """No false positive: a graph whose router emits Sends keeps working.

    compile() derives its allowlist from the graph's own callables, so the
    fan-out Sends round-trip through the checkpointer under strict mode.
    """

    def source(state: MapState) -> dict:
        return {"xs": state["xs"]}

    def fan_out(state: MapState) -> list[Send]:
        return [Send("work", {"k": x}) for x in state["xs"]]

    def work(state: dict) -> dict:
        return {"out": [state["k"] + "!"]}

    builder = StateGraph(MapState)
    builder.add_node("source", source)
    builder.add_node("work", work)
    builder.add_edge(START, "source")
    builder.add_conditional_edges("source", fan_out)
    builder.add_edge("work", END)
    graph = builder.compile(checkpointer=InMemorySaver(serde=_strict_serde()))

    config = {"configurable": {"thread_id": str(uuid4())}}
    result = graph.invoke({"xs": ["a", "b"]}, config)

    assert result["out"] == ["a!", "b!"]


def test_node_returning_command_goto_send_still_works() -> None:
    """A node routing through Command(goto=[Send(...)]) keeps working."""

    def one(state: CmdState) -> Command:
        return Command(goto=[Send("two", {"name": state["name"]})])

    def two(state: dict) -> dict:
        return {"name2": state["name"] + "-two"}

    builder = StateGraph(CmdState)
    builder.add_node("one", one)
    builder.add_node("two", two)
    builder.add_edge(START, "one")
    builder.add_edge("two", END)
    graph = builder.compile(checkpointer=InMemorySaver(serde=_strict_serde()))

    config = {"configurable": {"thread_id": str(uuid4())}}
    result = graph.invoke({"name": "x"}, config)

    assert result == {"name": "x", "name2": "x-two"}


# --- strict mode without opt-in degrades control types to data ---


@pytest.mark.parametrize(
    ("obj", "raw"),
    [
        (Send("worker", {"x": 1}), ["worker", {"x": 1}]),
        (
            Command(goto="worker"),
            {
                "goto": "worker",
                "update": None,
                "resume": None,
                "graph": None,
            },
        ),
        (Overwrite([1, 2]), {"value": [1, 2], "type": "__overwrite__"}),
    ],
    ids=["send", "command", "overwrite"],
)
def test_control_types_degrade_to_data_without_allowlist(
    obj: object, raw: object
) -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)

    revived = serde.loads_typed(serde.dumps_typed(obj))

    assert not isinstance(revived, (Send, Command, Overwrite))
    assert revived == raw


# --- (c) plain data types are unaffected ---


def test_data_types_still_revive_under_strict_mode() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    msg = HumanMessage(content="hello")
    when = datetime(2026, 1, 1, 12, 30)
    uid = uuid.uuid4()

    assert serde.loads_typed(serde.dumps_typed(msg)) == msg
    assert serde.loads_typed(serde.dumps_typed(when)) == when
    assert serde.loads_typed(serde.dumps_typed(uid)) == uid


def test_message_state_still_round_trips() -> None:
    def greet(state: MsgState) -> dict:
        return {"messages": [HumanMessage(content="hi", id="1")]}

    builder = StateGraph(MsgState)
    builder.add_node("greet", greet)
    builder.add_edge(START, "greet")
    builder.add_edge("greet", END)
    graph = builder.compile(checkpointer=InMemorySaver(serde=_strict_serde()))

    config = {"configurable": {"thread_id": str(uuid4())}}
    result = graph.invoke({"messages": []}, config)

    assert [m.content for m in result["messages"]] == ["hi"]


# --- helpers ---


def forge_pending_write(
    *,
    db_path: Path,
    thread_id: str,
    channel: str,
    blob: tuple[str, bytes],
) -> None:
    """INSERT one forged pending write, as an attacker with store write
    access would: NULL_TASK_ID task, the given channel, and the given
    legitimately serialized blob."""
    conn = sqlite3.connect(db_path)
    try:
        ns, ckpt = conn.execute(
            "SELECT checkpoint_ns, checkpoint_id FROM checkpoints "
            "WHERE thread_id=? ORDER BY rowid DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        assert ns is not None
        conn.execute(
            "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, "
            "task_id, idx, channel, type, value) VALUES (?,?,?,?,?,?,?,?)",
            (
                thread_id,
                ns,
                ckpt,
                NULL_TASK_ID,
                9,
                channel,
                blob[0],
                blob[1],
            ),
        )
        conn.commit()
    finally:
        conn.close()
