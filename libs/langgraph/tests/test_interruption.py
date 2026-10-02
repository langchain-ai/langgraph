from dataclasses import dataclass
from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from langgraph.func import task
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Durability, Interrupt, Send, StateUpdate, interrupt
from tests.any_str import AnyStr

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("resume_style", ["null", "id_map"])
@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("subgraph", [False, True])
@pytest.mark.parametrize("checkpoint_config", [False, True])
@pytest.mark.parametrize("task_depth", [0, 1, 2])
def test_update_state_preserves_interrupts(
    sync_checkpointer: BaseCheckpointSaver,
    durability: Durability,
    resume_style: str,
    bulk: bool,
    subgraph: bool,
    checkpoint_config: bool,
    task_depth: int,
) -> None:
    class State(TypedDict):
        x: int
        note: str

    def approve(state: State):
        first = interrupt({"x": state["x"]}, response_schema=Decision)
        second = interrupt("second approval")
        return {
            "x": state["x"] + 1,
            "note": f"{state['note']}:{first.approved}:{second}",
        }

    approve_task = task(approve)
    calls: list[str] = []

    @task
    def before_approval():
        calls.append("before")
        return "cached"

    @task
    def nested_approval(state: State):
        return approve_task(state).result()

    def gate(state: State):
        if task_depth:
            assert before_approval().result() == "cached"
            worker = nested_approval if task_depth == 2 else approve_task
            return worker(state).result()
        return approve(state)

    builder = StateGraph(State).add_node("gate", gate).add_edge(START, "gate")
    if subgraph:
        builder = (
            StateGraph(State)
            .add_node("gate", builder.compile())
            .add_edge(START, "gate")
        )
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    invocation_config = config
    graph.invoke({"x": 1, "note": ""}, config, durability=durability)

    def update(values: dict[str, Any]) -> None:
        nonlocal invocation_config
        before = graph.get_state(config)
        if bulk:
            updated_config = graph.bulk_update_state(config, [[StateUpdate(values)]])
        else:
            updated_config = graph.update_state(config, values)
        after = graph.get_state(config)
        assert after.interrupts == before.interrupts
        assert after.tasks[0].interrupts == before.tasks[0].interrupts
        assert after.next == ("gate",)
        assert graph.get_state(updated_config).interrupts == before.interrupts
        assert graph.get_state(before.config).interrupts == before.interrupts
        invocation_config = updated_config if checkpoint_config else config

    def resume(value: Any) -> Command:
        pending = graph.get_state(config).interrupts[0]
        return Command(resume=value if resume_style == "null" else {pending.id: value})

    update({"note": "patched"})
    update({"x": 2})
    graph.invoke(resume({"approved": True}), invocation_config, durability=durability)
    assert graph.get_state(config).interrupts[0].value == "second approval"
    update({"note": "patched again"})
    expected = (
        {"x": 2, "note": ":True:yes"}
        if subgraph
        else {"x": 3, "note": "patched again:True:yes"}
    )
    assert (
        graph.invoke(resume("yes"), invocation_config, durability=durability)
        == expected
    )
    assert graph.get_state(config).interrupts == ()
    assert graph.get_state(config).next == ()
    if not subgraph and not checkpoint_config:
        assert calls == (["before"] if task_depth else [])


@pytest.mark.parametrize("resume_style", ["null", "id_map"])
@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("subgraph", [False, True])
@pytest.mark.parametrize("checkpoint_config", [False, True])
@pytest.mark.parametrize("task_depth", [0, 1, 2])
async def test_update_state_preserves_interrupts_async(
    async_checkpointer: BaseCheckpointSaver,
    durability: Durability,
    resume_style: str,
    bulk: bool,
    subgraph: bool,
    checkpoint_config: bool,
    task_depth: int,
) -> None:
    class State(TypedDict):
        x: int
        note: str

    async def approve(state: State):
        first = interrupt({"x": state["x"]}, response_schema=Decision)
        second = interrupt("second approval")
        return {
            "x": state["x"] + 1,
            "note": f"{state['note']}:{first.approved}:{second}",
        }

    approve_task = task(approve)
    calls: list[str] = []

    @task
    async def before_approval():
        calls.append("before")
        return "cached"

    @task
    async def nested_approval(state: State):
        return await approve_task(state)

    async def gate(state: State):
        if task_depth:
            assert await before_approval() == "cached"
            worker = nested_approval if task_depth == 2 else approve_task
            return await worker(state)
        return await approve(state)

    builder = StateGraph(State).add_node("gate", gate).add_edge(START, "gate")
    if subgraph:
        builder = (
            StateGraph(State)
            .add_node("gate", builder.compile())
            .add_edge(START, "gate")
        )
    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    invocation_config = config
    await graph.ainvoke({"x": 1, "note": ""}, config, durability=durability)

    async def update(values: dict[str, Any]) -> None:
        nonlocal invocation_config
        before = await graph.aget_state(config)
        if bulk:
            updated_config = await graph.abulk_update_state(
                config, [[StateUpdate(values)]]
            )
        else:
            updated_config = await graph.aupdate_state(config, values)
        after = await graph.aget_state(config)
        assert after.interrupts == before.interrupts
        assert after.tasks[0].interrupts == before.tasks[0].interrupts
        assert after.next == ("gate",)
        assert (await graph.aget_state(updated_config)).interrupts == before.interrupts
        assert (await graph.aget_state(before.config)).interrupts == before.interrupts
        invocation_config = updated_config if checkpoint_config else config

    async def resume(value: Any) -> Command:
        pending = (await graph.aget_state(config)).interrupts[0]
        return Command(resume=value if resume_style == "null" else {pending.id: value})

    await update({"note": "patched"})
    await update({"x": 2})
    await graph.ainvoke(
        await resume({"approved": True}), invocation_config, durability=durability
    )
    assert (await graph.aget_state(config)).interrupts[0].value == "second approval"
    await update({"note": "patched again"})
    expected = (
        {"x": 2, "note": ":True:yes"}
        if subgraph
        else {"x": 3, "note": "patched again:True:yes"}
    )
    assert (
        await graph.ainvoke(
            await resume("yes"), invocation_config, durability=durability
        )
        == expected
    )
    assert (await graph.aget_state(config)).interrupts == ()
    assert (await graph.aget_state(config)).next == ()
    if not subgraph and not checkpoint_config:
        assert calls == (["before"] if task_depth else [])


@pytest.mark.parametrize("action", ["resume", "reroute", "end", "copy"])
@pytest.mark.parametrize("send", [False, True])
@pytest.mark.parametrize("tasked", [False, True])
def test_update_state_parallel_interrupts(
    sync_checkpointer: BaseCheckpointSaver, action: str, send: bool, tasked: bool
) -> None:
    class State(TypedDict):
        paused: bool
        left: str
        right: str

    @task
    def ask(side: str):
        return interrupt(side)

    def answer(side: str):
        return ask(side).result() if tasked else interrupt(side)

    builder = StateGraph(State)
    builder.add_node("route", lambda state: None)
    if send:
        builder.add_node("gate", lambda state: {state["side"]: answer(state["side"])})
    else:
        builder.add_node("left", lambda state: {"left": answer("left")})
        builder.add_node("right", lambda state: {"right": answer("right")})
    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda state: (
            (
                [Send("gate", {"side": side}) for side in ("left", "right")]
                if send
                else ["left", "right"]
            )
            if state["paused"]
            else END
        ),
    )
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"paused": True, "left": "", "right": ""}, config)
    before = graph.get_state(config)
    assert len(before.interrupts) == 2

    if action == "resume":
        graph.update_state(config, {"left": "patched"}, as_node="route")
        after = graph.get_state(config)
        assert after.interrupts == before.interrupts
        assert after.next == (("gate", "gate") if send else ("left", "right"))
        assert graph.invoke(
            Command(resume={i.id: f"approved:{i.value}" for i in before.interrupts}),
            config,
        ) == {"paused": True, "left": "approved:left", "right": "approved:right"}
        assert graph.get_state(config).interrupts == ()
    else:
        if action == "reroute":
            graph.update_state(config, {"paused": False}, as_node="route")
        else:
            graph.update_state(
                config, None, as_node=END if action == "end" else "__copy__"
            )
        after = graph.get_state(config)
        assert after.interrupts == ()
        if action == "copy":
            assert after.next == (("gate", "gate") if send else ("left", "right"))
            graph.invoke(None, config)
            assert {i.id for i in graph.get_state(config).interrupts}.isdisjoint(
                i.id for i in before.interrupts
            )
        else:
            assert after.next == ()


@pytest.mark.parametrize("action", ["resume", "reroute", "end", "copy"])
@pytest.mark.parametrize("send", [False, True])
@pytest.mark.parametrize("tasked", [False, True])
async def test_update_state_parallel_interrupts_async(
    async_checkpointer: BaseCheckpointSaver, action: str, send: bool, tasked: bool
) -> None:
    class State(TypedDict):
        paused: bool
        left: str
        right: str

    @task
    def ask(side: str):
        return interrupt(side)

    def answer(side: str):
        return ask(side).result() if tasked else interrupt(side)

    builder = StateGraph(State)
    builder.add_node("route", lambda state: None)
    if send:
        builder.add_node("gate", lambda state: {state["side"]: answer(state["side"])})
    else:
        builder.add_node("left", lambda state: {"left": answer("left")})
        builder.add_node("right", lambda state: {"right": answer("right")})
    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda state: (
            (
                [Send("gate", {"side": side}) for side in ("left", "right")]
                if send
                else ["left", "right"]
            )
            if state["paused"]
            else END
        ),
    )
    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    await graph.ainvoke({"paused": True, "left": "", "right": ""}, config)
    before = await graph.aget_state(config)
    assert len(before.interrupts) == 2

    if action == "resume":
        await graph.aupdate_state(config, {"left": "patched"}, as_node="route")
        after = await graph.aget_state(config)
        assert after.interrupts == before.interrupts
        assert after.next == (("gate", "gate") if send else ("left", "right"))
        assert await graph.ainvoke(
            Command(resume={i.id: f"approved:{i.value}" for i in before.interrupts}),
            config,
        ) == {"paused": True, "left": "approved:left", "right": "approved:right"}
        assert (await graph.aget_state(config)).interrupts == ()
    else:
        if action == "reroute":
            await graph.aupdate_state(config, {"paused": False}, as_node="route")
        else:
            await graph.aupdate_state(
                config, None, as_node=END if action == "end" else "__copy__"
            )
        after = await graph.aget_state(config)
        assert after.interrupts == ()
        if action == "copy":
            assert after.next == (("gate", "gate") if send else ("left", "right"))
            await graph.ainvoke(None, config)
            assert {
                i.id for i in (await graph.aget_state(config)).interrupts
            }.isdisjoint(i.id for i in before.interrupts)
        else:
            assert after.next == ()


def test_interruption_without_state_updates(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """Test interruption without state updates. This test confirms that
    interrupting doesn't require a state key having been updated in the prev step"""

    class State(TypedDict):
        input: str

    def noop(_state):
        pass

    builder = StateGraph(State)
    builder.add_node("step_1", noop)
    builder.add_node("step_2", noop)
    builder.add_node("step_3", noop)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_after="*")

    initial_input = {"input": "hello world"}
    thread = {"configurable": {"thread_id": "1"}}

    graph.invoke(initial_input, thread, durability=durability)
    assert graph.get_state(thread).next == ("step_2",)
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (3 if durability != "exit" else 1)

    graph.invoke(None, thread, durability=durability)
    assert graph.get_state(thread).next == ("step_3",)
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (4 if durability != "exit" else 2)

    graph.invoke(None, thread, durability=durability)
    assert graph.get_state(thread).next == ()
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (5 if durability != "exit" else 3)


async def test_interruption_without_state_updates_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """Test interruption without state updates. This test confirms that
    interrupting doesn't require a state key having been updated in the prev step"""

    class State(TypedDict):
        input: str

    async def noop(_state):
        pass

    builder = StateGraph(State)
    builder.add_node("step_1", noop)
    builder.add_node("step_2", noop)
    builder.add_node("step_3", noop)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    graph = builder.compile(checkpointer=async_checkpointer, interrupt_after="*")

    initial_input = {"input": "hello world"}
    thread = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke(initial_input, thread, durability=durability)
    assert (await graph.aget_state(thread)).next == ("step_2",)
    n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
    assert n_checkpoints == (3 if durability != "exit" else 1)

    await graph.ainvoke(None, thread, durability=durability)
    assert (await graph.aget_state(thread)).next == ("step_3",)
    n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
    assert n_checkpoints == (4 if durability != "exit" else 2)

    await graph.ainvoke(None, thread, durability=durability)
    assert (await graph.aget_state(thread)).next == ()
    n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
    assert n_checkpoints == (5 if durability != "exit" else 3)


class Decision(BaseModel):
    approved: bool
    note: str | None = None


class DecisionDict(TypedDict):
    approved: bool


@dataclass
class DecisionData:
    approved: bool


RAW_SCHEMA = {"type": "object", "properties": {"approved": {"type": "boolean"}}}


@pytest.mark.parametrize(
    ("response_schema", "expected_schema", "expected_answer"),
    [
        (None, None, {"approved": True, "extra": 1}),
        (RAW_SCHEMA, RAW_SCHEMA, {"approved": True, "extra": 1}),
        (Decision, Decision.model_json_schema(), Decision(approved=True)),
        (
            DecisionDict,
            {
                "properties": {"approved": {"title": "Approved", "type": "boolean"}},
                "required": ["approved"],
                "title": "DecisionDict",
                "type": "object",
            },
            {"approved": True},
        ),
        (
            DecisionData,
            {
                "properties": {"approved": {"title": "Approved", "type": "boolean"}},
                "required": ["approved"],
                "title": "DecisionData",
                "type": "object",
            },
            DecisionData(approved=True),
        ),
    ],
    ids=["none", "raw_dict", "pydantic", "typeddict", "dataclass"],
)
def test_interrupt_response_schema(
    sync_checkpointer: BaseCheckpointSaver,
    response_schema: Any,
    expected_schema: dict[str, Any] | None,
    expected_answer: Any,
) -> None:
    class State(TypedDict):
        answer: Any

    def node(state: State) -> State:
        return {
            "answer": interrupt(
                {"question": "approve?"}, response_schema=response_schema
            )
        }

    graph = (
        StateGraph(State)
        .add_node("node", node)
        .add_edge(START, "node")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}
    expected = Interrupt(
        value={"question": "approve?"}, id=AnyStr(), response_schema=expected_schema
    )

    assert list(graph.stream({"answer": None}, config)) == [
        {"__interrupt__": (expected,)}
    ]
    assert graph.get_state(config).tasks[0].interrupts == (expected,)
    assert graph.invoke(Command(resume={"approved": True, "extra": 1}), config) == {
        "answer": expected_answer
    }


@pytest.mark.parametrize("resume_style", ["null", "map"])
def test_interrupt_response_schema_rejects_invalid_resume(
    sync_checkpointer: BaseCheckpointSaver, resume_style: str
) -> None:
    class State(TypedDict):
        answer: Any

    def node(state: State) -> State:
        return {"answer": interrupt("approve?", response_schema=Decision)}

    graph = (
        StateGraph(State)
        .add_node("node", node)
        .add_edge(START, "node")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"answer": None}, config)
    [pending] = graph.get_state(config).tasks[0].interrupts

    def resume(value: dict[str, Any]) -> Command:
        return Command(resume=value if resume_style == "null" else {pending.id: value})

    with pytest.raises(ValidationError, match="approved"):
        graph.invoke(resume({"approved": "nope"}), config)

    assert graph.invoke(resume({"approved": False}), config) == {
        "answer": Decision(approved=False)
    }


@pytest.mark.parametrize("resume_style", ["null", "id_map"])
def test_interrupt_response_schema_invalid_resume_after_earlier_interrupt(
    sync_checkpointer: BaseCheckpointSaver, resume_style: str
) -> None:
    class State(TypedDict):
        answer: Any

    def node(state: State) -> State:
        first = interrupt("first")
        second = interrupt("approve?", response_schema=Decision)
        return {"answer": [first, second]}

    graph = (
        StateGraph(State)
        .add_node("node", node)
        .add_edge(START, "node")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"answer": None}, config)
    graph.invoke(Command(resume="ok"), config)
    [pending] = graph.get_state(config).tasks[0].interrupts

    def resume(value: dict[str, Any]) -> Command:
        return Command(resume=value if resume_style == "null" else {pending.id: value})

    with pytest.raises(ValidationError, match="approved"):
        graph.invoke(resume({"approved": "nope"}), config)

    assert graph.invoke(resume({"approved": True}), config) == {
        "answer": ["ok", Decision(approved=True)]
    }
