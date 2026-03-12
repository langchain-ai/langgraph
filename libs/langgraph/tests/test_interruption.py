import pytest
from typing_extensions import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Durability, interrupt

pytestmark = pytest.mark.anyio


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


def test_resume_from_parent_state_config_does_not_rerun_completed_subgraph(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    class State(TypedDict):
        msg: str

    called: list[str] = []

    def agent(state: State) -> State:
        called.append("agent")
        return {"msg": state["msg"] + " agent"}

    def approve(state: State) -> State:
        called.append("approve")
        answer = interrupt("approve?")
        return {"msg": state["msg"] + f" {answer}"}

    middle = (
        StateGraph(State)
        .add_node("agent", agent)
        .add_node("approve", approve)
        .add_edge(START, "agent")
        .add_edge("agent", "approve")
        .add_edge("approve", END)
        .compile(checkpointer=True)
    )

    def after(state: State) -> State:
        called.append("after")
        return {"msg": state["msg"] + " after"}

    graph = (
        StateGraph(State)
        .add_node("middle", middle)
        .add_node("after", after)
        .add_edge(START, "middle")
        .add_edge("middle", "after")
        .add_edge("after", END)
        .compile(checkpointer=sync_checkpointer)
    )

    thread = {"configurable": {"thread_id": "resume-parent-config-sync"}}

    result = graph.invoke({"msg": "hi"}, thread)
    assert result["__interrupt__"][0].value == "approve?"

    parent_state = graph.get_state(thread)

    called.clear()
    resumed = graph.invoke(Command(resume="yes"), parent_state.config)

    assert resumed == {"msg": "hi agent yes after"}
    assert called == ["approve", "after"]


async def test_resume_from_parent_state_config_does_not_rerun_completed_subgraph_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    class State(TypedDict):
        msg: str

    called: list[str] = []

    async def agent(state: State) -> State:
        called.append("agent")
        return {"msg": state["msg"] + " agent"}

    async def approve(state: State) -> State:
        called.append("approve")
        answer = interrupt("approve?")
        return {"msg": state["msg"] + f" {answer}"}

    middle = (
        StateGraph(State)
        .add_node("agent", agent)
        .add_node("approve", approve)
        .add_edge(START, "agent")
        .add_edge("agent", "approve")
        .add_edge("approve", END)
        .compile(checkpointer=True)
    )

    async def after(state: State) -> State:
        called.append("after")
        return {"msg": state["msg"] + " after"}

    graph = (
        StateGraph(State)
        .add_node("middle", middle)
        .add_node("after", after)
        .add_edge(START, "middle")
        .add_edge("middle", "after")
        .add_edge("after", END)
        .compile(checkpointer=async_checkpointer)
    )

    thread = {"configurable": {"thread_id": "resume-parent-config-async"}}

    result = await graph.ainvoke({"msg": "hi"}, thread)
    assert result["__interrupt__"][0].value == "approve?"

    parent_state = await graph.aget_state(thread)

    called.clear()
    resumed = await graph.ainvoke(Command(resume="yes"), parent_state.config)

    assert resumed == {"msg": "hi agent yes after"}
    assert called == ["approve", "after"]


def test_resume_with_empty_checkpoint_id_does_not_rerun_completed_subgraph(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    class State(TypedDict):
        msg: str

    called: list[str] = []

    def agent(state: State) -> State:
        called.append("agent")
        return {"msg": state["msg"] + " agent"}

    def approve(state: State) -> State:
        called.append("approve")
        answer = interrupt("approve?")
        return {"msg": state["msg"] + f" {answer}"}

    middle = (
        StateGraph(State)
        .add_node("agent", agent)
        .add_node("approve", approve)
        .add_edge(START, "agent")
        .add_edge("agent", "approve")
        .add_edge("approve", END)
        .compile(checkpointer=True)
    )

    def after(state: State) -> State:
        called.append("after")
        return {"msg": state["msg"] + " after"}

    graph = (
        StateGraph(State)
        .add_node("middle", middle)
        .add_node("after", after)
        .add_edge(START, "middle")
        .add_edge("middle", "after")
        .add_edge("after", END)
        .compile(checkpointer=sync_checkpointer)
    )

    base_thread = {"configurable": {"thread_id": "resume-empty-checkpoint-sync"}}
    resume_thread = {
        "configurable": {
            "thread_id": "resume-empty-checkpoint-sync",
            "checkpoint_id": "",
        }
    }

    result = graph.invoke({"msg": "hi"}, base_thread)
    assert result["__interrupt__"][0].value == "approve?"

    called.clear()
    resumed = graph.invoke(Command(resume="yes"), resume_thread)

    assert resumed == {"msg": "hi agent yes after"}
    assert called == ["approve", "after"]


async def test_resume_with_empty_checkpoint_id_does_not_rerun_completed_subgraph_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    class State(TypedDict):
        msg: str

    called: list[str] = []

    async def agent(state: State) -> State:
        called.append("agent")
        return {"msg": state["msg"] + " agent"}

    async def approve(state: State) -> State:
        called.append("approve")
        answer = interrupt("approve?")
        return {"msg": state["msg"] + f" {answer}"}

    middle = (
        StateGraph(State)
        .add_node("agent", agent)
        .add_node("approve", approve)
        .add_edge(START, "agent")
        .add_edge("agent", "approve")
        .add_edge("approve", END)
        .compile(checkpointer=True)
    )

    async def after(state: State) -> State:
        called.append("after")
        return {"msg": state["msg"] + " after"}

    graph = (
        StateGraph(State)
        .add_node("middle", middle)
        .add_node("after", after)
        .add_edge(START, "middle")
        .add_edge("middle", "after")
        .add_edge("after", END)
        .compile(checkpointer=async_checkpointer)
    )

    base_thread = {"configurable": {"thread_id": "resume-empty-checkpoint-async"}}
    resume_thread = {
        "configurable": {
            "thread_id": "resume-empty-checkpoint-async",
            "checkpoint_id": "",
        }
    }

    result = await graph.ainvoke({"msg": "hi"}, base_thread)
    assert result["__interrupt__"][0].value == "approve?"

    called.clear()
    resumed = await graph.ainvoke(Command(resume="yes"), resume_thread)

    assert resumed == {"msg": "hi agent yes after"}
    assert called == ["approve", "after"]
