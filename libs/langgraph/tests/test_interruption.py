from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Durability, Interrupt, interrupt
from tests.any_str import AnyStr

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


class Decision(BaseModel):
    approved: bool
    note: str | None = None


class DecisionDict(TypedDict):
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
    ],
    ids=["none", "raw_dict", "pydantic", "typeddict"],
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
