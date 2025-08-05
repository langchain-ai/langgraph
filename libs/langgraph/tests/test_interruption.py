import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict, Annotated
import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt, Command, Durability

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


def test_interrupt_with_send_payloads(sync_checkpointer: BaseCheckpointSaver) -> None:
    """Test interruption in map node with Send payloads and human-in-the-loop resume."""
    
    class State(TypedDict):
        items: list[str]
        processed: Annotated[list[str], operator.add]
    
    def entry_node(state: State):
        return {"items": ["item1", "item2"]}
    
    def send_to_map(state: State):
        return [Send("map_node", {"item": item}) for item in state["items"]]
    
    def map_node(state: State):
        value = interrupt({"processing": state["item"]})
        return {"processed": [f"processed_{value}"]}
    
    builder = StateGraph(State)
    builder.add_node("entry", entry_node)
    builder.add_node("map_node", map_node)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges("entry", send_to_map, ["map_node"])
    builder.add_edge("map_node", END)
    
    graph = builder.compile(checkpointer=sync_checkpointer)
    
    config = {"configurable": {"thread_id": "test_interrupt_send"}}
    
    # Run until interrupts
    result = graph.invoke({"items": [], "processed": []}, config=config)
    
    # Verify we have interrupts
    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) == 2
    assert all(i.resumable for i in interrupts)
    
    # Resume with mapping of interrupt IDs to values
    resume_map = {
        i.interrupt_id: f"human_input_{i.value['processing']}"
        for i in interrupts
    }
    
    final_result = graph.invoke(Command(resume=resume_map), config=config)
    
    # Verify final result contains processed items
    assert "processed" in final_result
    processed_items = final_result["processed"]
    assert len(processed_items) == 2
    assert "processed_human_input_item1" in processed_items
    assert "processed_human_input_item2" in processed_items
