import operator
import sys
from typing import Annotated

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Durability, Send, interrupt

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


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

    # Global counter to track node executions
    node_counter = {"entry": 0, "map_node": 0}

    class State(TypedDict):
        items: list[str]
        processed: Annotated[list[str], operator.add]

    def entry_node(state: State):
        node_counter["entry"] += 1
        return {}  # No state updates in entry node

    def send_to_map(state: State):
        return [Send("map_node", {"item": item}) for item in state["items"]]

    def map_node(state: State):
        node_counter["map_node"] += 1
        if "dangerous" in state["item"]:
            value = interrupt({"processing": state["item"]})
            return {"processed": [f"processed_{value}"]}
        else:
            return {"processed": [f"processed_{state['item']}_auto"]}

    builder = StateGraph(State)
    builder.add_node("entry", entry_node)
    builder.add_node("map_node", map_node)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges("entry", send_to_map, ["map_node"])
    builder.add_edge("map_node", END)

    graph = builder.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "test_interrupt_send"}}

    # Run until interrupts
    result = graph.invoke(
        {"items": ["item1", "dangerous_item1", "dangerous_item2"]}, config=config
    )

    # Verify we have interrupts (only one for dangerous_item)
    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) == 2
    assert "dangerous_item" in interrupts[0].value["processing"]

    # Resume with mapping of interrupt IDs to values
    resume_map = {i.id: f"human_input_{i.value['processing']}" for i in interrupts}

    final_result = graph.invoke(Command(resume=resume_map), config=config)

    # Verify final result contains processed items
    assert "processed" in final_result
    processed_items = final_result["processed"]
    assert len(processed_items) == 3
    assert "processed_item1_auto" in processed_items  # item1 processed automatically
    assert any(
        "processed_human_input_dangerous_item1" in item for item in processed_items
    )  # dangerous_item1 processed after interrupt
    assert any(
        "processed_human_input_dangerous_item2" in item for item in processed_items
    )  # dangerous_item2 processed after interrupt

    # Verify node execution counts
    assert node_counter["entry"] == 1  # Entry node runs once
    # Map node runs 3 times initially (item1 completes, 2 dangerous_items interrupt),
    # then 2 times on resume
    assert node_counter["map_node"] == 5


async def test_interrupt_with_send_payloads_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """Test interruption in map node with Send payloads and human-in-the-loop resume."""

    # Global counter to track node executions
    node_counter = {"entry": 0, "map_node": 0}

    class State(TypedDict):
        items: list[str]
        processed: Annotated[list[str], operator.add]

    def entry_node(state: State):
        node_counter["entry"] += 1
        return {}  # No state updates in entry node

    def send_to_map(state: State):
        return [Send("map_node", {"item": item}) for item in state["items"]]

    def map_node(state: State):
        node_counter["map_node"] += 1
        if "dangerous" in state["item"]:
            value = interrupt({"processing": state["item"]})
            return {"processed": [f"processed_{value}"]}
        else:
            return {"processed": [f"processed_{state['item']}_auto"]}

    builder = StateGraph(State)
    builder.add_node("entry", entry_node)
    builder.add_node("map_node", map_node)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges("entry", send_to_map, ["map_node"])
    builder.add_edge("map_node", END)

    graph = builder.compile(checkpointer=async_checkpointer)

    config = {"configurable": {"thread_id": "test_interrupt_send"}}

    # Run until interrupts
    result = await graph.ainvoke(
        {"items": ["item1", "dangerous_item1", "dangerous_item2"]}, config=config
    )

    # Verify we have interrupts (only one for dangerous_item)
    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) == 2
    assert "dangerous_item" in interrupts[0].value["processing"]

    # Resume with mapping of interrupt IDs to values
    resume_map = {i.id: f"human_input_{i.value['processing']}" for i in interrupts}

    final_result = await graph.ainvoke(Command(resume=resume_map), config=config)

    # Verify final result contains processed items
    assert "processed" in final_result
    processed_items = final_result["processed"]
    assert len(processed_items) == 3
    assert "processed_item1_auto" in processed_items  # item1 processed automatically
    assert any(
        "processed_human_input_dangerous_item1" in item for item in processed_items
    )  # dangerous_item1 processed after interrupt
    assert any(
        "processed_human_input_dangerous_item2" in item for item in processed_items
    )  # dangerous_item2 processed after interrupt

    # Verify node execution counts
    assert node_counter["entry"] == 1  # Entry node runs once
    # Map node runs 3 times initially (item1 completes, 2 dangerous_items interrupt),
    # then 2 times on resume
    assert node_counter["map_node"] == 5


def test_interrupt_with_send_payloads_sequential_resume(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test interruption in map node with Send payloads and sequential resume."""

    # Global counter to track node executions
    node_counter = {"entry": 0, "map_node": 0}

    class State(TypedDict):
        items: list[str]
        processed: Annotated[list[str], operator.add]

    def entry_node(state: State):
        node_counter["entry"] += 1
        return {}  # No state updates in entry node

    def send_to_map(state: State):
        return [Send("map_node", {"item": item}) for item in state["items"]]

    def map_node(state: State):
        node_counter["map_node"] += 1
        if "dangerous" in state["item"]:
            value = interrupt({"processing": state["item"]})
            return {"processed": [f"processed_{value}"]}
        else:
            return {"processed": [f"processed_{state['item']}_auto"]}

    builder = StateGraph(State)
    builder.add_node("entry", entry_node)
    builder.add_node("map_node", map_node)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges("entry", send_to_map, ["map_node"])
    builder.add_edge("map_node", END)

    graph = builder.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "test_interrupt_send_sequential"}}

    # Run until interrupts
    result = graph.invoke(
        {"items": ["item1", "dangerous_item1", "dangerous_item2"]}, config=config
    )

    # Verify we have interrupts
    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) == 2
    assert "dangerous_item" in interrupts[0].value["processing"]

    # Resume first interrupt only
    first_interrupt = interrupts[0]
    first_resume_map = {
        first_interrupt.id: f"human_input_{first_interrupt.value['processing']}"
    }

    partial_result = graph.invoke(Command(resume=first_resume_map), config=config)

    # Verify we still have one pending interrupt
    remaining_interrupts = partial_result.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1

    # Resume second interrupt
    second_interrupt = remaining_interrupts[0]
    second_resume_map = {
        second_interrupt.id: f"human_input_{second_interrupt.value['processing']}"
    }

    final_result = graph.invoke(Command(resume=second_resume_map), config=config)

    # Verify final result contains processed items
    assert "processed" in final_result
    processed_items = final_result["processed"]
    assert len(processed_items) == 3
    assert "processed_item1_auto" in processed_items  # item1 processed automatically
    assert any(
        "processed_human_input_dangerous_item1" in item for item in processed_items
    )  # dangerous_item1 processed after interrupt
    assert any(
        "processed_human_input_dangerous_item2" in item for item in processed_items
    )  # dangerous_item2 processed after interrupt

    # Verify node execution counts
    assert node_counter["entry"] == 1  # Entry node runs once
    # Map node runs 3 times initially (item1 completes, 2 dangerous_items interrupt),
    # then 1 time on first resume, then 1 time on second resume
    assert node_counter["map_node"] == 5


async def test_interrupt_with_send_payloads_sequential_resume_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test interruption in map node with Send payloads and sequential resume."""

    # Global counter to track node executions
    node_counter = {"entry": 0, "map_node": 0}

    class State(TypedDict):
        items: list[str]
        processed: Annotated[list[str], operator.add]

    def entry_node(state: State):
        node_counter["entry"] += 1
        return {}  # No state updates in entry node

    def send_to_map(state: State):
        return [Send("map_node", {"item": item}) for item in state["items"]]

    def map_node(state: State):
        node_counter["map_node"] += 1
        if "dangerous" in state["item"]:
            value = interrupt({"processing": state["item"]})
            return {"processed": [f"processed_{value}"]}
        else:
            return {"processed": [f"processed_{state['item']}_auto"]}

    builder = StateGraph(State)
    builder.add_node("entry", entry_node)
    builder.add_node("map_node", map_node)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges("entry", send_to_map, ["map_node"])
    builder.add_edge("map_node", END)

    graph = builder.compile(checkpointer=async_checkpointer)

    config = {"configurable": {"thread_id": "test_interrupt_send_sequential"}}

    # Run until interrupts
    result = await graph.ainvoke(
        {"items": ["item1", "dangerous_item1", "dangerous_item2"]}, config=config
    )

    # Verify we have interrupts
    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) == 2
    assert "dangerous_item" in interrupts[0].value["processing"]

    # Resume first interrupt only
    first_interrupt = interrupts[0]
    first_resume_map = {
        first_interrupt.id: f"human_input_{first_interrupt.value['processing']}"
    }

    partial_result = await graph.ainvoke(
        Command(resume=first_resume_map), config=config
    )

    # Verify we still have one pending interrupt
    remaining_interrupts = partial_result.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1

    # Resume second interrupt
    second_interrupt = remaining_interrupts[0]
    second_resume_map = {
        second_interrupt.id: f"human_input_{second_interrupt.value['processing']}"
    }

    final_result = await graph.ainvoke(Command(resume=second_resume_map), config=config)

    # Verify final result contains processed items
    assert "processed" in final_result
    processed_items = final_result["processed"]
    assert len(processed_items) == 3
    assert "processed_item1_auto" in processed_items  # item1 processed automatically
    assert any(
        "processed_human_input_dangerous_item1" in item for item in processed_items
    )  # dangerous_item1 processed after interrupt
    assert any(
        "processed_human_input_dangerous_item2" in item for item in processed_items
    )  # dangerous_item2 processed after interrupt

    # Verify node execution counts
    assert node_counter["entry"] == 1  # Entry node runs once
    # Map node runs 3 times initially (item1 completes, 2 dangerous_items interrupt),
    # then 1 time on first resume, then 1 time on second resume
    assert node_counter["map_node"] == 5


def test_node_with_multiple_interrupts_requires_full_resume(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test a number of different resume patterns for a node with multiple interrupts,

    Ensures that a node is not re-executed until valid resume values have been provided to all
    discovered interrupts"""

    node_counter = 0

    class State(TypedDict):
        input: str

    def double_interrupt_node(state: State):
        nonlocal node_counter
        node_counter += 1
        first = interrupt("first")
        second = interrupt("second")
        third = interrupt("third")
        return {"input": f"{first}-{second}-{third}"}

    builder = StateGraph(State)
    builder.add_node("double_interrupt", double_interrupt_node)
    builder.add_edge(START, "double_interrupt")
    builder.add_edge("double_interrupt", END)

    graph = builder.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "test_double_interrupt"}}

    result = graph.invoke({"input": "start"}, config=config)

    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) == 1
    first_interrupt = interrupts[0]
    assert node_counter == 1

    # invoke with an interrupt map that matches double_interrupt_node.
    # this should execute the node
    partial = graph.invoke(
        Command(resume={first_interrupt.id: "human_first"}), config=config
    )
    remaining_interrupts = partial.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1
    assert remaining_interrupts[0].value == "second"
    assert node_counter == 2

    # invoke with an interrupt map that DOES NOT match double_interrupt_node.
    # this should not execute the node because the optimization kicks in
    partial = graph.invoke(
        Command(resume={"00000000000000000000000000000000": "nothing_burger"}),
        config=config,
    )
    remaining_interrupts = partial.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1
    assert remaining_interrupts[0].value == "second"
    assert node_counter == 2

    # invoke with None resume. this should execute the node
    partial = graph.invoke(None, config=config)
    remaining_interrupts = partial.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1
    assert remaining_interrupts[0].value == "second"
    assert node_counter == 3

    # invoke with nonspecific resume. this should execute the node
    partial = graph.invoke(Command(resume="human_second"), config=config)
    remaining_interrupts = partial.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1
    print("REMAINING INTERRUPTS: ", remaining_interrupts)
    assert remaining_interrupts[0].value == "third"
    assert node_counter == 4

    # finally, invoke with an interrupt map that matches double_interrupt_node.
    # this should execute the node and all interrupts should be resolved
    final_result = graph.invoke(Command(resume="human_third"), config=config)
    assert "input" in final_result
    assert final_result["input"] == "human_first-human_second-human_third"
    assert node_counter == 5


@NEEDS_CONTEXTVARS
async def test_node_with_multiple_interrupts_requires_full_resume_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test a number of different resume patterns for a node with multiple interrupts,

    Ensures that a node is not re-executed until valid resume values have been provided to all
    discovered interrupts"""

    node_counter = 0

    class State(TypedDict):
        input: str

    def double_interrupt_node(state: State):
        nonlocal node_counter
        node_counter += 1
        first = interrupt("first")
        second = interrupt("second")
        third = interrupt("third")
        return {"input": f"{first}-{second}-{third}"}

    builder = StateGraph(State)
    builder.add_node("double_interrupt", double_interrupt_node)
    builder.add_edge(START, "double_interrupt")
    builder.add_edge("double_interrupt", END)

    graph = builder.compile(checkpointer=async_checkpointer)

    config = {"configurable": {"thread_id": "test_double_interrupt"}}

    result = await graph.ainvoke({"input": "start"}, config=config)

    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) == 1
    first_interrupt = interrupts[0]
    assert node_counter == 1

    # invoke with an interrupt map that matches double_interrupt_node.
    # this should execute the node
    partial = await graph.ainvoke(
        Command(resume={first_interrupt.id: "human_first"}), config=config
    )
    remaining_interrupts = partial.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1
    assert remaining_interrupts[0].value == "second"
    assert node_counter == 2

    # invoke with an interrupt map that DOES NOT match double_interrupt_node.
    # this should not execute the node because the optimization kicks in
    partial = await graph.ainvoke(
        Command(resume={"00000000000000000000000000000000": "nothing_burger"}),
        config=config,
    )
    remaining_interrupts = partial.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1
    assert remaining_interrupts[0].value == "second"
    assert node_counter == 2

    # invoke with None resume. this should execute the node
    partial = await graph.ainvoke(None, config=config)
    remaining_interrupts = partial.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1
    assert remaining_interrupts[0].value == "second"
    assert node_counter == 3

    # invoke with nonspecific resume. this should execute the node
    partial = await graph.ainvoke(Command(resume="human_second"), config=config)
    remaining_interrupts = partial.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1
    print("REMAINING INTERRUPTS: ", remaining_interrupts)
    assert remaining_interrupts[0].value == "third"
    assert node_counter == 4

    # finally, invoke with an interrupt map that matches double_interrupt_node.
    # this should execute the node and all interrupts should be resolved
    final_result = await graph.ainvoke(Command(resume="human_third"), config=config)
    assert "input" in final_result
    assert final_result["input"] == "human_first-human_second-human_third"
    assert node_counter == 5
