from typing import TypedDict

from langgraph_checkpoint.checkpoint.memory import MemorySaver

from langgraph.graph import END, START, StateGraph


def test_interruption_without_state_updates():
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

    memory = MemorySaver()

    graph = builder.compile(checkpointer=memory, interrupt_after="*")

    initial_input = {"input": "hello world"}
    thread = {"configurable": {"thread_id": "1"}}

    graph.invoke(initial_input, thread, debug=True)
    assert graph.get_state(thread).next == ("step_2",)

    graph.invoke(None, thread, debug=True)
    assert graph.get_state(thread).next == ("step_3",)

    graph.invoke(None, thread, debug=True)
    assert graph.get_state(thread).next == ()


async def test_interruption_without_state_updates_async():
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

    memory = MemorySaver()

    graph = builder.compile(checkpointer=memory, interrupt_after="*")

    initial_input = {"input": "hello world"}
    thread = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke(initial_input, thread, debug=True)
    assert (await graph.aget_state(thread)).next == ("step_2",)

    await graph.ainvoke(None, thread, debug=True)
    assert (await graph.aget_state(thread)).next == ("step_3",)

    await graph.ainvoke(None, thread, debug=True)
    assert (await graph.aget_state(thread)).next == ()
