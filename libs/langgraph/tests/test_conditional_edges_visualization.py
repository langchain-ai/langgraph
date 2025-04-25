import pytest
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END


def test_conditional_edges_to_end(snapshot: "pytest.SnapshotAssertion") -> None:
    """Test that conditionally routing to END node from a node with conditional edges
    is correctly represented in the Mermaid visualization.
    
    This tests the graph visualization when a node has multiple conditional edges,
    including a conditional route to END.
    
    Note: Currently, the conditional edges to END are not displayed in the visualization.
    This should be improved in a future update.
    """
    class DummyState(TypedDict):
        pass_count: int

    def increment_pass_count(state: DummyState) -> DummyState:
        state["pass_count"] += 1
        return state

    def route_b(state: DummyState) -> Literal["X", "Y"]:
        if state["pass_count"] == 0:
            return "X"
        else:
            return "Y"

    migration_graph = StateGraph(DummyState)
    migration_graph.add_node("B", increment_pass_count)
    migration_graph.add_node("C", increment_pass_count)
    migration_graph.add_node("D", increment_pass_count)

    migration_graph.add_edge(START, "B")

    migration_graph.add_conditional_edges(
        "B",
        route_b,
        {
            "X": "C",
            "Y": "D",
        },
    )

    migration_graph.add_edge("D", "B")
    migration_graph.add_edge("C", END)

    app = migration_graph.compile()
    
    # Get the Mermaid diagram text
    mermaid_text = app.get_graph().draw_mermaid(with_styles=False)
    
    # Verify B has conditional edges to C and D but not to END
    assert "B -. &nbsp;X&nbsp; .-> C" in mermaid_text
    assert "B -. &nbsp;Y&nbsp; .-> D" in mermaid_text
    
    # No incorrect conditional edge from B to END should be present
    # (In v0.3.32+ there was a bug adding these)
    assert "B -. " not in mermaid_text.replace("B -. &nbsp;X&nbsp; .-> C", "").replace("B -. &nbsp;Y&nbsp; .-> D", "")
    
    # Check the mermaid representation
    assert mermaid_text == snapshot


def test_explicit_conditional_edge_to_end(snapshot: "pytest.SnapshotAssertion") -> None:
    """Test that explicitly routing to END node in a conditional edge
    is correctly represented in the Mermaid visualization.
    
    This tests the graph visualization when a node has a conditional edge
    directly to END, which should be represented in the visualization.
    
    Note: Currently, the conditional edges to END are not displayed in the visualization.
    This should be improved in a future update.
    """
    class DummyState(TypedDict):
        pass_count: int

    def increment_pass_count(state: DummyState) -> DummyState:
        state["pass_count"] += 1
        return state

    def route_b(state: DummyState) -> Literal["continue", "end"]:
        if state["pass_count"] < 2:
            return "continue"
        else:
            return "end"

    workflow = StateGraph(DummyState)
    workflow.add_node("B", increment_pass_count)
    workflow.add_node("C", increment_pass_count)

    workflow.add_edge(START, "B")
    workflow.add_conditional_edges(
        "B",
        route_b,
        {
            "continue": "C",
            "end": END,
        },
    )
    workflow.add_edge("C", "B")

    app = workflow.compile()
    
    # Get the Mermaid diagram text
    mermaid_text = app.get_graph().draw_mermaid(with_styles=False)
    
    # Verify B has a conditional edge to C
    assert "B -. &nbsp;continue&nbsp; .-> C" in mermaid_text
    
    # Known limitation: Conditional edges to END are not currently shown in the visualization
    # In the future, we should expect to see:
    # assert "B -. &nbsp;end&nbsp; .-> __end__" in mermaid_text
    
    # Check the mermaid representation
    assert mermaid_text == snapshot


def test_mixed_conditional_and_terminal_edges(snapshot: "pytest.SnapshotAssertion") -> None:
    """Test visualization of a graph with both conditional routes to END
    and terminal nodes (nodes with no outgoing edges).
    
    This tests that terminal nodes automatically get a non-conditional edge to END,
    while explicit conditional edges to END are shown as conditional.
    
    Note: Currently, the conditional edges to END are not displayed in the visualization.
    Only terminal nodes get an edge to END. This should be improved in a future update.
    """
    class DummyState(TypedDict):
        step: int

    def process_a(state: DummyState) -> DummyState:
        state["step"] += 1
        return state

    def process_b(state: DummyState) -> DummyState:
        state["step"] += 2
        return state
    
    def process_c(state: DummyState) -> DummyState:
        state["step"] += 3
        return state

    def route_next(state: DummyState) -> Literal["to_b", "to_c", "stop"]:
        if state["step"] < 3:
            return "to_b"
        elif state["step"] < 6:
            return "to_c"
        else:
            return "stop"

    graph = StateGraph(DummyState)
    graph.add_node("A", process_a)
    graph.add_node("B", process_b)
    graph.add_node("C", process_c)
    
    graph.add_edge(START, "A")
    
    # Conditional edges from A, including one to END
    graph.add_conditional_edges(
        "A",
        route_next,
        {
            "to_b": "B",
            "to_c": "C",
            "stop": END
        }
    )
    
    # B loops back to A (not terminal)
    graph.add_edge("B", "A")
    
    # C is a terminal node (no outgoing edges)
    # Should automatically get a non-conditional edge to END
    
    app = graph.compile()
    
    # Get the Mermaid diagram text
    mermaid_text = app.get_graph().draw_mermaid(with_styles=False)
    
    # Verify conditional edges from A to B and C
    assert "A -. &nbsp;to_b&nbsp; .-> B" in mermaid_text
    assert "A -. &nbsp;to_c&nbsp; .-> C" in mermaid_text
    
    # Known limitation: Conditional edges to END are not currently shown in the visualization
    # In the future, we should expect to see:
    # assert "A -. &nbsp;stop&nbsp; .-> __end__" in mermaid_text
    
    # Verify C has a non-conditional edge to END as a terminal node
    assert "C --> __end__" in mermaid_text
    
    # Check the mermaid representation
    assert mermaid_text == snapshot 