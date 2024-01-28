import os

import pytest
from pydantic import BaseModel

from langgraph.graph import END, StateGraph
from langgraph.graph.helpers import StateGraphDrawer


@pytest.fixture
def drawer() -> StateGraphDrawer:
    return StateGraphDrawer()

@pytest.fixture
def drawer_with_label_overrides() -> StateGraphDrawer:
    return StateGraphDrawer(
        label_overrides={
            "nodes": {
                "node1": "Start Node",
                "node2": "CustomLabel2",
                "__end__": "End Node"
            },
            "conditional_edges": {
                "should_continue": "ConditionLabel",
                "should_continue2": "ConditionLabel2",
            },
            "edges": {
                "continue": "ContinueLabel",
                "end": "EndLabel"
            }
        }
    )

@pytest.fixture
def state_graph() -> StateGraph:
    def should_continue(self, state): pass
    def should_continue2(self, state): pass
    def mock_call_function(self, state): pass

    graph = StateGraph(BaseModel)

    graph.add_node("node1", mock_call_function)
    graph.add_node("node2", mock_call_function)
    graph.add_node("node3", mock_call_function)
    graph.add_node("node4", mock_call_function)
    graph.add_node("node5", mock_call_function)

    graph.add_edge("node1", "node2")
    graph.add_conditional_edges("node2", should_continue, {"go_to_3": "node3", "go_to_4": "node4"})
    graph.add_edge("node3", "node4")
    graph.add_conditional_edges("node4", should_continue2, {"shortcut": END, "go_to_5": "node5"})
    graph.add_edge("node5", END)

    graph.set_entry_point("node1")
    graph.compile()

    return graph

def test_static_methods(drawer: StateGraphDrawer):
    node_label = drawer.get_node_label("node1")
    condition_label = drawer.get_conditional_edge_label("should_continue")
    edge_label = drawer.get_edge_label("continue")
    
    assert node_label == "<<B>node1</B>>"
    assert condition_label == "<<I>should_continue</I>>"
    assert edge_label == "<<U>continue</U>>"

def test_labels_override():
    drawer = StateGraphDrawer(
        label_overrides={
            "nodes": {
                "node1": "Start Node",
                "node2": "CustomLabel2",
                "__end__": "End Node"
            },
            "conditional_edges": {
                "should_continue": "ConditionLabel",
                "should_continue2": "ConditionLabel2",
            },
            "edges": {
                "continue": "ContinueLabel",
                "end": "EndLabel"
            }
        }
    )
    node_label = drawer.get_node_label("node1")
    condition_label = drawer.get_conditional_edge_label("should_continue")
    edge_label = drawer.get_edge_label("continue")

    assert node_label == "<<B>Start Node</B>>"
    assert condition_label == "<<I>ConditionLabel</I>>"
    assert edge_label == "<<U>ContinueLabel</U>>"

def test_state_graph_drawer(
    drawer: StateGraphDrawer,
    state_graph: StateGraph
):
    drawer.draw(state_graph, output_file_path='graph.png')

    # Check file has been created and is not empty
    assert os.path.exists('graph.png')
    assert os.path.getsize('graph.png') > 0
    os.remove('graph.png')

def test_state_graph_drawer_with_label_overrides(
    drawer_with_label_overrides: StateGraphDrawer,
    state_graph: StateGraph
):
    drawer_with_label_overrides.draw(state_graph, output_file_path='graph.png')

    # Check file has been created and is not empty
    assert os.path.exists('graph.png')
    assert os.path.getsize('graph.png') > 0
    # s.remove('graph.png')