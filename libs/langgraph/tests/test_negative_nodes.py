"""Test suite for negative nodes functionality in LangGraph.

Tests cover:
- Basic negative node creation and edge routing
- Probabilistic distribution handling
- Error validation and edge cases
- Integration with StateGraph
"""

import pytest
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    """Simple state for testing."""

    value: int


def dummy_node(state: State, config: RunnableConfig | None = None) -> State:
    """Simple node that increments value."""
    return {"value": state["value"] + 1}


class TestNegativeNodesBasic:
    """Test basic negative node functionality."""

    def test_normal_node_edge(self):
        """Test basic edge creation between normal nodes."""
        builder = StateGraph(State)
        builder.add_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_edge("a", "b")

    def test_start_to_node(self):
        """Test edge from START to a node."""
        builder = StateGraph(State)
        builder.add_node("a", dummy_node)
        builder.add_edge(START, "a")

    def test_node_to_end(self):
        """Test edge from a node to END."""
        builder = StateGraph(State)
        builder.add_node("a", dummy_node)
        builder.add_edge("a", END)


class TestNegativeNodesValidation:
    """Test negative node validation and error handling."""

    def test_negative_without_prob_distribution_rejects(self):
        """Test that negative nodes require probability distribution."""
        builder = StateGraph(State)
        builder.add_negative_node("a", dummy_node)
        builder.add_node("b", dummy_node)

        with pytest.raises(ValueError, match="probabilistic distribution"):
            builder.add_edge("a", "b")

    def test_negative_with_prob_distribution_succeeds(self):
        """Test that negative nodes accept valid probability distribution."""
        builder = StateGraph(State)
        builder.add_negative_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_node("c", dummy_node)
        builder.add_edge("a", ["b", "c"], nodes_prob_distribution=[0.5, 0.5])

    def test_invalid_prob_sum_rejects(self):
        """Test that probabilities must sum to 1.0."""
        builder = StateGraph(State)
        builder.add_negative_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_node("c", dummy_node)

        with pytest.raises(ValueError, match="sum to 1.0"):
            builder.add_edge("a", ["b", "c"], nodes_prob_distribution=[0.3, 0.5])

    def test_negative_probabilities_reject(self):
        """Test that negative probabilities are rejected."""
        builder = StateGraph(State)
        builder.add_negative_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_node("c", dummy_node)

        with pytest.raises(ValueError, match="non-negative"):
            builder.add_edge("a", ["b", "c"], nodes_prob_distribution=[-0.2, 1.2])

    def test_mismatched_prob_length_rejects(self):
        """Test that prob distribution length must match end nodes."""
        builder = StateGraph(State)
        builder.add_negative_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_node("c", dummy_node)

        with pytest.raises(ValueError, match="Length"):
            builder.add_edge("a", ["b", "c"], nodes_prob_distribution=[0.5])

    def test_regular_node_multiple_edges_rejects(self):
        """Test that regular nodes cannot have multiple edges without prob."""
        builder = StateGraph(State)
        builder.add_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_node("c", dummy_node)

        with pytest.raises(ValueError, match="Only negative nodes"):
            builder.add_edge("a", ["b", "c"])

    def test_negative_to_end_with_prob_succeeds(self):
        """Test that negative nodes can route to END with probability."""
        builder = StateGraph(State)
        builder.add_negative_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_edge("a", ["b", END], nodes_prob_distribution=[0.6, 0.4])

    def test_multiple_starts_with_negative_rejects(self):
        """Test that multiple start nodes with negative nodes are rejected."""
        builder = StateGraph(State)
        builder.add_negative_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_node("c", dummy_node)

        with pytest.raises(ValueError, match="Cannot have multiple start nodes"):
            builder.add_edge(["a", "b"], "c")

    def test_multiple_regular_starts_succeeds(self):
        """Test that multiple regular start nodes are allowed."""
        builder = StateGraph(State)
        builder.add_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_node("c", dummy_node)
        builder.add_edge(["a", "b"], "c")

    def test_nonexistent_start_node_rejects(self):
        """Test that edges from non-existent start nodes are rejected."""
        builder = StateGraph(State)
        builder.add_node("a", dummy_node)

        with pytest.raises(ValueError, match="Need to add_node"):
            builder.add_edge("nonexistent", "a")

    def test_nonexistent_end_node_rejects(self):
        """Test that edges to non-existent end nodes are rejected."""
        builder = StateGraph(State)
        builder.add_node("a", dummy_node)

        with pytest.raises(ValueError, match="Need to add_node"):
            builder.add_edge("a", "nonexistent")

    def test_end_as_start_rejects(self):
        """Test that END cannot be used as a start node."""
        builder = StateGraph(State)
        builder.add_node("a", dummy_node)

        with pytest.raises(ValueError, match="END cannot be a start"):
            builder.add_edge(END, "a")

    def test_start_as_end_rejects(self):
        """Test that START cannot be used as an end node."""
        builder = StateGraph(State)
        builder.add_node("a", dummy_node)

        with pytest.raises(ValueError, match="START cannot be an end"):
            builder.add_edge("a", START)


class TestNegativeNodesIntegration:
    """Test negative nodes integration with StateGraph."""

    def test_negative_node_graph_compilation(self):
        """Test that a graph with negative nodes can be compiled."""
        builder = StateGraph(State)
        builder.add_node("start", dummy_node)
        builder.add_negative_node("decision", dummy_node)
        builder.add_node("path_a", dummy_node)
        builder.add_node("path_b", dummy_node)

        builder.add_edge(START, "start")
        builder.add_edge("start", ["decision"], nodes_prob_distribution=[1.0])
        builder.add_edge(
            "decision", ["path_a", "path_b"], nodes_prob_distribution=[0.5, 0.5]
        )
        builder.add_edge("path_a", END)
        builder.add_edge("path_b", END)

        # Should compile without errors
        graph = builder.compile()
        assert graph is not None

    def test_negative_node_with_mixed_paths(self):
        """Test negative node with multiple routing options."""
        builder = StateGraph(State)
        builder.add_negative_node("router", dummy_node)
        builder.add_node("process_a", dummy_node)
        builder.add_node("process_b", dummy_node)

        builder.add_edge(START, "router")
        builder.add_edge(
            "router", ["process_a", "process_b"], nodes_prob_distribution=[0.7, 0.3]
        )
        builder.add_edge("process_a", END)
        builder.add_edge("process_b", END)

        graph = builder.compile()
        assert graph is not None

    def test_probability_distribution_list_format(self):
        """Test that probability distribution can be specified as a list."""
        builder = StateGraph(State)
        builder.add_negative_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_node("c", dummy_node)

        # Should accept list format
        builder.add_edge("a", ["b", "c"], nodes_prob_distribution=[0.6, 0.4])
        graph = builder.compile()
        assert graph is not None

    def test_uniform_distribution(self):
        """Test uniform probability distribution."""
        builder = StateGraph(State)
        builder.add_negative_node("a", dummy_node)
        builder.add_node("b", dummy_node)
        builder.add_node("c", dummy_node)
        builder.add_node("d", dummy_node)

        # Uniform distribution: equal probability for all paths
        builder.add_edge(
            "a", ["b", "c", "d"], nodes_prob_distribution=[1 / 3, 1 / 3, 1 / 3]
        )
        graph = builder.compile()
        assert graph is not None

    def test_asymmetric_distribution(self):
        """Test asymmetric probability distribution."""
        builder = StateGraph(State)
        builder.add_negative_node("a", dummy_node)
        builder.add_node("primary", dummy_node)
        builder.add_node("fallback", dummy_node)

        # Asymmetric: 95% to primary, 5% to fallback
        builder.add_edge(
            "a", ["primary", "fallback"], nodes_prob_distribution=[0.95, 0.05]
        )
        graph = builder.compile()
        assert graph is not None
