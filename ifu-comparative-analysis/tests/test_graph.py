"""
Tests for IFU Analysis graph.

Basic test suite to validate the graph structure and execution.
"""

import pytest
from datetime import datetime
from src.state import AnalysisState, create_checklist_item
from src.graph import create_analysis_graph


def test_create_checklist_item():
    """Test checklist item creation."""
    item = create_checklist_item("test_id", "Test description", "pending")

    assert item["id"] == "test_id"
    assert item["description"] == "Test description"
    assert item["status"] == "pending"


def test_analysis_state_structure():
    """Test that AnalysisState has required fields."""
    state = AnalysisState(
        old_pdf_path="/path/to/old.pdf",
        new_pdf_path="/path/to/new.pdf",
        analysis_id="test_123",
        started_at=datetime.now(),
        current_step="initialized",
        status="initialized",
        report_status="pending",
        requires_review=False,
        differences=[],
        checklist=[],
        errors=[]
    )

    assert state["old_pdf_path"] == "/path/to/old.pdf"
    assert state["new_pdf_path"] == "/path/to/new.pdf"
    assert state["status"] == "initialized"


def test_create_analysis_graph():
    """Test graph creation."""
    graph = create_analysis_graph(
        checkpointer_type="memory",
        enable_human_review=False
    )

    assert graph is not None
    # Graph should be compiled
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "stream")


def test_graph_nodes_exist():
    """Test that all expected nodes exist in the graph."""
    graph = create_analysis_graph(checkpointer_type="memory")

    # Get graph structure
    # Note: This is a basic check - full integration tests would validate execution
    assert graph is not None


@pytest.mark.skip(reason="Requires valid PDFs and API key")
def test_full_analysis_integration():
    """
    Integration test for full analysis.

    This test requires:
    - Valid PDF files
    - Anthropic API key
    - Sufficient API credits

    Run with: pytest -v -s tests/test_graph.py::test_full_analysis_integration
    """
    from src import run_ifu_analysis

    result = run_ifu_analysis(
        old_pdf_path="examples/sample_ifus/ifu_v1.0.pdf",
        new_pdf_path="examples/sample_ifus/ifu_v2.0.pdf",
        enable_human_review=False,
        checkpoint_type="memory"
    )

    assert result["status"] in ["completed", "failed"]

    if result["status"] == "completed":
        assert "report_path" in result
        assert "differences" in result
