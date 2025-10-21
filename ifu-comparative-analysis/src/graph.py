"""
Main graph for IFU Comparative Analysis System.

Implements a sophisticated agentic workflow using LangGraph 1.0 best practices:
- StateGraph with typed state
- Conditional edges for flow control
- Checkpointing for persistence
- Retry policies for resilience
- Human-in-the-loop for critical reviews
"""

from typing import Literal
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt

from .state import AnalysisState, AnalysisContext
from .nodes import (
    initialize_analysis,
    extract_documents,
    analyze_structure,
    compare_documents,
    generate_report,
    update_checklist,
)


def should_continue_after_comparison(state: AnalysisState) -> Literal["review", "generate"]:
    """
    Decide whether human review is needed after comparison.

    Args:
        state: Current analysis state

    Returns:
        "review" if critical changes detected, otherwise "generate"
    """
    if state.get("requires_review", False):
        return "review"
    return "generate"


def human_review_node(state: AnalysisState) -> dict:
    """
    Pause for human review of critical changes.

    Uses LangGraph 1.0 interrupt() feature.

    Args:
        state: Current analysis state

    Returns:
        Partial state updates based on human input
    """
    differences = state.get("differences", [])
    critical_diffs = [d for d in differences if d.get("severity") == "critical"]

    review_data = {
        "total_differences": len(differences),
        "critical_count": len(critical_diffs),
        "critical_changes": [
            {
                "section": d.get("section"),
                "type": d.get("type"),
                "old": d.get("old_text", "")[:200],
                "new": d.get("new_text", "")[:200],
            }
            for d in critical_diffs[:10]  # Show top 10
        ],
        "message": "Critical changes detected. Please review before generating report.",
    }

    # Interrupt and wait for human input
    approval = interrupt(review_data)

    if approval and approval.get("approved", False):
        notes = approval.get("notes", "Reviewed and approved")
        return {
            "review_notes": notes,
            "current_step": "review_approved"
        }
    else:
        return {
            "status": "failed",
            "current_step": "review_rejected",
            "errors": ["Human review rejected the analysis"]
        }


def create_analysis_graph(
    checkpointer_type: Literal["memory", "sqlite", "postgres"] = "sqlite",
    enable_human_review: bool = True
):
    """
    Create the IFU comparative analysis graph.

    Args:
        checkpointer_type: Type of checkpointer to use
        enable_human_review: Whether to enable human-in-the-loop review

    Returns:
        Compiled graph ready for execution
    """
    # Create StateGraph
    builder = StateGraph(AnalysisState)

    # Add nodes
    builder.add_node("initialize", initialize_analysis)
    builder.add_node("extract", extract_documents)
    builder.add_node("analyze", analyze_structure)
    builder.add_node("compare", compare_documents)
    builder.add_node("review", human_review_node)
    builder.add_node("generate", generate_report)
    builder.add_node("update_checklist", update_checklist)

    # Define the flow
    # Start -> Initialize
    builder.add_edge(START, "initialize")

    # Initialize -> Extract
    builder.add_edge("initialize", "extract")

    # Extract -> Analyze
    builder.add_edge("extract", "analyze")

    # Analyze -> Compare
    builder.add_edge("analyze", "compare")

    # Compare -> Review or Generate (conditional)
    if enable_human_review:
        builder.add_conditional_edges(
            "compare",
            should_continue_after_comparison,
            {
                "review": "review",
                "generate": "generate"
            }
        )

        # Review -> Generate
        builder.add_edge("review", "generate")
    else:
        builder.add_edge("compare", "generate")

    # Generate -> Update Checklist
    builder.add_edge("generate", "update_checklist")

    # Update Checklist -> End
    builder.add_edge("update_checklist", END)

    # Configure checkpointer
    if checkpointer_type == "sqlite":
        import sqlite3
        conn = sqlite3.connect("ifu-comparative-analysis/checkpoints.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
    elif checkpointer_type == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    else:
        # PostgreSQL would be configured here for production
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()

    # Compile graph with checkpointing and interrupts
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["review"] if enable_human_review else None,
    )

    return graph


def run_ifu_analysis(
    old_pdf_path: str,
    new_pdf_path: str,
    analysis_id: str = None,
    enable_human_review: bool = True,
    checkpoint_type: Literal["memory", "sqlite", "postgres"] = "sqlite"
) -> dict:
    """
    Run complete IFU comparative analysis.

    Args:
        old_pdf_path: Path to old version PDF
        new_pdf_path: Path to new version PDF
        analysis_id: Optional analysis ID (generated if not provided)
        enable_human_review: Whether to enable human review for critical changes
        checkpoint_type: Type of checkpointer to use

    Returns:
        Final state with report path and all differences
    """
    # Generate analysis ID
    if not analysis_id:
        analysis_id = f"ifu_analysis_{uuid.uuid4().hex[:8]}"

    # Create initial state
    initial_state = AnalysisState(
        old_pdf_path=old_pdf_path,
        new_pdf_path=new_pdf_path,
        analysis_id=analysis_id,
        started_at=datetime.now(),
        current_step="initialized",
        status="initialized",
        report_status="pending",
        requires_review=False,
        differences=[],
        checklist=[],
        errors=[]
    )

    # Create and run graph
    graph = create_analysis_graph(
        checkpointer_type=checkpoint_type,
        enable_human_review=enable_human_review
    )

    # Configure thread for checkpointing
    config = {
        "configurable": {
            "thread_id": analysis_id
        }
    }

    print(f"\n{'='*60}")
    print(f"Starting IFU Comparative Analysis")
    print(f"Analysis ID: {analysis_id}")
    print(f"Old PDF: {old_pdf_path}")
    print(f"New PDF: {new_pdf_path}")
    print(f"Human Review: {'Enabled' if enable_human_review else 'Disabled'}")
    print(f"{'='*60}\n")

    # Run graph with streaming for progress updates
    final_state = None

    try:
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            # Print progress
            for node_name, node_output in event.items():
                current_step = node_output.get("current_step", "")
                status = node_output.get("status", "")

                if current_step:
                    print(f"✓ {node_name}: {current_step} (status: {status})")

                # Check for errors
                if node_output.get("errors"):
                    for error in node_output["errors"]:
                        print(f"✗ ERROR: {error}")

            final_state = node_output

        # Check if we hit an interrupt (human review needed)
        state = graph.get_state(config)

        if state.next and "review" in state.next:
            print("\n⚠ HUMAN REVIEW REQUIRED")
            print("Critical changes detected. Review needed before generating report.")
            print("To continue:")
            print(f"  1. Review the changes")
            print(f"  2. Call resume_analysis('{analysis_id}', approved=True/False)")
            return {
                "status": "awaiting_review",
                "analysis_id": analysis_id,
                "state": state.values,
                "message": "Analysis paused for human review"
            }

    except Exception as e:
        print(f"\n✗ Analysis failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "analysis_id": analysis_id
        }

    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"Report: {final_state.get('report_path', 'N/A')}")
    print(f"Total Differences: {len(final_state.get('differences', []))}")
    print(f"{'='*60}\n")

    return final_state


def resume_analysis(
    analysis_id: str,
    approved: bool = True,
    notes: str = "",
    checkpoint_type: Literal["memory", "sqlite", "postgres"] = "sqlite"
) -> dict:
    """
    Resume analysis after human review.

    Args:
        analysis_id: ID of the analysis to resume
        approved: Whether the review is approved
        notes: Optional review notes
        checkpoint_type: Type of checkpointer used

    Returns:
        Final state after resuming
    """
    # Create graph
    graph = create_analysis_graph(
        checkpointer_type=checkpoint_type,
        enable_human_review=True
    )

    config = {
        "configurable": {
            "thread_id": analysis_id
        }
    }

    print(f"\nResuming analysis {analysis_id}...")
    print(f"Approved: {approved}")

    # Resume with Command
    resume_command = Command(
        resume={
            "approved": approved,
            "notes": notes
        }
    )

    final_state = None

    try:
        for event in graph.stream(resume_command, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                current_step = node_output.get("current_step", "")
                print(f"✓ {node_name}: {current_step}")

            final_state = node_output

    except Exception as e:
        print(f"✗ Resume failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

    print(f"\n✓ Analysis resumed and completed")
    print(f"Report: {final_state.get('report_path', 'N/A')}")

    return final_state


# Export main functions
__all__ = [
    "create_analysis_graph",
    "run_ifu_analysis",
    "resume_analysis",
]
