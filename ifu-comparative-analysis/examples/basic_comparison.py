"""
Basic IFU Comparison Example

This example demonstrates how to run a complete IFU comparative analysis
using the IFU Analysis System.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import run_ifu_analysis, resume_analysis


def main():
    """
    Run a basic IFU comparison.
    """
    # Paths to your IFU PDFs
    old_pdf = "examples/sample_ifus/ifu_v1.0.pdf"
    new_pdf = "examples/sample_ifus/ifu_v2.0.pdf"

    print("=" * 80)
    print("IFU Comparative Analysis System")
    print("Powered by Claude Sonnet 4.5 and LangGraph 1.0")
    print("=" * 80)
    print()

    # Check if files exist
    if not Path(old_pdf).exists():
        print(f"Error: Old PDF not found: {old_pdf}")
        print("Please place your IFU PDFs in examples/sample_ifus/")
        return

    if not Path(new_pdf).exists():
        print(f"Error: New PDF not found: {new_pdf}")
        print("Please place your IFU PDFs in examples/sample_ifus/")
        return

    # Run analysis
    print("Starting IFU comparative analysis...")
    print(f"  Old version: {old_pdf}")
    print(f"  New version: {new_pdf}")
    print()

    result = run_ifu_analysis(
        old_pdf_path=old_pdf,
        new_pdf_path=new_pdf,
        enable_human_review=True,  # Enable human review for critical changes
        checkpoint_type="sqlite"    # Use SQLite for persistence
    )

    # Check result
    if result.get("status") == "awaiting_review":
        print()
        print("=" * 80)
        print("ANALYSIS PAUSED - HUMAN REVIEW REQUIRED")
        print("=" * 80)
        print()
        print("Critical changes were detected that require human review.")
        print(f"Analysis ID: {result['analysis_id']}")
        print()
        print("To approve and continue:")
        print(f"  resume_analysis('{result['analysis_id']}', approved=True)")
        print()
        print("To reject:")
        print(f"  resume_analysis('{result['analysis_id']}', approved=False)")
        print()

        # For this example, we'll auto-approve
        print("Auto-approving for demonstration purposes...")
        print()

        result = resume_analysis(
            analysis_id=result["analysis_id"],
            approved=True,
            notes="Reviewed and approved - example run",
            checkpoint_type="sqlite"
        )

    # Print results
    if result.get("status") == "completed":
        print()
        print("=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print(f"Report generated: {result.get('report_path')}")
        print(f"Total differences found: {len(result.get('differences', []))}")
        print()

        # Count by severity
        differences = result.get("differences", [])
        critical = sum(1 for d in differences if d.get("severity") == "critical")
        major = sum(1 for d in differences if d.get("severity") == "major")
        minor = sum(1 for d in differences if d.get("severity") == "minor")

        print("Breakdown by severity:")
        print(f"  Critical: {critical}")
        print(f"  Major: {major}")
        print(f"  Minor: {minor}")
        print()

        # Show checklist
        checklist = result.get("checklist", [])
        completed = sum(1 for item in checklist if item["status"] == "completed")
        print(f"Checklist: {completed}/{len(checklist)} items completed")
        print()

    elif result.get("status") == "failed":
        print()
        print("=" * 80)
        print("ANALYSIS FAILED")
        print("=" * 80)
        print()
        print(f"Error: {result.get('error', 'Unknown error')}")
        errors = result.get("errors", [])
        if errors:
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
        print()

    else:
        print()
        print(f"Analysis status: {result.get('status', 'unknown')}")
        print()


def example_with_custom_config():
    """
    Example with custom configuration.
    """
    from src.utils.config import load_config

    # Load config
    config = load_config()

    # Override specific settings
    config["enable_human_review"] = False  # Skip human review
    config["max_pages_to_compare"] = 50    # Limit to 50 pages

    # Run analysis
    result = run_ifu_analysis(
        old_pdf_path="examples/sample_ifus/ifu_v1.0.pdf",
        new_pdf_path="examples/sample_ifus/ifu_v2.0.pdf",
        enable_human_review=config["enable_human_review"],
        checkpoint_type=config["checkpoint_backend"]
    )

    return result


def example_streaming():
    """
    Example showing streaming progress updates.
    """
    from src.graph import create_analysis_graph
    from src.state import AnalysisState
    from datetime import datetime

    # Create graph
    graph = create_analysis_graph(
        checkpointer_type="sqlite",
        enable_human_review=False
    )

    # Initial state
    initial_state = AnalysisState(
        old_pdf_path="examples/sample_ifus/ifu_v1.0.pdf",
        new_pdf_path="examples/sample_ifus/ifu_v2.0.pdf",
        analysis_id="streaming_example",
        started_at=datetime.now(),
        current_step="initialized",
        status="initialized",
        report_status="pending",
        requires_review=False,
        differences=[],
        checklist=[],
        errors=[]
    )

    config = {"configurable": {"thread_id": "streaming_example"}}

    # Stream with progress updates
    print("Streaming analysis progress...")
    print()

    for event in graph.stream(initial_state, config, stream_mode="updates"):
        for node_name, output in event.items():
            step = output.get("current_step", "")
            status = output.get("status", "")

            print(f"[{node_name}] {step} - Status: {status}")

            # Show checklist progress
            checklist = output.get("checklist", [])
            if checklist:
                completed = sum(1 for item in checklist if item["status"] == "completed")
                print(f"  Checklist: {completed}/{len(checklist)} completed")

    print()
    print("Streaming complete!")


if __name__ == "__main__":
    # Run basic example
    main()

    # Uncomment to try other examples:
    # example_with_custom_config()
    # example_streaming()
