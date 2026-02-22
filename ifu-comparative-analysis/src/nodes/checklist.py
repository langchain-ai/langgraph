"""
Checklist management node.

Tracks and updates progress throughout the workflow.
"""

from datetime import datetime
from ..state import AnalysisState


def update_checklist(state: AnalysisState) -> dict:
    """
    Update checklist based on current state.

    This node can be called at any point to update checklist status.

    Args:
        state: Current analysis state

    Returns:
        Partial state updates with updated checklist
    """
    checklist = state.get("checklist", []).copy()
    current_step = state.get("current_step", "")

    # Update checklist based on current step
    step_mapping = {
        "initialization_complete": ["init", "validate_pdfs"],
        "extraction_complete": ["extract_old", "extract_new"],
        "structure_analyzed": ["analyze_structure"],
        "comparison_complete": ["compare_sections", "detect_differences", "classify_severity"],
        "report_generated": ["generate_summary", "generate_report", "finalize"]
    }

    if current_step in step_mapping:
        for item_id in step_mapping[current_step]:
            for item in checklist:
                if item["id"] == item_id and item["status"] != "completed":
                    item["status"] = "completed"
                    item["completed_at"] = datetime.now()

    # Count completed items
    completed_count = sum(1 for item in checklist if item["status"] == "completed")
    total_count = len(checklist)

    print(f"Checklist progress: {completed_count}/{total_count} items completed")

    return {
        "checklist": checklist
    }
