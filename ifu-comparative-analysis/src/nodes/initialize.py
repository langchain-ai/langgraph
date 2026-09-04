"""
Initialization node for IFU analysis workflow.

Sets up the analysis session and validates inputs.
"""

from datetime import datetime
from ..state import AnalysisState, create_checklist_item
from ..tools.pdf_tools import validate_pdf_accessibility


def initialize_analysis(state: AnalysisState) -> dict:
    """
    Initialize the analysis workflow.

    Validates PDF files and sets up initial checklist.

    Args:
        state: Current analysis state

    Returns:
        Partial state updates
    """
    checklist_items = []
    errors = []

    # Validate old PDF
    old_validation = validate_pdf_accessibility.invoke({"pdf_path": state["old_pdf_path"]})

    if not old_validation["valid"]:
        errors.append(f"Old PDF validation failed: {old_validation['error']}")
        return {
            "status": "failed",
            "errors": errors,
            "current_step": "initialization_failed"
        }

    # Validate new PDF
    new_validation = validate_pdf_accessibility.invoke({"pdf_path": state["new_pdf_path"]})

    if not new_validation["valid"]:
        errors.append(f"New PDF validation failed: {new_validation['error']}")
        return {
            "status": "failed",
            "errors": errors,
            "current_step": "initialization_failed"
        }

    # Create initial checklist
    checklist_items = [
        create_checklist_item("init", "Initialize analysis workflow", "completed"),
        create_checklist_item("validate_pdfs", "Validate PDF accessibility", "completed"),
        create_checklist_item("extract_old", "Extract old document structure", "pending"),
        create_checklist_item("extract_new", "Extract new document structure", "pending"),
        create_checklist_item("analyze_structure", "Analyze document structures", "pending"),
        create_checklist_item("compare_sections", "Compare document sections", "pending"),
        create_checklist_item("detect_differences", "Detect all differences", "pending"),
        create_checklist_item("classify_severity", "Classify change severity", "pending"),
        create_checklist_item("generate_summary", "Generate executive summary", "pending"),
        create_checklist_item("generate_report", "Generate Word report", "pending"),
        create_checklist_item("finalize", "Finalize report", "pending"),
    ]

    # Mark checklist items as completed
    checklist_items[0]["completed_at"] = datetime.now()
    checklist_items[1]["completed_at"] = datetime.now()

    return {
        "status": "extracting",
        "current_step": "initialization_complete",
        "checklist": checklist_items,
        "started_at": datetime.now(),
    }
