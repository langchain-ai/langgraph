"""
Structure analysis node.

Analyzes and organizes document sections for comparison.
"""

from datetime import datetime
from ..state import AnalysisState, Section


def analyze_structure(state: AnalysisState) -> dict:
    """
    Analyze document structure and organize sections.

    Args:
        state: Current analysis state

    Returns:
        Partial state updates with organized sections
    """
    errors = []

    try:
        # Extract sections from old document
        old_sections = []
        old_structure = state["old_document"]["structure"]

        if isinstance(old_structure, dict):
            for section_data in old_structure.values():
                if isinstance(section_data, dict):
                    section = Section(
                        title=section_data.get("title", "Untitled"),
                        page_start=section_data.get("page_start", 0),
                        page_end=section_data.get("page_end", 0),
                        content=section_data.get("content", ""),
                        level=section_data.get("level", 1)
                    )
                    old_sections.append(section)

        # Extract sections from new document
        new_sections = []
        new_structure = state["new_document"]["structure"]

        if isinstance(new_structure, dict):
            for section_data in new_structure.values():
                if isinstance(section_data, dict):
                    section = Section(
                        title=section_data.get("title", "Untitled"),
                        page_start=section_data.get("page_start", 0),
                        page_end=section_data.get("page_end", 0),
                        content=section_data.get("content", ""),
                        level=section_data.get("level", 1)
                    )
                    new_sections.append(section)

        # Update checklist
        updated_checklist = state.get("checklist", []).copy()
        for item in updated_checklist:
            if item["id"] == "analyze_structure":
                item["status"] = "completed"
                item["completed_at"] = datetime.now()
                item["result"] = f"Analyzed {len(old_sections)} old sections and {len(new_sections)} new sections"

        return {
            "old_sections": old_sections,
            "new_sections": new_sections,
            "status": "comparing",
            "current_step": "structure_analyzed",
            "checklist": updated_checklist,
        }

    except Exception as e:
        errors.append(f"Structure analysis failed: {str(e)}")

        return {
            "status": "failed",
            "current_step": "analysis_failed",
            "errors": errors,
        }
