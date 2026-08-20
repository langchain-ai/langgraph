"""
Document extraction node using Claude's native PDF vision.

Extracts structure and content from both PDF versions.
"""

from datetime import datetime
from ..state import AnalysisState, PDFDocument
from ..tools.pdf_tools import extract_pdf_structure


def extract_documents(state: AnalysisState) -> dict:
    """
    Extract structure and content from both PDF documents.

    Leverages Claude Sonnet 4.5's native PDF processing.

    Args:
        state: Current analysis state

    Returns:
        Partial state updates with extracted documents
    """
    errors = []

    try:
        # Extract old document
        print(f"Extracting old document: {state['old_pdf_path']}")
        old_result = extract_pdf_structure.invoke({
            "pdf_path": state["old_pdf_path"],
            "model": "claude-sonnet-4.5"
        })

        old_document = PDFDocument(
            file_path=state["old_pdf_path"],
            total_pages=old_result.get("total_pages", 0),
            extracted_text=old_result.get("pages", {}),
            structure=old_result.get("sections", {}),
            metadata=old_result.get("metadata", {})
        )

        # Extract new document
        print(f"Extracting new document: {state['new_pdf_path']}")
        new_result = extract_pdf_structure.invoke({
            "pdf_path": state["new_pdf_path"],
            "model": "claude-sonnet-4.5"
        })

        new_document = PDFDocument(
            file_path=state["new_pdf_path"],
            total_pages=new_result.get("total_pages", 0),
            extracted_text=new_result.get("pages", {}),
            structure=new_result.get("sections", {}),
            metadata=new_result.get("metadata", {})
        )

        # Update checklist
        updated_checklist = state.get("checklist", []).copy()
        for item in updated_checklist:
            if item["id"] in ["extract_old", "extract_new"]:
                item["status"] = "completed"
                item["completed_at"] = datetime.now()

        return {
            "old_document": old_document,
            "new_document": new_document,
            "status": "analyzing",
            "current_step": "extraction_complete",
            "checklist": updated_checklist,
        }

    except Exception as e:
        errors.append(f"Document extraction failed: {str(e)}")

        return {
            "status": "failed",
            "current_step": "extraction_failed",
            "errors": errors,
        }
