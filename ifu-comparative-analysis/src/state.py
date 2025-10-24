"""
State schemas for IFU Comparative Analysis System.

Following LangGraph 1.0 best practices:
- TypedDict with clear field names
- Annotated fields with appropriate reducers
- Separate schemas for different contexts
"""

from typing import TypedDict, Annotated, Literal, Optional
from typing_extensions import NotRequired
import operator
from datetime import datetime


class ChecklistItem(TypedDict):
    """Individual checklist item for tracking analysis progress."""
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    started_at: NotRequired[datetime]
    completed_at: NotRequired[datetime]
    result: NotRequired[str]
    error: NotRequired[str]


class PDFDocument(TypedDict):
    """Represents a parsed PDF document."""
    file_path: str
    total_pages: int
    extracted_text: dict[int, str]  # page_num -> text
    structure: dict[int, dict]  # page_num -> structure metadata
    metadata: dict


class Section(TypedDict):
    """Represents a document section."""
    title: str
    page_start: int
    page_end: int
    content: str
    level: int  # Heading level


class Difference(TypedDict):
    """Represents a detected difference between documents."""
    section: str
    page_old: int
    page_new: int
    type: Literal["added", "removed", "modified"]
    old_text: str
    new_text: str
    context: str
    severity: Literal["critical", "major", "minor"]


class AnalysisState(TypedDict):
    """Main state for IFU comparative analysis workflow.

    Following best practices:
    - Use Annotated with operator.add for list accumulation
    - Simple fields use LastValue reducer (default)
    - Clear naming for all fields
    """

    # Input files
    old_pdf_path: str
    new_pdf_path: str

    # Parsed documents
    old_document: NotRequired[PDFDocument]
    new_document: NotRequired[PDFDocument]

    # Document structure
    old_sections: NotRequired[list[Section]]
    new_sections: NotRequired[list[Section]]

    # Detected differences - accumulated across multiple detections
    differences: Annotated[list[Difference], operator.add]

    # Checklist tracking - accumulated as items are added
    checklist: Annotated[list[ChecklistItem], operator.add]

    # Current processing step
    current_step: str

    # Analysis metadata
    analysis_id: str
    started_at: datetime
    completed_at: NotRequired[datetime]

    # Report generation
    report_path: NotRequired[str]
    report_status: Literal["pending", "generating", "completed", "failed"]

    # Error tracking
    errors: Annotated[list[str], operator.add]

    # Status
    status: Literal["initialized", "extracting", "analyzing", "comparing", "generating_report", "completed", "failed"]

    # Human review flags
    requires_review: bool
    review_notes: NotRequired[str]


class AnalysisContext(TypedDict):
    """Immutable context for the analysis run.

    Following LangGraph 1.0 best practices for context schema.
    This data doesn't change during execution.
    """

    # Model configuration
    model_name: Literal["claude-sonnet-4.5"]
    temperature: float
    max_tokens: int

    # Analysis configuration
    language: str
    include_images: bool
    severity_threshold: Literal["all", "major", "critical"]

    # Report configuration
    output_format: Literal["docx"]
    include_summary: bool
    include_statistics: bool

    # Processing options
    enable_checkpointing: bool
    checkpoint_backend: Literal["memory", "sqlite", "postgres"]


class ReportConfig(TypedDict):
    """Configuration for report generation."""
    template: str
    title: str
    author: str
    company: str
    date: str
    logo_path: NotRequired[str]
    include_toc: bool
    highlight_critical: bool


def create_checklist_item(
    id: str,
    description: str,
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
) -> ChecklistItem:
    """Helper to create a checklist item."""
    return ChecklistItem(
        id=id,
        description=description,
        status=status
    )


def create_difference(
    section: str,
    page_old: int,
    page_new: int,
    type: Literal["added", "removed", "modified"],
    old_text: str,
    new_text: str,
    context: str,
    severity: Literal["critical", "major", "minor"] = "minor"
) -> Difference:
    """Helper to create a difference record."""
    return Difference(
        section=section,
        page_old=page_old,
        page_new=page_new,
        type=type,
        old_text=old_text,
        new_text=new_text,
        context=context,
        severity=severity
    )
