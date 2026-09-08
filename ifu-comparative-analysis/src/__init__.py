"""
IFU Comparative Analysis System

An advanced agentic platform for comprehensive IFU (Instructions For Use) comparison
using Claude Sonnet 4.5's native vision capabilities and LangGraph 1.0.

Features:
- Native PDF processing with Claude's vision
- Comprehensive difference detection
- Side-by-side Word report generation
- Human-in-the-loop review for critical changes
- Progress tracking with checklist
- Persistent checkpointing for long-running analyses
"""

from .graph import (
    create_analysis_graph,
    run_ifu_analysis,
    resume_analysis,
)

from .state import (
    AnalysisState,
    AnalysisContext,
    ChecklistItem,
    Difference,
    PDFDocument,
    Section,
)

__version__ = "1.0.0"

__all__ = [
    "create_analysis_graph",
    "run_ifu_analysis",
    "resume_analysis",
    "AnalysisState",
    "AnalysisContext",
    "ChecklistItem",
    "Difference",
    "PDFDocument",
    "Section",
]
