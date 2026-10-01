"""
Tools for IFU comparative analysis.

All tools leverage Claude Sonnet 4.5's capabilities for maximum accuracy.
"""

from .pdf_tools import (
    extract_pdf_structure,
    extract_pdf_page_range,
    compare_pdf_pages_visually,
    validate_pdf_accessibility,
)

from .diff_tools import (
    detect_section_differences,
    analyze_text_differences,
    classify_change_severity,
    identify_missing_content,
    generate_change_summary,
)

from .word_tools import (
    initialize_comparison_report,
    add_executive_summary,
    add_section_comparison,
    add_detailed_differences_table,
    add_table_of_contents,
    finalize_report,
)

__all__ = [
    # PDF tools
    "extract_pdf_structure",
    "extract_pdf_page_range",
    "compare_pdf_pages_visually",
    "validate_pdf_accessibility",
    # Diff tools
    "detect_section_differences",
    "analyze_text_differences",
    "classify_change_severity",
    "identify_missing_content",
    "generate_change_summary",
    # Word tools
    "initialize_comparison_report",
    "add_executive_summary",
    "add_section_comparison",
    "add_detailed_differences_table",
    "add_table_of_contents",
    "finalize_report",
]
