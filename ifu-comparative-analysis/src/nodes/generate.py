"""
Report generation node.

Generates comprehensive Word report with side-by-side comparison.
"""

from datetime import datetime
from pathlib import Path
from ..state import AnalysisState
from ..tools.word_tools import (
    initialize_comparison_report,
    add_executive_summary,
    add_section_comparison,
    add_detailed_differences_table,
    add_table_of_contents,
    finalize_report
)
from ..tools.diff_tools import generate_change_summary


def generate_report(state: AnalysisState) -> dict:
    """
    Generate comprehensive Word report.

    Creates a professional, formatted report with:
    - Executive summary
    - Side-by-side section comparisons
    - Detailed differences table
    - Color-coded changes
    - Table of contents

    Args:
        state: Current analysis state

    Returns:
        Partial state updates with report path
    """
    errors = []

    try:
        # Generate output filename
        output_dir = Path("ifu-comparative-analysis/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"IFU_Comparison_Report_{timestamp}.docx"
        report_path = str(output_dir / report_filename)

        print(f"Generating report: {report_path}")

        # Step 1: Initialize report
        print("Step 1: Initializing report document...")

        initialize_comparison_report.invoke({
            "output_path": report_path,
            "title": "IFU Comparative Analysis Report",
            "author": "IFU Analysis System (Claude Sonnet 4.5)",
            "company": "Medical Device Documentation"
        })

        # Update checklist
        updated_checklist = state.get("checklist", []).copy()
        for item in updated_checklist:
            if item["id"] == "generate_report":
                item["status"] = "in_progress"
                item["started_at"] = datetime.now()

        # Step 2: Generate and add executive summary
        print("Step 2: Generating executive summary...")

        all_differences = state.get("differences", [])

        summary_data = generate_change_summary.invoke({
            "all_differences": all_differences,
            "model": "claude-sonnet-4.5"
        })

        add_executive_summary.invoke({
            "doc_path": report_path,
            "summary_data": summary_data
        })

        for item in updated_checklist:
            if item["id"] == "generate_summary":
                item["status"] = "completed"
                item["completed_at"] = datetime.now()

        # Step 3: Add section-by-section comparisons
        print("Step 3: Adding section comparisons...")

        old_sections = state.get("old_sections", [])
        new_sections = state.get("new_sections", [])

        # Match sections by title
        section_pairs = []
        for old_section in old_sections:
            for new_section in new_sections:
                if old_section["title"] == new_section["title"]:
                    section_pairs.append((old_section, new_section))
                    break

        for old_section, new_section in section_pairs:
            # Find differences for this section
            section_diffs = [
                diff for diff in all_differences
                if diff.get("section") == old_section["title"]
            ]

            add_section_comparison.invoke({
                "doc_path": report_path,
                "section_name": old_section["title"],
                "page_old": old_section.get("page_start", 0),
                "page_new": new_section.get("page_start", 0),
                "old_content": old_section.get("content", "")[:5000],  # Truncate if too long
                "new_content": new_section.get("content", "")[:5000],
                "differences": section_diffs[:20]  # Limit to top 20 differences per section
            })

        # Step 4: Add detailed differences table
        print("Step 4: Adding detailed differences table...")

        add_detailed_differences_table.invoke({
            "doc_path": report_path,
            "all_differences": all_differences
        })

        # Step 5: Add table of contents
        print("Step 5: Adding table of contents...")

        add_table_of_contents.invoke({
            "doc_path": report_path
        })

        # Step 6: Finalize report
        print("Step 6: Finalizing report...")

        finalize_report.invoke({
            "doc_path": report_path,
            "add_footer": True
        })

        # Update checklist
        for item in updated_checklist:
            if item["id"] == "generate_report":
                item["status"] = "completed"
                item["completed_at"] = datetime.now()
                item["result"] = f"Report saved to {report_path}"
            elif item["id"] == "finalize":
                item["status"] = "completed"
                item["completed_at"] = datetime.now()

        print(f"\n✓ Report generated successfully: {report_path}")

        return {
            "report_path": report_path,
            "report_status": "completed",
            "status": "completed",
            "current_step": "report_generated",
            "completed_at": datetime.now(),
            "checklist": updated_checklist,
        }

    except Exception as e:
        errors.append(f"Report generation failed: {str(e)}")

        return {
            "report_status": "failed",
            "status": "failed",
            "current_step": "report_generation_failed",
            "errors": errors,
        }
