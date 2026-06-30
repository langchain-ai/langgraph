"""
Comparison node - detects all differences between documents.

Uses Claude Sonnet 4.5 to perform comprehensive diff analysis.
"""

from datetime import datetime
from typing import List
from ..state import AnalysisState, Difference, create_difference
from ..tools.diff_tools import (
    detect_section_differences,
    analyze_text_differences,
    classify_change_severity,
    identify_missing_content
)
from ..tools.pdf_tools import compare_pdf_pages_visually


def compare_documents(state: AnalysisState) -> dict:
    """
    Comprehensive comparison of old and new documents.

    Performs multiple levels of analysis:
    1. Section-level comparison
    2. Page-by-page visual comparison
    3. Text-level granular comparison
    4. Severity classification

    Args:
        state: Current analysis state

    Returns:
        Partial state updates with all detected differences
    """
    all_differences: List[Difference] = []
    errors = []

    try:
        print("Step 1: Comparing document sections...")

        # Step 1: Section-level comparison
        section_diffs = detect_section_differences.invoke({
            "old_sections": state.get("old_sections", []),
            "new_sections": state.get("new_sections", []),
            "model": "claude-sonnet-4.5"
        })

        for diff_data in section_diffs:
            diff = create_difference(
                section=diff_data.get("section_name", "Unknown"),
                page_old=diff_data.get("old_location", 0),
                page_new=diff_data.get("new_location", 0),
                type=diff_data.get("type", "modified"),
                old_text=diff_data.get("old_content", ""),
                new_text=diff_data.get("new_content", ""),
                context=diff_data.get("description", ""),
                severity=diff_data.get("severity", "minor")
            )
            all_differences.append(diff)

        # Update checklist
        updated_checklist = state.get("checklist", []).copy()
        for item in updated_checklist:
            if item["id"] == "compare_sections":
                item["status"] = "completed"
                item["completed_at"] = datetime.now()
                item["result"] = f"Found {len(section_diffs)} section-level differences"

        print(f"Found {len(section_diffs)} section-level differences")

        # Step 2: Page-by-page visual comparison
        print("Step 2: Performing page-by-page visual comparison...")

        old_pages = state["old_document"]["total_pages"]
        new_pages = state["new_document"]["total_pages"]

        # Compare common pages
        max_pages = max(old_pages, new_pages)

        for page_num in range(1, min(max_pages + 1, 51)):  # Limit to 50 pages for demo
            try:
                page_diffs = compare_pdf_pages_visually.invoke({
                    "old_pdf_path": state["old_pdf_path"],
                    "new_pdf_path": state["new_pdf_path"],
                    "page_number": page_num,
                    "model": "claude-sonnet-4.5"
                })

                if "differences" in page_diffs:
                    for diff_data in page_diffs["differences"]:
                        diff = create_difference(
                            section=f"Page {page_num}",
                            page_old=page_num,
                            page_new=page_num,
                            type=diff_data.get("type", "modified"),
                            old_text=diff_data.get("old_content", ""),
                            new_text=diff_data.get("new_content", ""),
                            context=diff_data.get("location", ""),
                            severity=diff_data.get("severity", "minor")
                        )
                        all_differences.append(diff)

            except Exception as e:
                print(f"Warning: Failed to compare page {page_num}: {e}")
                continue

        print(f"Total differences after page comparison: {len(all_differences)}")

        # Step 3: Text-level granular comparison for each section
        print("Step 3: Performing detailed text-level comparison...")

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
            try:
                text_diffs = analyze_text_differences.invoke({
                    "old_text": old_section.get("content", ""),
                    "new_text": new_section.get("content", ""),
                    "section_name": old_section["title"],
                    "model": "claude-sonnet-4.5"
                })

                for diff_data in text_diffs:
                    diff = create_difference(
                        section=old_section["title"],
                        page_old=old_section.get("page_start", 0),
                        page_new=new_section.get("page_start", 0),
                        type=diff_data.get("type", "modified"),
                        old_text=diff_data.get("old_content", ""),
                        new_text=diff_data.get("new_content", ""),
                        context=diff_data.get("context", ""),
                        severity=diff_data.get("severity", "minor")
                    )
                    all_differences.append(diff)

            except Exception as e:
                print(f"Warning: Failed to compare section {old_section['title']}: {e}")
                continue

        # Update checklist
        for item in updated_checklist:
            if item["id"] == "detect_differences":
                item["status"] = "completed"
                item["completed_at"] = datetime.now()
                item["result"] = f"Detected {len(all_differences)} total differences"

        print(f"Total differences after text comparison: {len(all_differences)}")

        # Step 4: Identify potentially missing content
        print("Step 4: Identifying missing content...")

        try:
            missing_content = identify_missing_content.invoke({
                "old_document": state["old_document"],
                "new_document": state["new_document"],
                "model": "claude-sonnet-4.5"
            })

            for missing in missing_content:
                diff = create_difference(
                    section=missing.get("type", "Unknown"),
                    page_old=missing.get("old_location", 0),
                    page_new=0,
                    type="removed",
                    old_text=missing.get("description", ""),
                    new_text="",
                    context=missing.get("recommendation", ""),
                    severity=missing.get("severity", "major")
                )
                all_differences.append(diff)

        except Exception as e:
            print(f"Warning: Missing content detection failed: {e}")

        # Step 5: Classify severity for any unclassified differences
        print("Step 5: Classifying change severity...")

        severity_count = {"critical": 0, "major": 0, "minor": 0}

        for diff in all_differences:
            severity_count[diff.get("severity", "minor")] += 1

            # Re-classify if needed
            if diff.get("severity") == "minor" and ("warning" in diff.get("context", "").lower() or
                                                     "safety" in diff.get("context", "").lower()):
                try:
                    classification = classify_change_severity.invoke({
                        "change_description": f"{diff['type']}: {diff.get('old_text', '')} -> {diff.get('new_text', '')}",
                        "context": diff.get("context", ""),
                        "model": "claude-sonnet-4.5"
                    })

                    diff["severity"] = classification.get("severity", "minor")
                    severity_count[diff["severity"]] += 1

                except Exception as e:
                    print(f"Warning: Severity classification failed: {e}")

        # Update checklist
        for item in updated_checklist:
            if item["id"] == "classify_severity":
                item["status"] = "completed"
                item["completed_at"] = datetime.now()
                item["result"] = f"Classified: {severity_count['critical']} critical, {severity_count['major']} major, {severity_count['minor']} minor"

        # Determine if human review is needed
        requires_review = severity_count["critical"] > 0

        print(f"\nComparison complete:")
        print(f"  Total differences: {len(all_differences)}")
        print(f"  Critical: {severity_count['critical']}")
        print(f"  Major: {severity_count['major']}")
        print(f"  Minor: {severity_count['minor']}")
        print(f"  Requires review: {requires_review}")

        return {
            "differences": all_differences,
            "status": "generating_report",
            "current_step": "comparison_complete",
            "checklist": updated_checklist,
            "requires_review": requires_review,
        }

    except Exception as e:
        errors.append(f"Comparison failed: {str(e)}")

        return {
            "status": "failed",
            "current_step": "comparison_failed",
            "errors": errors,
        }
