"""
Diff detection and analysis tools using Claude Sonnet 4.5.

Leverages Claude's native understanding to detect semantic and structural
differences between document versions.
"""

from typing import List
from anthropic import Anthropic
from langchain_core.tools import tool
import json


@tool
def detect_section_differences(
    old_sections: List[dict],
    new_sections: List[dict],
    model: str = "claude-sonnet-4.5"
) -> List[dict]:
    """
    Detect differences in document sections using Claude's analysis.

    Args:
        old_sections: List of sections from old document
        new_sections: List of sections from new document
        model: Model to use

    Returns:
        List of detected section-level differences
    """
    client = Anthropic()

    prompt = f"""
    Analyze the differences between these two sets of document sections.

    OLD SECTIONS:
    {json.dumps(old_sections, indent=2)}

    NEW SECTIONS:
    {json.dumps(new_sections, indent=2)}

    Identify:
    1. **Added sections**: Sections in NEW but not in OLD
    2. **Removed sections**: Sections in OLD but not in NEW
    3. **Renamed sections**: Sections with changed titles
    4. **Reordered sections**: Sections that moved positions
    5. **Modified sections**: Sections with content changes

    For each difference, provide:
    - type: "added", "removed", "renamed", "reordered", or "modified"
    - section_name: Name of the section
    - old_location: Page/position in old document (if applicable)
    - new_location: Page/position in new document (if applicable)
    - severity: "critical", "major", or "minor"
    - description: Brief description of the change

    Return as JSON array of differences.
    """

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return json.loads(response.content[0].text)


@tool
def analyze_text_differences(
    old_text: str,
    new_text: str,
    section_name: str,
    model: str = "claude-sonnet-4.5"
) -> List[dict]:
    """
    Perform detailed text-level difference analysis.

    Args:
        old_text: Text from old version
        new_text: Text from new version
        section_name: Name of the section being compared
        model: Model to use

    Returns:
        List of granular text differences
    """
    client = Anthropic()

    prompt = f"""
    Perform a detailed comparison of these two text versions from section: {section_name}

    OLD TEXT:
    '''
    {old_text}
    '''

    NEW TEXT:
    '''
    {new_text}
    '''

    Identify EVERY difference, no matter how small:

    1. **Word/phrase changes**: Exact words that changed
    2. **Sentence additions**: New sentences added
    3. **Sentence deletions**: Sentences removed
    4. **Numerical changes**: Any number that changed (critical for medical devices!)
    5. **Terminology changes**: Medical/technical terms that changed
    6. **Instruction changes**: Changes to procedural steps
    7. **Warning changes**: Changes to safety warnings (CRITICAL)
    8. **Formatting changes**: Bullet points, numbering, emphasis

    For each difference:
    - type: "added", "removed", "modified"
    - old_content: The text in old version
    - new_content: The text in new version
    - change_type: "word", "sentence", "number", "warning", "instruction", etc.
    - severity: "critical" (safety/medical), "major" (functional), "minor" (cosmetic)
    - context: Surrounding text for reference
    - position: Approximate position in text

    Return as JSON array. Be EXHAUSTIVE - missing changes is not acceptable.
    """

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return json.loads(response.content[0].text)


@tool
def classify_change_severity(
    change_description: str,
    context: str,
    model: str = "claude-sonnet-4.5"
) -> dict:
    """
    Classify the severity of a detected change.

    Args:
        change_description: Description of the change
        context: Context where change occurred
        model: Model to use

    Returns:
        Classification with severity and rationale
    """
    client = Anthropic()

    prompt = f"""
    Classify the severity of this change in an IFU (Instructions For Use) for a medical device.

    CHANGE: {change_description}
    CONTEXT: {context}

    Severity levels:
    - **CRITICAL**: Changes affecting safety, warnings, contraindications, adverse events,
      sterilization procedures, dosage/measurements, or regulatory information
    - **MAJOR**: Changes affecting functionality, usage instructions, technical specifications,
      performance claims, or clinical information
    - **MINOR**: Cosmetic changes, typo fixes, formatting improvements, or clarifications
      that don't affect safety or functionality

    Return JSON with:
    - severity: "critical", "major", or "minor"
    - rationale: Explanation for the classification
    - requires_review: boolean (true if should be human-reviewed)
    - regulatory_impact: boolean (true if may require regulatory submission)
    """

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return json.loads(response.content[0].text)


@tool
def identify_missing_content(
    old_document: dict,
    new_document: dict,
    model: str = "claude-sonnet-4.5"
) -> List[dict]:
    """
    Identify content that appears to be missing or incomplete in comparison.

    Args:
        old_document: Structured old document data
        new_document: Structured new document data
        model: Model to use

    Returns:
        List of potentially missing or incomplete content areas
    """
    client = Anthropic()

    prompt = f"""
    Analyze these two IFU documents for missing or incomplete content.

    OLD DOCUMENT STRUCTURE:
    {json.dumps(old_document, indent=2)[:5000]}  # Truncate if too long

    NEW DOCUMENT STRUCTURE:
    {json.dumps(new_document, indent=2)[:5000]}

    Look for:
    1. Required IFU sections that are missing in new version
    2. Regulatory-required information that was removed
    3. Safety warnings that were removed or weakened
    4. Technical specifications that are missing
    5. Instructions that appear incomplete

    For each issue:
    - type: "missing_section", "missing_warning", "incomplete_info", etc.
    - description: What is missing
    - old_location: Where it was in old document
    - severity: "critical", "major", or "minor"
    - recommendation: What should be done

    Return as JSON array.
    """

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return json.loads(response.content[0].text)


@tool
def generate_change_summary(
    all_differences: List[dict],
    model: str = "claude-sonnet-4.5"
) -> dict:
    """
    Generate an executive summary of all detected changes.

    Args:
        all_differences: Complete list of detected differences
        model: Model to use

    Returns:
        Structured summary with statistics and highlights
    """
    client = Anthropic()

    prompt = f"""
    Generate an executive summary of changes in this IFU comparison.

    ALL DIFFERENCES:
    {json.dumps(all_differences, indent=2)}

    Provide:
    1. **Statistics**:
       - Total number of changes
       - Count by type (added, removed, modified)
       - Count by severity (critical, major, minor)
       - Sections affected

    2. **Critical highlights**:
       - List all critical changes with brief description
       - Any safety-related changes
       - Regulatory-relevant changes

    3. **Overall assessment**:
       - Characterization of changes (minor update, major revision, etc.)
       - Regulatory filing implications (if any)
       - Recommended review level

    4. **Top 10 most significant changes**:
       - Ranked list with descriptions

    Return as structured JSON.
    """

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return json.loads(response.content[0].text)
