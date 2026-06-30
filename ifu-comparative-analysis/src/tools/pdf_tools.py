"""
PDF processing tools leveraging Claude Sonnet 4.5's native vision capabilities.

The new Claude models can process PDFs natively, so we pass the PDF directly
to the model rather than extracting text separately.
"""

from typing import Optional
from pathlib import Path
import base64
from anthropic import Anthropic
from langchain_core.tools import tool


def encode_pdf_to_base64(pdf_path: str) -> str:
    """
    Encode a PDF file to base64 for sending to Claude.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Base64-encoded PDF content
    """
    with open(pdf_path, "rb") as pdf_file:
        return base64.standard_b64encode(pdf_file.read()).decode("utf-8")


@tool
def extract_pdf_structure(pdf_path: str, model: str = "claude-sonnet-4.5") -> dict:
    """
    Extract structure and content from a PDF using Claude's native vision.

    This tool leverages Claude Sonnet 4.5's ability to process PDFs directly,
    allowing it to understand layout, formatting, tables, and images.

    Args:
        pdf_path: Path to the PDF file
        model: Model to use (default: claude-sonnet-4.5)

    Returns:
        Dictionary containing:
        - sections: List of identified sections
        - pages: Page-by-page content
        - structure: Document structure metadata
        - total_pages: Total number of pages
    """
    pdf_base64 = encode_pdf_to_base64(pdf_path)

    client = Anthropic()

    prompt = """
    Analyze this IFU (Instructions For Use) document and extract its complete structure.

    Please provide:
    1. **Sections**: Identify all major sections with:
       - Section title
       - Page numbers where it starts and ends
       - Heading level (1, 2, 3, etc.)
       - Brief description

    2. **Page-by-page content**: For each page, extract:
       - Page number
       - All text content (preserve formatting)
       - Any tables (with structure)
       - Any images/diagrams (describe them)
       - Any warnings or critical information

    3. **Document metadata**:
       - Document title
       - Version/revision number
       - Date
       - Total pages
       - Language

    Return the information in a structured JSON format.
    Be extremely thorough - capture EVERY detail, as this will be used for comparative analysis.
    """

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    # Parse the response (Claude will return structured JSON)
    import json
    result = json.loads(response.content[0].text)

    return result


@tool
def extract_pdf_page_range(
    pdf_path: str,
    start_page: int,
    end_page: int,
    model: str = "claude-sonnet-4.5"
) -> str:
    """
    Extract content from a specific page range in a PDF.

    Args:
        pdf_path: Path to the PDF file
        start_page: Starting page number
        end_page: Ending page number
        model: Model to use

    Returns:
        Extracted text content from the specified pages
    """
    pdf_base64 = encode_pdf_to_base64(pdf_path)

    client = Anthropic()

    prompt = f"""
    Extract all content from pages {start_page} to {end_page} of this document.

    Include:
    - All text (preserve exact formatting and line breaks)
    - Table content (preserve structure)
    - Image descriptions
    - Any warnings, notes, or callouts

    Be extremely precise and thorough.
    """

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    return response.content[0].text


@tool
def compare_pdf_pages_visually(
    old_pdf_path: str,
    new_pdf_path: str,
    page_number: int,
    model: str = "claude-sonnet-4.5"
) -> dict:
    """
    Compare a specific page between two PDFs using Claude's vision.

    This is particularly powerful for detecting visual changes in:
    - Diagrams and images
    - Table formatting
    - Layout changes
    - Visual warnings/icons

    Args:
        old_pdf_path: Path to old version PDF
        new_pdf_path: Path to new version PDF
        page_number: Page number to compare
        model: Model to use

    Returns:
        Dictionary with detected differences
    """
    old_pdf_base64 = encode_pdf_to_base64(old_pdf_path)
    new_pdf_base64 = encode_pdf_to_base64(new_pdf_path)

    client = Anthropic()

    prompt = f"""
    Compare page {page_number} between these two versions of an IFU document.

    OLD VERSION is the first PDF.
    NEW VERSION is the second PDF.

    Identify ALL differences, including:
    1. **Text changes**: Any additions, deletions, or modifications
    2. **Visual changes**: Changes to diagrams, images, icons
    3. **Formatting changes**: Layout, font, spacing, emphasis
    4. **Table changes**: Content or structure of tables
    5. **Warning/safety changes**: Critical for medical devices

    For each difference, specify:
    - Type: "added", "removed", or "modified"
    - Location on page
    - Old content (if applicable)
    - New content (if applicable)
    - Severity: "critical" (safety/medical), "major" (functional), or "minor" (cosmetic)

    Return as structured JSON with a list of differences.
    Be EXTREMELY thorough - missing a change in an IFU could have serious consequences.
    """

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": old_pdf_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "This is the OLD version."
                    },
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": new_pdf_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    import json
    return json.loads(response.content[0].text)


@tool
def validate_pdf_accessibility(pdf_path: str) -> dict:
    """
    Validate that a PDF file exists and is accessible.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Validation result with file info
    """
    path = Path(pdf_path)

    if not path.exists():
        return {
            "valid": False,
            "error": f"File not found: {pdf_path}"
        }

    if not path.is_file():
        return {
            "valid": False,
            "error": f"Path is not a file: {pdf_path}"
        }

    if path.suffix.lower() != ".pdf":
        return {
            "valid": False,
            "error": f"File is not a PDF: {pdf_path}"
        }

    file_size = path.stat().st_size

    return {
        "valid": True,
        "path": str(path.absolute()),
        "size_bytes": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2)
    }
