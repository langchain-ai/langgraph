#!/usr/bin/env python
"""Extracts typescript code blocks from a markdown file."""

import argparse
import json
import re
import os
from typing import List, TypedDict, Literal


class CodeBlock(TypedDict):
    """A code block extracted from a markdown file."""
    starting_line: int
    """The line number where the code block starts in the source file"""
    ending_line: int
    """The line number where the code block ends in the source file"""
    indentation: int
    """Number of spaces/tabs used for indentation of the code block"""
    source_file: str
    """Path to the markdown file containing this code block"""
    frontmatter: str
    """Any metadata or frontmatter specified after the opening code fence"""
    code: str
    """The actual code content within the code block"""
    language: str
    """The language of the code block (e.g. typescript, javascript)"""


def extract_code_blocks(markdown_content: str, source_file: str) -> List[CodeBlock]:
    """Extracts code blocks from a markdown file.

    Args:
        markdown_content: The content of the markdown file.
        source_file: The path to the markdown file.

    Returns:
        A list of TypedDicts, where each dict represents a code block.
    """
    # Regex to find code blocks with specified languages, capturing indentation
    # and frontmatter.
    pattern = re.compile(
        r"^(?P<indentation>\s*)```(?P<language>typescript|javascript|ts|js)(?P<frontmatter>[^\n]*)\n(?P<code>.*?)\n^(?P=indentation)```\s*$",
        re.DOTALL | re.MULTILINE,
    )

    code_blocks: List[CodeBlock] = []
    for match in pattern.finditer(markdown_content):
        start_pos = match.start()

        # Calculate line numbers
        starting_line = markdown_content.count("\n", 0, start_pos) + 1
        ending_line = starting_line + match.group(0).count("\n")

        indentation_str = match.group("indentation")

        code_block: CodeBlock = {
            "starting_line": starting_line,
            "ending_line": ending_line,
            "indentation": len(indentation_str),
            "source_file": source_file,
            "frontmatter": match.group("frontmatter").strip(),
            "code": match.group("code"),
            "language": match.group("language"),
        }
        code_blocks.append(code_block)

    return code_blocks

def dump_code_blocks(input_file: str, output_file: str, format: Literal["json", "inline"]) -> None:
    """Function to extract and save code blocks from a markdown file.

    Args:
        input_file: Path to the input markdown file.
        output_file: Path to the output JSON file for the extracted code blocks.
        format: Output format - either "json" or "inline"
    """
    with open(input_file, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    extracted_code = extract_code_blocks(markdown_content, input_file)

    if len(extracted_code) == 0:
        print(f"No code blocks found in {input_file}")
        return
    
    if format == "json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_code, f, indent=2)
    elif format == "inline":
        with open(output_file, "w", encoding="utf-8") as f:
            for code_block in extracted_code:
                f.write(f"// {json.dumps({k:v for k,v in code_block.items() if k != 'code'})}\n")
                f.write("\n")
                f.write(code_block["code"])
                f.write("\n")
    print(f"Extracted {len(extracted_code)} code blocks from {input_file} to {output_file}")

def main(input_path: str, output_path: str, format: Literal["json", "inline"]) -> None:
    """Main function to extract code blocks from a markdown file.

    Args:
        input_file: Path to the input markdown file.
        output_file: Path to the output JSON file for the extracted code blocks.
        format: Output format - either "json" or "inline"
    """
    # Check if input path is a directory
    if os.path.isdir(input_path):
        if os.path.isfile(output_path):
            raise ValueError("If input_path is a directory, output_path must also be a directory")
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
        
        # Process each markdown file in the directory recursively
        for root, _, files in os.walk(input_path):
            for filename in files:
                if filename.endswith(".md"):
                    # Get relative path to maintain directory structure
                    rel_path = os.path.relpath(root, input_path)
                    input_file = os.path.join(root, filename)
                    # Create output directory if it doesn't exist
                    output_dir = os.path.join(output_path, rel_path)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, filename.replace(".md", ".ts"))
                    dump_code_blocks(input_file, output_file, format)
    else:
        # Process single file
        dump_code_blocks(input_path, output_path, format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract typescript code blocks from a markdown file."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input markdown file.",
    )
    parser.add_argument(
        "output_file",
        help="Path to the output JSON file for the extracted code blocks.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "inline"],
        default="json",
        help="Output format - either 'json' or 'inline'",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.format)
