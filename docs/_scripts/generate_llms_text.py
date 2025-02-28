"""Experimental script to generate consolidated llms text from the docs."""

import glob
import os

from mkdocs.structure.files import File
from mkdocs.structure.pages import Page

from _scripts.notebook_hooks import _on_page_markdown_with_config

HERE = os.path.dirname(os.path.abspath(__file__))
# Get source directory (parent of HERE / docs)
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(HERE), "docs"))


def _make_llms_text(output_file: str) -> str:
    """Generate a consolidated text file from markdown/notebook files for LLM training.

    Args:
        output_file: Path to output the consolidated text file
    """
    # Collect all markdown and notebook files
    relative_paths = [
        # Files relative to docs/docs/
        "tutorials/introduction.ipynb",
    ]
    all_files = [os.path.join(SOURCE_DIR, path) for path in relative_paths]

    all_files.extend(
        glob.glob(os.path.join(SOURCE_DIR, "how-tos/*.md"), recursive=True)
    )
    all_files.extend(
        glob.glob(os.path.join(SOURCE_DIR, "how-tos/*.ipynb"), recursive=True)
    )
    # Add all concepts
    all_files.extend(
        glob.glob(os.path.join(SOURCE_DIR, "concepts/*.md"), recursive=True)
    )
    all_files.extend(
        glob.glob(os.path.join(SOURCE_DIR, "concepts/*.ipynb"), recursive=True)
    )

    all_content = []

    # Process each file
    for file_path in all_files:
        print(f"Processing {file_path}")
        rel_path = os.path.relpath(file_path, SOURCE_DIR)

        # Create File and Page objects to match mkdocs structure
        file_obj = File(
            path=rel_path, src_dir=SOURCE_DIR, dest_dir="", use_directory_urls=True
        )
        page = Page(
            title="",
            file=file_obj,
            config={},
        )

        # Read raw content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Convert to markdown without logic to resolve API references
        processed_content = _on_page_markdown_with_config(
            content, page, add_api_references=False, remove_base64_images=True
        )
        if processed_content:
            # Add file name
            all_content.append(f"---\n{rel_path}\n---")
            # Add content
            all_content.append(processed_content)

    # Write consolidated output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_content))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate consolidated text file from markdown/notebook files for LLMs."
        )
    )
    parser.add_argument("output_file", help="Path to output the consolidated text file")

    args = parser.parse_args()
    _make_llms_text(args.output_file)
