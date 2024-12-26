import logging
import os
import re
from typing import Any, Dict

from mkdocs.structure.files import Files, File
from mkdocs.structure.pages import Page

from notebook_convert import convert_notebook
from generate_api_reference_links import update_markdown_with_imports

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)
DISABLED = os.getenv("DISABLE_NOTEBOOK_CONVERT") in ("1", "true", "True")


class NotebookFile(File):
    def is_documentation_page(self):
        return True


def on_files(files: Files, **kwargs: Dict[str, Any]):
    if DISABLED:
        return files
    new_files = Files([])
    for file in files:
        if file.src_path.endswith(".ipynb"):
            new_file = NotebookFile(
                path=file.src_path,
                src_dir=file.src_dir,
                dest_dir=file.dest_dir,
                use_directory_urls=file.use_directory_urls,
            )
            new_files.append(new_file)
        else:
            new_files.append(file)
    return new_files


def _highlight_code_blocks(markdown: str) -> str:
    """Find code blocks with highlight comments and add hl_lines attribute.

    Args:
        markdown: The markdown content to process.

    Returns:
        updated Markdown code with code blocks containing highlight comments
        updated to use the hl_lines attribute.
    """
    # Pattern to find code blocks with highlight comments and without
    # existing hl_lines for Python and JavaScript
    # Pattern to find code blocks with highlight comments, handling optional indentation
    code_block_pattern = re.compile(
        r"(?P<indent>[ \t]*)```(?P<language>py|python|js|javascript)(?!\s+hl_lines=)\n"
        r"(?P<code>((?:.*\n)*?))"  # Capture the code inside the block using named group
        r"(?P=indent)```"  # Match closing backticks with the same indentation
    )

    def replace_highlight_comments(match: re.Match) -> str:
        indent = match.group("indent")
        language = match.group("language")
        code_block = match.group("code")
        lines = code_block.split("\n")
        highlighted_lines = []

        # Skip initial empty lines
        while lines and not lines[0].strip():
            lines.pop(0)

        lines_to_keep = []

        comment_syntax = (
            "# highlight-next-line"
            if language in ["py", "python"]
            else "// highlight-next-line"
        )

        for line in lines:
            if comment_syntax in line:
                count = len(lines_to_keep) + 1
                highlighted_lines.append(str(count))
            else:
                lines_to_keep.append(line)

        # Reconstruct the new code block
        new_code_block = "\n".join(lines_to_keep)

        if highlighted_lines:
            return (
                f'{indent}```{language} hl_lines="{" ".join(highlighted_lines)}"\n'
                # The indent and terminating \n is already included in the code block
                f'{new_code_block}'
                f'{indent}```'
            )
        else:
            return (
                f"{indent}```{language}\n"
                # The indent and terminating \n is already included in the code block
                f"{new_code_block}"
                f"{indent}```"
            )

    # Replace all code blocks in the markdown
    markdown = code_block_pattern.sub(replace_highlight_comments, markdown)
    return markdown


def on_page_markdown(markdown: str, page: Page, **kwargs: Dict[str, Any]):
    if DISABLED:
        return markdown
    if page.file.src_path.endswith(".ipynb"):
        logger.info("Processing Jupyter notebook: %s", page.file.src_path)
        markdown = convert_notebook(page.file.abs_src_path)

    # Append API reference links to code blocks
    markdown = update_markdown_with_imports(markdown)
    # Apply highlight comments to code blocks
    markdown = _highlight_code_blocks(markdown)
    return markdown
