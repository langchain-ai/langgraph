import argparse
import ast
import glob
import os
import re
from pathlib import Path
from typing import Literal, Optional

import nbformat
from nbconvert.exporters import MarkdownExporter
from nbconvert.preprocessors import Preprocessor


def _uses_input(source: str) -> bool:
    """Parse the source code to determine if it uses the input() function."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # If there's a syntax error, assume input() might be present to be safe.
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if the function called is named 'input'
            if isinstance(node.func, ast.Name) and node.func.id == "input":
                return True
    return False


def _rewrite_cell_magic(code: str) -> str:
    """Process a code block that uses cell magic.:w

    - Lines starting with "%%capture" are ignored.
    - Lines starting with "%pip" are rewritten by removing the leading "%" character.
    - Any other non-empty line causes a NotImplementedError.

    Args:
        code (str): The original code block.

    Returns:
        str: The transformed code block.

    Raises:
        NotImplementedError: If a line doesn't start with either "%%capture" or "%pip".
    """
    rewritten_lines = []

    for line in code.splitlines():
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            continue
        # Ignore %%capture lines
        if stripped.startswith("%%capture"):
            continue
        # Rewrite %pip lines by dropping the '%'
        elif stripped.startswith("%pip"):
            # Drop the leading '%' character
            rewritten_lines.append(stripped[1:])
        # Anything else is not supported
        else:
            raise NotImplementedError(f"Unhandled line: {line}")

    return "\n".join(rewritten_lines)


class PrintCallVisitor(ast.NodeVisitor):
    """
    This visitor sets self.has_print to True if it encounters a call
    to a print within the global scope.

    This should catch calls to print(), print_stream(), etc. (Prefixed with "print").

    May have some false positives, but it's not meant to be perfect.

    Temporary code for notebook conversion.
    """

    def __init__(self):
        self.has_print = False
        self.scope_level = 0  # counter to track whether we're inside a def/lambda

    def visit_FunctionDef(self, node):
        self.scope_level += 1
        self.generic_visit(node)
        self.scope_level -= 1

    def visit_AsyncFunctionDef(self, node):
        self.scope_level += 1
        self.generic_visit(node)
        self.scope_level -= 1

    def visit_Lambda(self, node):
        self.scope_level += 1
        self.generic_visit(node)
        self.scope_level -= 1

    def visit_ClassDef(self, node):
        self.scope_level += 1
        self.generic_visit(node)
        self.scope_level -= 1

    def visit_Call(self, node):
        # Only consider calls when not inside a function definition.
        if self.scope_level == 0:
            if isinstance(node.func, ast.Name) and node.func.id.startswith("print"):
                self.has_print = True
        self.generic_visit(node)


def _has_output(source: str) -> bool:
    """Determine if the code block is expected to produce output.

    Args:
        source (str): The source code of the code block.

    Returns:
        True if the code block is expected to produce output, False otherwise.

        Must meet the following conditions:

        1. There is a call to a printing function (name starts with "print")
           that is not inside a function definition.
        2. The last top-level statement is an expression that is valid if:
         - It is any expression (including calls) AND
         - It is NOT a call to `display(...)`.

        `display` isn't handled currently by markdown-exec
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    # Condition (1): Check for a global print-like call.
    visitor = PrintCallVisitor()
    visitor.visit(tree)
    condition_a = visitor.has_print

    # Condition (2): Check the last top-level statement.
    condition_b = False
    if tree.body:
        last_stmt = tree.body[-1]
        if isinstance(last_stmt, ast.Expr):
            # If the expression is a call, ensure it's not a call to "display"
            if isinstance(last_stmt.value, ast.Call):
                if (
                    isinstance(last_stmt.value.func, ast.Name)
                    and last_stmt.value.func.id == "display"
                ):
                    condition_b = False  # exclude display-wrapped expressions
                else:
                    condition_b = True
            else:
                # Any other expression qualifies.
                condition_b = True

    return condition_a or condition_b


def _convert_links_in_markdown(markdown: str) -> str:
    """Convert links present in notebook markdown cells to standardized format.

    We want to update markdown links code cells by linking to markdown
    files rather than assuming that the link is to the finalized HTML.

    This code is needed temporarily since the markdown links that are present
    in ipython notebooks do not follow the same conventions as regular markdown
    files in mkdocs (which should link to a .md file).
    """

    # Define the regex pattern in parts for clarity:
    pattern = (
        r"(?<!!)"  # Negative lookbehind: ensure the link is not an image (i.e., doesn't start with "!")
        r"\["  # Literal '[' indicating the start of the link text.
        r"(?P<text>[^\]]*)"  # Named group 'text': match any characters except ']', representing the link text.
        r"\]"  # Literal ']' indicating the end of the link text.
        r"\("  # Literal '(' indicating the start of the URL.
        r"(?![^\)]*//)"  # Negative lookahead: ensure that the URL does not contain '//' (skip absolute URLs).
        r"(?P<url>[^)]*)"  # Named group 'url': match any characters except ')', representing the URL.
        r"\)"  # Literal ')' indicating the end of the URL.
    )

    def custom_replacement(match):
        """logic will correct the link format used in ipython notebooks

        Ipython notebooks were being converted directly into HTML links
        instead of markdown links that retain the markdown extension.

        It needs to handle the following cases:
        - optional fragments (e.g., `#section`)
            e.g., `[text](url/#section)` -> `[text](url.md#section)`
            e.g., `[text](url#section)` -> `[text](url.md#section)`
        - relative paths (e.g., `../path/to/file`) need to be denested by 1 level
        """
        text = match.group("text")
        url = match.group("url")

        if url.startswith("../"):
            # we strip the "../" from the start of the URL
            # We only need to denest one level.
            url = url[3:]

        url = url.rstrip("/")  # Strip `/` from the end of the URL

        # if url has a fragment
        if "#" in url:
            url, fragment = url.split("#")
            url = url.rstrip("/")
            # Strip `/` from the end of the URL
            return f"[{text}]({url}.md#{fragment})"
        # Otherwise add the .md extension
        return f"[{text}]({url}.md)"

    return re.sub(
        pattern,
        custom_replacement,
        markdown,
    )


class EscapePreprocessor(Preprocessor):
    def __init__(self, markdown_exec_migration: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.markdown_exec_migration = markdown_exec_migration

    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == "markdown":
            if not self.markdown_exec_migration:
                # Old logic is to convert ipynb links to HTML links
                cell.source = re.sub(
                    r"(?<!!)\[([^\]]*)\]\((?![^\)]*//)([^)]*)(?:\.ipynb)?\)",
                    r'<a href="\2">\1</a>',
                    cell.source,
                )
            else:
                cell.source = _convert_links_in_markdown(cell.source)

            # Fix image paths in <img> tags
            cell.source = re.sub(
                r'<img\s+src="\.?/img/([^"]+)"', r'<img src="../img/\1"', cell.source
            )

        elif cell.cell_type == "code":
            # Determine if the cell has bash or cell magic
            source = cell.source
            is_exec = not (
                source.startswith("%") or source.startswith("!") or _uses_input(source)
            )
            cell.metadata["exec"] = is_exec

            if self.markdown_exec_migration:
                # For markdown exec migration we'll re-write cell magic as bash commands
                if source.startswith("%%"):
                    cell.source = _rewrite_cell_magic(source)
                    cell.metadata["language"] = "shell"

                cell.metadata["has_output"] = _has_output(source)

            # Remove noqa comments
            cell.source = re.sub(r"#\s*noqa.*$", "", cell.source, flags=re.MULTILINE)
            # escape ``` in code
            # This is needed because the markdown exporter will wrap code blocks in
            # triple backticks, which will break the markdown output if the code block
            # contains triple backticks.
            cell.source = cell.source.replace("```", r"\`\`\`")
            # escape ``` in output
            if "outputs" in cell:
                filter_out = set()
                for i, output in enumerate(cell["outputs"]):
                    if "text" in output:
                        if not output["text"].strip():
                            filter_out.add(i)
                            continue

                        value = output["text"].replace("```", r"\`\`\`")
                        # handle a funky case w/ references in text
                        value = re.sub(r"\[(\d+)\](?=\[(\d+)\])", r"[\1]\\", value)
                        output["text"] = value
                    elif "data" in output:
                        for key, value in output["data"].items():
                            if isinstance(value, str):
                                value = value.replace("```", r"\`\`\`")
                                # handle a funky case w/ references in text
                                output["data"][key] = re.sub(
                                    r"\[(\d+)\](?=\[(\d+)\])", r"[\1]\\", value
                                )
                cell["outputs"] = [
                    output
                    for i, output in enumerate(cell["outputs"])
                    if i not in filter_out
                ]

        return cell, resources


class ExtractAttachmentsPreprocessor(Preprocessor):
    """
    Extracts all of the outputs from the notebook file.  The extracted
    outputs are returned in the 'resources' dictionary.
    """

    def preprocess_cell(self, cell, resources, cell_index):
        """
        Apply a transformation on each cell,
        Parameters
        ----------
        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        cell_index : int
            Index of the cell being processed (see base.py)
        """

        # Get files directory if it has been specified

        # Make sure outputs key exists
        if not isinstance(resources["outputs"], dict):
            resources["outputs"] = {}

        # Loop through all of the attachments in the cell
        for name, attach in cell.get("attachments", {}).items():
            for mime, data in attach.items():
                if mime not in {
                    "image/png",
                    "image/jpeg",
                    "image/svg+xml",
                    "application/pdf",
                }:
                    continue

                # attachments are pre-rendered. Only replace markdown-formatted
                # images with the following logic
                attach_str = f"({name})"
                if attach_str in cell.source:
                    data = f"(data:{mime};base64,{data})"
                    cell.source = cell.source.replace(attach_str, data)

        return cell, resources


exporter = MarkdownExporter(
    preprocessors=[
        EscapePreprocessor,
        ExtractAttachmentsPreprocessor,
    ],
    template_name="mdoutput",
    extra_template_basedirs=[
        os.path.join(os.path.dirname(__file__), "notebook_convert_templates")
    ],
)

md_executable = MarkdownExporter(
    preprocessors=[
        ExtractAttachmentsPreprocessor,
        EscapePreprocessor(markdown_exec_migration=True),
    ],
    template_name="md_executable",
    extra_template_basedirs=[
        os.path.join(os.path.dirname(__file__), "notebook_convert_templates")
    ],
)


def convert_notebook(
    notebook_path: Path,
    mode: Literal["markdown", "exec"] = "markdown",
) -> str:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    nb.metadata.mode = mode
    if mode == "markdown":
        body, _ = exporter.from_notebook_node(nb)
    else:
        body, _ = md_executable.from_notebook_node(nb)
    return body


HERE = Path(__file__).parent
DOCS = HERE.parent / "docs"


# Convert notebooks to markdown
def _convert_notebooks(
    *,
    output_dir: Optional[Path] = None,
    replace: bool = False,
    pattern: str = "*.ipynb",
) -> None:
    """Converting notebooks."""
    if not output_dir and not replace:
        raise ValueError("Either --output_dir or --replace must be specified")

    output_dir_path = DOCS if replace else Path(output_dir)

    # Get the directory where the script was executed
    base_dir = os.getcwd()
    # Build the full search pattern using the current working directory as the base
    full_pattern = os.path.join(base_dir, args.pattern)

    # Use glob with recursive search enabled
    matching_files = glob.glob(full_pattern, recursive=True)
    paths = [Path(file) for file in matching_files]

    file_names = [notebook.name for notebook in paths]

    for notebook in paths:
        markdown = convert_notebook(notebook, mode="exec")
        markdown_path = output_dir_path / notebook.relative_to(DOCS).with_suffix(".md")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        with open(markdown_path, "w") as f:
            f.write(markdown)
        if replace:
            notebook.unlink(missing_ok=False)

    if replace:
        # The regex will match markdown links that point to *.ipynb files.
        # It captures:
        #   group(1): the link text (inside the square brackets)
        #   group(2): the file path (without the trailing .ipynb)
        link_pattern = r"(?<!!)\[([^\]]+)\]\((?![^)]*//)([^)]+)\.ipynb\)"

        def replace_link(match: re.Match) -> str:
            link_text = match.group(1)
            link_target = match.group(2)
            # Reconstruct the file name with the .ipynb extension.
            # For example, if link_target is "foo/bar", then linked_file becomes "bar.ipynb".
            linked_file = Path(link_target).name + ".ipynb"
            # Only update if the notebook was among those converted.
            if linked_file in file_names:
                # Change the extension from .ipynb to .md
                return f"[{link_text}]({link_target}.md)"
            # Otherwise, leave the original link intact.
            return match.group(0)

        # Process all markdown files in the output directory.
        for path in output_dir_path.rglob("**/*.md"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = re.sub(link_pattern, replace_link, content)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert notebooks to markdown")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to output markdown files",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace original notebooks with markdown files",
    )
    parser.add_argument(
        "--pattern",
        default="*.ipynb",
        help="Glob pattern to match notebooks to convert",
    )
    args = parser.parse_args()
    _convert_notebooks(
        replace=args.replace,
        output_dir=args.output_dir,
        pattern=args.pattern,
    )
