import argparse
import os
import re
from pathlib import Path
from typing import Literal, Optional

import nbformat
from nbconvert.exporters import MarkdownExporter
from nbconvert.preprocessors import Preprocessor


class EscapePreprocessor(Preprocessor):
    def __init__(self, rewrite_links: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rewrite_links = rewrite_links

    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == "markdown":
            if self.rewrite_links:
                # We'll need to adjust the logic for this to keep markdown format
                # but link to markdown files rather than ipynb files.
                cell.source = re.sub(
                    r"(?<!!)\[([^\]]*)\]\((?![^\)]*//)([^)]*)(?:\.ipynb)?\)",
                    r'<a href="\2">\1</a>',
                    cell.source,
                )
            else:
                # Keep format but replace the .ipynb extension with .md
                cell.source = re.sub(
                    r"(?<!!)\[([^\]]*)\]\((?![^\)]*//)([^)]*)(?:\.ipynb)?\)",
                    r"[\1](\2.md)",
                    cell.source,
                )

            # Fix image paths in <img> tags
            cell.source = re.sub(
                r'<img\s+src="\.?/img/([^"]+)"', r'<img src="../img/\1"', cell.source
            )

        elif cell.cell_type == "code":
            # Determine if the cell has bash or cell magic
            if cell.source.startswith("%") or cell.source.startswith("!"):
                # update metadata to denote that it's not a python cell
                cell.metadata["language_info"] = {"name": "unknown"}

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
    Extracts all the outputs from the notebook file.  The extracted
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

        # Loop through all the attachments in the cell
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
        EscapePreprocessor(rewrite_links=False),
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
    *, output_dir: Optional[Path] = None, replace: bool = False
) -> None:
    """Converting notebooks."""
    if not output_dir and not replace:
        raise ValueError("Either --output_dir or --replace must be specified")

    output_dir_path = DOCS if replace else Path(output_dir)
    for notebook in DOCS.rglob("*.ipynb"):
        markdown = convert_notebook(notebook, mode="exec")
        markdown_path = output_dir_path / notebook.relative_to(DOCS).with_suffix(".md")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        with open(markdown_path, "w") as f:
            f.write(markdown)
        if replace:
            notebook.unlink(missing_ok=False)

    if replace:
        # Update links in markdown files from ipynb to md files
        for path in output_dir_path.rglob("*.md"):
            with open(path, "r") as f:
                content = f.read()
            # Keep format but replace the .ipynb extension with .md
            pattern = r"(?<!!)\[([^\]]*)\]\((?![^)]*//)([^)]*)\.ipynb\)"
            replacement = r"[\1](\2.md)"

            source = re.sub(
                pattern,
                replacement,
                content,
            )
            with open(path, "w") as f:
                f.write(source)


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
    args = parser.parse_args()
    _convert_notebooks(replace=args.replace, output_dir=args.output_dir)
