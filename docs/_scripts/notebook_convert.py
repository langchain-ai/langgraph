import os
import re
from pathlib import Path

import nbformat
from nbconvert.exporters import MarkdownExporter
from nbconvert.preprocessors import Preprocessor


class EscapePreprocessor(Preprocessor):
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == "markdown":
            # rewrite markdown links to html links (excluding image links)
            cell.source = re.sub(
                r"(?<!!)\[([^\]]*)\]\((?![^\)]*//)([^)]*)(?:\.ipynb)?\)",
                r'<a href="\2">\1</a>',
                cell.source,
            )
            # Fix image paths in <img> tags
            cell.source = re.sub(
                r'<img\s+src="\.?/img/([^"]+)"', r'<img src="../img/\1"', cell.source
            )

        elif cell.cell_type == "code":
            # escape ``` in code
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


def convert_notebook(
    notebook_path: Path,
) -> Path:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    body, _ = exporter.from_notebook_node(nb)
    return body
