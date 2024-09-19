"""Preprocess notebooks for CI. Currently adds VCR cassettes and optionally removes pip install cells."""

import logging
import os
import json
import click
import nbformat

logger = logging.getLogger(__name__)
NOTEBOOK_DIRS = ("docs/docs/how-tos","docs/docs/tutorials")
DOCS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASSETTES_PATH = os.path.join(DOCS_PATH, "cassettes")

NOTEBOOKS_NO_CASSETTES = (
    "docs/docs/how-tos/visualization.ipynb",
    "docs/docs/how-tos/many-tools.ipynb"
)

NOTEBOOKS_NO_EXECUTION = [
    "docs/docs/tutorials/customer-support/customer-support.ipynb",
    "docs/docs/tutorials/usaco/usaco.ipynb",
]


def comment_install_cells(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        if "pip install" in cell.source:
            # Comment out the lines in cells containing "pip install"
            cell.source = "\n".join(
                f"# {line}" if line.strip() else line
                for line in cell.source.splitlines()
            )

    return notebook


def is_magic_command(code: str) -> bool:
    return code.strip().startswith("%") or code.strip().startswith("!")


def is_comment(code: str) -> bool:
    return code.strip().startswith("#")


def add_vcr_to_notebook(
    notebook: nbformat.NotebookNode, cassette_prefix: str
) -> nbformat.NotebookNode:
    """Inject `with vcr.cassette` into each code cell of the notebook."""

    # Inject VCR context manager into each code cell
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue

        lines = cell.source.splitlines()
        # skip if empty cell
        if not lines:
            continue

        are_magic_lines = [is_magic_command(line) for line in lines]

        # skip if all magic
        if all(are_magic_lines):
            continue

        if any(are_magic_lines):
            raise ValueError(
                "Cannot process code cells with mixed magic and non-magic code."
            )

        # skip if just comments
        if all(is_comment(line) or not line.strip() for line in lines):
            continue

        cell_id = cell.get("id", idx)
        cassette_name = f"{cassette_prefix}_{cell_id}.msgpack"
        cell.source = f"with custom_vcr.use_cassette('{cassette_name}', filter_headers=['x-api-key', 'authorization'], record_mode='once', serializer='msgpack'):\n" + "\n".join(
            f"    {line}" for line in lines
        )

    # Add import statement
    vcr_import_lines = [
        "import vcr",
        "import msgpack",
        "import nest_asyncio",
        "import base64",
        "",
        "custom_vcr = vcr.VCR()",
        "",
        "def msgpack_serializer(cassette_dict):",
        "    packed = msgpack.packb(cassette_dict, use_bin_type=True)",
        "    return base64.b64encode(packed).decode('utf-8')",
        "",
        "def msgpack_deserializer(cassette_string):",
        "    try:",
        "        decoded = base64.b64decode(cassette_string)",
        "        return msgpack.unpackb(decoded, raw=False)",
        "    except (ValueError, msgpack.exceptions.ExtraData, msgpack.exceptions.UnpackValueError):",
        "        # If deserialization fails, return an empty dictionary",
        "        return {\"requests\": [], \"responses\": []}",
        "",
        "# Create a custom serializer class",
        "class MsgpackSerializer:",
        "    def serialize(self, cassette_dict):",
        "        return msgpack_serializer(cassette_dict)",
        "",
        "    def deserialize(self, cassette_string):",
        "        return msgpack_deserializer(cassette_string)",
        "",
        "custom_vcr.register_serializer('msgpack', MsgpackSerializer())",
        "",
        "# Apply nest_asyncio",
        "nest_asyncio.apply()",
    ]
    import_cell = nbformat.v4.new_code_cell(source="\n".join(vcr_import_lines))
    import_cell.pop("id", None)
    notebook.cells.insert(0, import_cell)
    return notebook


def process_notebooks(should_comment_install_cells: bool) -> None:
    for directory in NOTEBOOK_DIRS:
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.endswith(".ipynb") or "ipynb_checkpoints" in root:
                    continue

                notebook_path = os.path.join(root, file)
                try:
                    notebook = nbformat.read(notebook_path, as_version=4)

                    if should_comment_install_cells:
                        notebook = comment_install_cells(notebook)

                    base_filename = os.path.splitext(os.path.basename(file))[0]
                    cassette_prefix = os.path.join(CASSETTES_PATH, base_filename)
                    if notebook_path not in NOTEBOOKS_NO_CASSETTES:
                        notebook = add_vcr_to_notebook(
                            notebook, cassette_prefix=cassette_prefix
                        )

                    if notebook_path in NOTEBOOKS_NO_EXECUTION:
                        # Add a cell at the beginning to indicate that this notebook should not be executed
                        warning_cell = nbformat.v4.new_markdown_cell(
                            source="**Warning:** This notebook is not meant to be executed automatically."
                        )
                        notebook.cells.insert(0, warning_cell)

                        # Add a special tag to the first code cell
                        if notebook.cells and notebook.cells[1].cell_type == "code":
                            notebook.cells[1].metadata["tags"] = notebook.cells[1].metadata.get("tags", []) + ["no_execution"]

                    nbformat.write(notebook, notebook_path)
                    logger.info(f"Processed: {notebook_path}")
                except Exception as e:
                    logger.error(f"Error processing {notebook_path}: {e}")
    
    with open(os.path.join(DOCS_PATH, "notebooks_no_execution.json"), "w") as f:
        json.dump(NOTEBOOKS_NO_EXECUTION, f)


@click.command()
@click.option(
    "--comment-install-cells",
    is_flag=True,
    default=False,
    help="Whether to comment out install cells",
)
def main(comment_install_cells):
    process_notebooks(should_comment_install_cells=comment_install_cells)
    logger.info("All notebooks processed successfully.")


if __name__ == "__main__":
    main()
