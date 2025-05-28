"""Preprocess notebooks for CI. Currently adds VCR cassettes and optionally removes pip install cells."""

import logging
import os
import json
import click
import nbformat
import re

logger = logging.getLogger(__name__)
NOTEBOOK_DIRS = ("docs/how-tos","docs/tutorials")
DOCS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASSETTES_PATH = os.path.join(DOCS_PATH, "cassettes")

BLOCKLIST_COMMANDS = (
    # skip if has WebBaseLoader to avoid caching web pages
    "WebBaseLoader",
    # skip if has draw_mermaid_png to avoid generating mermaid images via API
    "draw_mermaid_png",
)

NOTEBOOKS_NO_CASSETTES = (
    "docs/how-tos/visualization.ipynb",
)

NOTEBOOKS_NO_EXECUTION = [
    # this uses a user provided project name for langsmith
    "docs/tutorials/tnt-llm/tnt-llm.ipynb",
    # this uses langsmith datasets
    "docs/tutorials/chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation.ipynb",
    # this uses browser APIs
    "docs/tutorials/web-navigation/web_voyager.ipynb",
    # these RAG guides use an ollama model
    "docs/tutorials/rag/langgraph_adaptive_rag_local.ipynb",
    "docs/tutorials/rag/langgraph_crag_local.ipynb",
    "docs/tutorials/rag/langgraph_self_rag_local.ipynb",
    # this loads a massive dataset from gcp
    "docs/tutorials/usaco/usaco.ipynb",
    # TODO: figure out why autogen notebook is not runnable (they are just hanging. possible due to code execution?)
    "docs/how-tos/autogen-integration.ipynb",
    "docs/how-tos/autogen-integration-functional.ipynb",
    # TODO: need to update these notebooks to make sure they are runnable in CI
    "docs/tutorials/storm/storm.ipynb",  # issues only when running with VCR
    "docs/tutorials/lats/lats.ipynb",  # issues only when running with VCR
    "docs/tutorials/rag/langgraph_crag.ipynb",  # flakiness from tavily
    "docs/tutorials/rag/langgraph_adaptive_rag.ipynb",  # flakiness only when running in GHA 
    "docs/tutorials/rag/langgraph_self_rag.ipynb",  # flakiness only when running in GHA
    "docs/tutorials/rag/langgraph_agentic_rag.ipynb",  # flakiness only when running in GHA
    "docs/how-tos/map-reduce.ipynb",  # flakiness from structured output, only when running with VCR
    "docs/tutorials/tot/tot.ipynb",
    "docs/how-tos/visualization.ipynb",
    "docs/how-tos/streaming-specific-nodes.ipynb",
    "docs/tutorials/llm-compiler/LLMCompiler.ipynb",
    "docs/tutorials/customer-support/customer-support.ipynb",  # relies on openai embeddings, doesn't play well w/ VCR
    "docs/how-tos/many-tools.ipynb",  # relies on openai embeddings, doesn't play well w/ VCR
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


def has_blocklisted_command(code: str, metadata: dict) -> bool:
    if 'hide_from_vcr' in metadata:
        return True
    
    code = code.strip()
    for blocklisted_pattern in BLOCKLIST_COMMANDS:
        if blocklisted_pattern in code:
            return True
    return False

MERMAID_PATTERN = re.compile(r'display\(Image\((\w+)\.get_graph\(\)\.draw_mermaid_png\(\)\)\)')

def remove_mermaid(code: str) -> str:
    return MERMAID_PATTERN.sub('print()', code)


def add_vcr_to_notebook(
    notebook: nbformat.NotebookNode, cassette_prefix: str
) -> nbformat.NotebookNode:
    """Inject `with vcr.cassette` into each code cell of the notebook."""

    uses_langsmith = False
    # Inject VCR context manager into each code cell
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue

        lines = cell.source.splitlines()
        # remove the special tag for hidden cells
        lines = [line for line in lines if not line.strip().startswith("# hide-cell")]
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

        if has_blocklisted_command(cell.source, cell.metadata):
            continue

        cell_id = cell.get("id", idx)
        cassette_name = f"{cassette_prefix}_{cell_id}.msgpack.zlib"
        cell.source = f"with custom_vcr.use_cassette('{cassette_name}', filter_headers=['x-api-key', 'authorization'], record_mode='once', serializer='advanced_compressed'):\n" + "\n".join(
            f"    {line}" for line in lines
        )

        if any("hub.pull" in line or "from langsmith import" in line for line in lines):
            uses_langsmith = True

    # Add import statement
    vcr_import_lines = []
    if uses_langsmith:
        vcr_import_lines.extend([
            # patch urllib3 to handle vcr errors, see more here:
            # https://github.com/langchain-ai/langsmith-sdk/blob/main/python/langsmith/_internal/_patch.py
            "import sys",
            f"sys.path.insert(0, '{os.path.join(DOCS_PATH, '_scripts')}')",
            "import _patch as patch_urllib3",
            "patch_urllib3.patch_urllib3()",
        ])

    vcr_import_lines.extend([
        "import nest_asyncio",
        "nest_asyncio.apply()",
        "import vcr",
        "import msgpack",
        "import base64",
        "import zlib",
        "import os",
        "os.environ.pop(\"LANGCHAIN_TRACING_V2\", None)",
        "custom_vcr = vcr.VCR()",
        "",
        "def compress_data(data, compression_level=9):",
        "    packed = msgpack.packb(data, use_bin_type=True)",
        "    compressed = zlib.compress(packed, level=compression_level)",
        "    return base64.b64encode(compressed).decode('utf-8')",
        "",
        "def decompress_data(compressed_string):",
        "    decoded = base64.b64decode(compressed_string)",
        "    decompressed = zlib.decompress(decoded)",
        "    return msgpack.unpackb(decompressed, raw=False)",
        "",
        "class AdvancedCompressedSerializer:",
        "    def serialize(self, cassette_dict):",
        "        return compress_data(cassette_dict)",
        "",
        "    def deserialize(self, cassette_string):",
        "        return decompress_data(cassette_string)",
        "",
        "custom_vcr.register_serializer('advanced_compressed', AdvancedCompressedSerializer())",
        "custom_vcr.serializer = 'advanced_compressed'",
    ])

    import_cell = nbformat.v4.new_code_cell(source="\n".join(vcr_import_lines))
    import_cell.pop("id", None)
    notebook.cells.insert(0, import_cell)
    return notebook


def remove_mermaid_from_notebook(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        cell.source = remove_mermaid(cell.source)

        # skip the cell entirely if it contains PYPPETEER
        if "PYPPETEER" in cell.source:
            cell.source = ""

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

                    notebook = remove_mermaid_from_notebook(notebook)

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
    
    with open("notebooks_no_execution.json", "w") as f:
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
