import importlib
import inspect
import logging
import os
import re
from typing import List, Literal, Optional
from typing_extensions import TypedDict

import nbformat
from nbconvert.preprocessors import Preprocessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Base URL for all class documentation
_LANGCHAIN_API_REFERENCE = "https://python.langchain.com/api_reference/"
_LANGGRAPH_API_REFERENCE = "https://langchain-ai.github.io/langgraph/reference/"


# (alias/re-exported modules, source module, class, docs namespace)
MANUAL_API_REFERENCES_LANGGRAPH = [
    (
        ["langgraph.prebuilt"],
        "langgraph.prebuilt.chat_agent_executor",
        "create_react_agent",
        "prebuilt",
    ),
    (["langgraph.prebuilt"], "langgraph.prebuilt.tool_node", "ToolNode", "prebuilt"),
    (
        ["langgraph.prebuilt"],
        "langgraph.prebuilt.tool_node",
        "tools_condition",
        "prebuilt",
    ),
    (
        ["langgraph.prebuilt"],
        "langgraph.prebuilt.tool_node",
        "InjectedState",
        "prebuilt",
    ),
    # Graph
    (["langgraph.graph"], "langgraph.graph.message", "add_messages", "graphs"),
    (["langgraph.graph"], "langgraph.graph.state", "StateGraph", "graphs"),
    (["langgraph.graph"], "langgraph.graph.state", "CompiledStateGraph", "graphs"),
    ([], "langgraph.types", "StreamMode", "types"),
    (["langgraph.graph"], "langgraph.constants", "START", "constants"),
    (["langgraph.graph"], "langgraph.constants", "END", "constants"),
    (["langgraph.constants"], "langgraph.types", "Send", "types"),
    (["langgraph.constants"], "langgraph.types", "Interrupt", "types"),
    ([], "langgraph.types", "RetryPolicy", "types"),
    ([], "langgraph.checkpoint.base", "Checkpoint", "checkpoints"),
    ([], "langgraph.checkpoint.base", "CheckpointMetadata", "checkpoints"),
    ([], "langgraph.checkpoint.base", "BaseCheckpointSaver", "checkpoints"),
    ([], "langgraph.checkpoint.base", "SerializerProtocol", "checkpoints"),
    ([], "langgraph.checkpoint.serde.jsonplus", "JsonPlusSerializer", "checkpoints"),
    ([], "langgraph.checkpoint.memory", "MemorySaver", "checkpoints"),
    ([], "langgraph.checkpoint.sqlite.aio", "AsyncSqliteSaver", "checkpoints"),
    ([], "langgraph.checkpoint.sqlite", "SqliteSaver", "checkpoints"),
    ([], "langgraph.checkpoint.postgres.aio", "AsyncPostgresSaver", "checkpoints"),
    ([], "langgraph.checkpoint.postgres", "PostgresSaver", "checkpoints"),
]

WELL_KNOWN_LANGGRAPH_OBJECTS = {
    (module_, class_): (source_module, namespace)
    for (modules, source_module, class_, namespace) in MANUAL_API_REFERENCES_LANGGRAPH
    for module_ in modules + [source_module]
}


def _make_regular_expression(pkg_prefix: str) -> re.Pattern:
    if not pkg_prefix.isidentifier():
        raise ValueError(f"Invalid package prefix: {pkg_prefix}")
    return re.compile(
        r"from\s+(" + pkg_prefix + "(?:_\w+)?(?:\.\w+)*?)\s+import\s+"
        r"((?:\w+(?:,\s*)?)*"  # Match zero or more words separated by a comma+optional ws
        r"(?:\s*\(.*?\))?)",  # Match optional parentheses block
        re.DOTALL,  # Match newlines as well
    )


# Regular expression to match langchain import lines
_IMPORT_LANGCHAIN_RE = _make_regular_expression("langchain")
_IMPORT_LANGGRAPH_RE = _make_regular_expression("langgraph")


def _get_full_module_name(module_path, class_name) -> Optional[str]:
    """Get full module name using inspect"""
    try:
        module = importlib.import_module(module_path)
        class_ = getattr(module, class_name)
        module = inspect.getmodule(class_)
        if module is None:
            # For constants, inspect.getmodule() might return None
            # In this case, we'll return the original module_path
            return module_path
        return module.__name__
    except AttributeError as e:
        logger.warning(f"Could not find module for {class_name}, {e}")
        return None
    except ImportError as e:
        logger.warning(f"Failed to load for class {class_name}, {e}")
        return None


def _get_doc_title(data: str, file_name: str) -> str:
    try:
        return re.findall(r"^#\s*(.*)", data, re.MULTILINE)[0]
    except IndexError:
        pass
    # Parse the rst-style titles
    try:
        return re.findall(r"^(.*)\n=+\n", data, re.MULTILINE)[0]
    except IndexError:
        return file_name


class ImportInformation(TypedDict):
    imported: str  # imported class name
    source: str  # module path
    docs: str  # URL to the documentation
    title: str  # Title of the document


def _get_imports(
    code: str, doc_title: str, package_ecosystem: Literal["langchain", "langgraph"]
) -> List[ImportInformation]:
    """Get imports from the given code block.

    Args:
        code: Python code block from which to extract imports
        doc_title: Title of the document
        package_ecosystem: "langchain" or "langgraph". The two live in different
            repositories and have separate documentation sites.

    Returns:
        List of import information for the given code block
    """
    imports = []

    if package_ecosystem == "langchain":
        pattern = _IMPORT_LANGCHAIN_RE
    elif package_ecosystem == "langgraph":
        pattern = _IMPORT_LANGGRAPH_RE
    else:
        raise ValueError(f"Invalid package ecosystem: {package_ecosystem}")

    for import_match in pattern.finditer(code):
        module = import_match.group(1)
        if "pydantic_v1" in module:
            continue
        imports_str = (
            import_match.group(2).replace("(\n", "").replace("\n)", "")
        )  # Handle newlines within parentheses
        # remove any newline and spaces, then split by comma
        imported_classes = [
            imp.strip()
            for imp in re.split(r",\s*", imports_str.replace("\n", ""))
            if imp.strip()
        ]
        for class_name in imported_classes:
            module_path = _get_full_module_name(module, class_name)
            if not module_path:
                continue
            if len(module_path.split(".")) < 2:
                continue

            if package_ecosystem == "langchain":
                pkg = module_path.split(".")[0].replace("langchain_", "")
                top_level_mod = module_path.split(".")[1]

                url = (
                    _LANGCHAIN_API_REFERENCE
                    + pkg
                    + "/"
                    + top_level_mod
                    + "/"
                    + module_path
                    + "."
                    + class_name
                    + ".html"
                )
            elif package_ecosystem == "langgraph":
                if (module, class_name) not in WELL_KNOWN_LANGGRAPH_OBJECTS:
                    # Likely not documented yet
                    continue

                source_module, namespace = WELL_KNOWN_LANGGRAPH_OBJECTS[
                    (module, class_name)
                ]
                url = (
                    _LANGGRAPH_API_REFERENCE
                    + namespace
                    + "/#"
                    + source_module
                    + "."
                    + class_name
                )
            else:
                raise ValueError(f"Invalid package ecosystem: {package_ecosystem}")

            # Add the import information to our list
            imports.append(
                {
                    "imported": class_name,
                    "source": module,
                    "docs": url,
                    "title": doc_title,
                }
            )

    return imports


class ImportPreprocessor(Preprocessor):
    """A preprocessor to replace imports in each Python code cell with links to their
    documentation and append the import info in a comment."""

    def preprocess(self, nb, resources):
        self.all_imports = []
        file_name = os.path.basename(resources.get("metadata", {}).get("name", ""))
        _DOC_TITLE = _get_doc_title(nb.cells[0].source, file_name)

        cells = []
        for cell in nb.cells:
            if cell.cell_type == "code":
                cells.append(cell)
                imports = _get_imports(
                    cell.source, _DOC_TITLE, "langchain"
                ) + _get_imports(cell.source, _DOC_TITLE, "langgraph")
                if not imports:
                    continue

                cells.append(
                    nbformat.v4.new_markdown_cell(
                        source=f"""
<div>
<b>API Reference:</b>
{' | '.join(f'<a href="{imp["docs"]}">{imp["imported"]}</a>' for imp in imports)}
</div>
                    """
                    )
                )
            else:
                cells.append(cell)
        nb.cells = cells
        return nb, resources
