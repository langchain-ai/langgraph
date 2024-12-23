import importlib
import inspect
import logging
import os
import re
from typing import List, Literal, Optional
from typing_extensions import TypedDict


from functools import lru_cache

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
    (["langgraph.constants"], "langgraph.types", "interrupt", "types"),
    (["langgraph.constants"], "langgraph.types", "Command", "types"),
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




@lru_cache(maxsize=10_000)
def _get_full_module_name(module_path: str, class_name: str) -> Optional[str]:
    """Get full module name using inspect, with LRU cache to memoize results."""
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
        logger.warning(f"API Reference: Could not find module for {class_name}, {e}")
        return None
    except ImportError as e:
        logger.warning(f"API Reference: Failed to load for class {class_name}, {e}")
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
    imported: str  # The name of the class that was imported.
    source: str  # The full module path from which the class was imported.
    docs: str  # The URL pointing to the class's documentation.
    title: str  # The title of the document where the import is used.


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


def get_imports(code: str, doc_title: str) -> List[ImportInformation]:
    """Retrieve all import references from the given code for specified ecosystems.

    Args:
        code: The source code from which to extract import references.
        doc_title: The documentation title associated with the code.

    Returns:
        A list of import information for each import found.
    """
    ecosystems = ["langchain", "langgraph"]
    all_imports = []
    for package_ecosystem in ecosystems:
        all_imports.extend(_get_imports(code, doc_title, package_ecosystem))
    return all_imports


def update_markdown_with_imports(markdown: str) -> str:
    """Update markdown to include API reference links for imports in Python code blocks.

    This function scans the markdown content for Python code blocks, extracts any imports, and appends links to their API documentation.

    Args:
        markdown: The markdown content to process.

    Returns:
        Updated markdown with API reference links appended to Python code blocks.

    Example:
        Given a markdown with a Python code block:

        ```python
        from langchain.nlp import TextGenerator
        ```
        This function will append an API reference link to the `TextGenerator` class from the `langchain.nlp` module if it's recognized.
    """
    code_block_pattern = re.compile(
        r'(?P<indent>[ \t]*)```(?P<language>python|py)\n(?P<code>.*?)\n(?P=indent)```', re.DOTALL
    )

    def replace_code_block(match: re.Match) -> str:
        """Replace the matched code block with additional API reference links if imports are found.

        Args:
            match (re.Match): The regex match object containing the code block.

        Returns:
            str: The modified code block with API reference links appended if applicable.
        """
        indent = match.group('indent')
        code_block = match.group('code')
        language = match.group('language')  # Preserve the language from the regex match
        # Retrieve import information from the code block
        imports = get_imports(code_block, "__unused__")

        original_code_block = match.group(0)
        # If no imports are found, return the original code block
        if not imports:
            return original_code_block

        # Generate API reference links for each import
        api_links = ' | '.join(
            f'<a href="{imp["docs"]}">{imp["imported"]}</a>' for imp in imports
        )
        # Return the code block with appended API reference links
        return f'{original_code_block}\n\n{indent}API Reference: {api_links}'

    # Apply the replace_code_block function to all matches in the markdown
    updated_markdown = code_block_pattern.sub(replace_code_block, markdown)
    return updated_markdown