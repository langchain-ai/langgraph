"""Generate API reference links for imports in Python code blocks within markdown files."""

import ast
import importlib
import logging
import re
from functools import lru_cache
from typing import List, Optional

from typing_extensions import TypedDict

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
    (
        [],
        "langgraph.prebuilt.chat_agent_executor",
        "AgentState",
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
    (["langgraph.graph"], "langgraph.constants", "START", "constants"),
    (["langgraph.graph"], "langgraph.constants", "END", "constants"),
    (["langgraph.constants"], "langgraph.types", "Send", "types"),
    (["langgraph.constants"], "langgraph.types", "Interrupt", "types"),
    (["langgraph.constants"], "langgraph.types", "interrupt", "types"),
    (["langgraph.constants"], "langgraph.types", "Command", "types"),
    (["langgraph.config"], "langgraph.config", "get_stream_writer", "config"),
    (["langgraph.config"], "langgraph.config", "get_store", "config"),
    (["langgraph.func"], "langgraph.func", "entrypoint", "func"),
    (["langgraph.func"], "langgraph.func", "task", "func"),
    (["langgraph.types"], "langgraph.types", "RetryPolicy", "types"),
    (["langgraph.types"], "langgraph.types", "StreamMode", "types"),
    (["langgraph.types"], "langgraph.types", "StreamWriter", "types"),
    ([], "langgraph.checkpoint.base", "Checkpoint", "checkpoints"),
    ([], "langgraph.checkpoint.base", "CheckpointMetadata", "checkpoints"),
    ([], "langgraph.checkpoint.base", "BaseCheckpointSaver", "checkpoints"),
    ([], "langgraph.checkpoint.base", "SerializerProtocol", "checkpoints"),
    ([], "langgraph.checkpoint.serde.jsonplus", "JsonPlusSerializer", "checkpoints"),
    ([], "langgraph.checkpoint.memory", "MemorySaver", "checkpoints"),
    ([], "langgraph.checkpoint.memory", "InMemorySaver", "checkpoints"),
    ([], "langgraph.checkpoint.sqlite.aio", "AsyncSqliteSaver", "checkpoints"),
    ([], "langgraph.checkpoint.sqlite", "SqliteSaver", "checkpoints"),
    ([], "langgraph.checkpoint.postgres.aio", "AsyncPostgresSaver", "checkpoints"),
    ([], "langgraph.checkpoint.postgres", "PostgresSaver", "checkpoints"),
    # other prebuilts
    (
        ["langgraph_supervisor"],
        "langgraph_supervisor.supervisor",
        "create_supervisor",
        "supervisor",
    ),
    (
        ["langgraph_supervisor"],
        "langgraph_supervisor.handoff",
        "create_handoff_tool",
        "supervisor",
    ),
    ([], "langgraph_supervisor.handoff", "create_forward_message_tool", "supervisor"),
    (["langgraph_swarm"], "langgraph_swarm.swarm", "create_swarm", "swarm"),
    (["langgraph_swarm"], "langgraph_swarm.swarm", "add_active_agent_router", "swarm"),
    (["langgraph_swarm"], "langgraph_swarm.swarm", "SwarmState", "swarm"),
    (["langgraph_swarm"], "langgraph_swarm.handoff", "create_handoff_tool", "swarm"),
    ([], "langchain_mcp_adapters.client", "MultiServerMCPClient", "mcp"),
    ([], "langchain_mcp_adapters.tools", "load_mcp_tools", "mcp"),
    ([], "langchain_mcp_adapters.prompts", "load_mcp_prompt", "mcp"),
    ([], "langchain_mcp_adapters.resources", "load_mcp_resources", "mcp"),
]

WELL_KNOWN_LANGGRAPH_OBJECTS = {
    (module_, class_): (source_module, namespace)
    for (modules, source_module, class_, namespace) in MANUAL_API_REFERENCES_LANGGRAPH
    for module_ in modules + [source_module]
}


@lru_cache(maxsize=10_000)
def _get_full_module_name(module_path: str, class_name: str) -> Optional[str]:
    """Get full module name using inspect, with LRU cache to memoize results."""
    try:
        module = importlib.import_module(module_path)
        symbol = getattr(module, class_name)
        # First check the __module__ attribute on the symbol.
        mod_name = getattr(symbol, "__module__", None)
        # If __module__ is not set or comes from typing,
        # assume the definition is in module_path.
        if mod_name is None or mod_name.startswith("typing"):
            return module_path
        return mod_name
    except AttributeError as e:
        logger.warning(f"API Reference: Could not find module for {class_name}, {e}")
        return None
    except ImportError as e:
        logger.warning(f"API Reference: Failed to load for class {class_name}, {e}")
        return None


class ImportInformation(TypedDict):
    imported: str  # The name of the class that was imported.
    source: str  # The full module path from which the class was imported.
    docs: str  # The URL pointing to the class's documentation.
    path: str  # The path of the file where the markdown content originated.


def get_imports(code: str, path: str) -> List[ImportInformation]:
    """Retrieve all import references from the given code for specified ecosystems.

    Args:
        code: The source code from which to extract import references.
        path: The path of the file where the markdown content originated.

    Returns:
        A list of import information for each import found.
    """
    # Parse the code into an AST.
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    found_imports = []

    # Walk through the AST and process ImportFrom nodes.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # node.module is the source module.
            if node.module is None:
                continue
            for alias in node.names:
                if not (
                    node.module.startswith("langchain")
                    or node.module.startswith("langgraph")
                ):
                    continue

                found_imports.append(
                    {
                        "source": node.module,
                        # alias.name is the original name even if an alias exists.
                        "imported": alias.name,
                    }
                )

    imports: list[ImportInformation] = []

    for found_import in found_imports:
        module = found_import["source"]

        if module.startswith("langchain_mcp_adapters"):
            package_ecosystem = "langgraph"
        elif module.startswith("langchain"):
            # Handles things like `langchain` or `langchain_anthropic`
            package_ecosystem = "langchain"
        elif module.startswith("langgraph"):
            package_ecosystem = "langgraph"
        else:
            continue

        class_name = found_import["imported"]
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
                "path": path,
            }
        )

    return imports


def update_markdown_with_imports(markdown: str, path: str) -> str:
    """Update markdown to include API reference links for imports in Python code blocks.

    This function scans the markdown content for Python code blocks, extracts any
    imports, and appends links to their API documentation.

    Args:
        markdown: The markdown content to process.
        path: The path of the file where the markdown content originated.

    Returns:
        Updated markdown with API reference links prepended to Python code blocks.

    Example:
        Given a markdown with a Python code block:

        ```python
        from langchain.nlp import TextGenerator
        ```
        This function will append an API reference link to the `TextGenerator` class
        from the `langchain.nlp` module if it's recognized.
    """
    code_block_pattern = re.compile(
        r"(?P<indent>[ \t]*)```(?P<language>python|py)\n(?P<code>.*?)\n(?P=indent)```",
        re.DOTALL,
    )

    def replace_code_block(match: re.Match) -> str:
        """Replace the matched code block with additional API reference links if imports are found.

        Args:
            match (re.Match): The regex match object containing the code block.

        Returns:
            str: The modified code block with API reference links prepended if applicable.
        """
        indent = match.group("indent")
        code_block = match.group("code")
        # Retrieve import information from the code block
        imports = get_imports(code_block, "__unused__")

        original_code_block = match.group(0)
        # If no imports are found, return the original code block
        if not imports:
            return original_code_block

        # Generate API reference links for each import
        api_links = " | ".join(
            f'<a href="{imp["docs"]}">{imp["imported"]}</a>' for imp in imports
        )
        # Return the code block with prepended API reference links
        return f"{indent}<sup><i>API Reference: {api_links}</i></sup>\n\n{original_code_block}"

    # Apply the replace_code_block function to all matches in the markdown
    updated_markdown = code_block_pattern.sub(replace_code_block, markdown)
    return updated_markdown
