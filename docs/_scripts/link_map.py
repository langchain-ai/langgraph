"""Link mapping for cross-reference resolution across different scopes.

This module provides link mappings for different language/framework scopes
to resolve @[link_name] references to actual URLs.
"""

# Python-specific link mappings
PYTHON_LINK_MAP = {
    "StateGraph": "reference/graphs/#langgraph.graph.StateGraph",
    "interrupt": "reference/graphs/#langgraph.graph.interrupt",
    "create_react_agent": "reference/prebuilt/#langgraph.prebuilt.create_react_agent",
    "Command": "reference/types/#langgraph.types.Command",
}


# JavaScript-specific link mappings
JS_LINK_MAP = {
    "StateGraph": "reference/classes/langgraph.StateGraph.html",
    "interrupt": "reference/functions/langgraph.interrupt-2.html",
    "create_react_agent": "reference/functions/langgraph_prebuilt.createReactAgent.html",
    "Command": "reference/classes/langgraph.Command.html",
}

# TODO: Allow updating these to localhost for local development
PY_REFERENCE_HOST = "https://langchain-ai.github.io/langgraph/"
JS_REFERENCE_HOST = "https://langchain-ai.github.io/langgraphjs/"

for key, value in PYTHON_LINK_MAP.items():
    # Ensure the link is absolute
    if not value.startswith("http"):
        PYTHON_LINK_MAP[key] = f"{PY_REFERENCE_HOST}{value}"

for key, value in JS_LINK_MAP.items():
    # Ensure the link is absolute
    if not value.startswith("http"):
        JS_LINK_MAP[key] = f"{JS_REFERENCE_HOST}{value}"

# Global scope is assembled from the Python and JS mappings
# Combined mapping by scope
SCOPE_LINK_MAPS = {
    "python": PYTHON_LINK_MAP,
    "js": JS_LINK_MAP,
}
