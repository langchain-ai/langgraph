"""Link mapping for cross-reference resolution across different scopes.

This module provides link mappings for different language/framework scopes
to resolve @[link_name] references to actual URLs.
"""
# Python-specific link mappings
PYTHON_LINK_MAP = {
    "StateGraph": "https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph",
    "interrupt": "https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.interrupt",
    "create_react_agent": "https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.create_react_agent",
    "Command": "https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.Command",
}

# JavaScript-specific link mappings
JS_LINK_MAP = {
    "StateGraph": "https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph.StateGraph.html",
    "interrupt": "https://langchain-ai.github.io/langgraphjs/reference/functions/langgraph.interrupt-2.html",
    "create_react_agent": "https://langchain-ai.github.io/langgraphjs/reference/functions/langgraph_prebuilt.createReactAgent.html",
    "Command": "https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph.Command.html",
}


# Global scope is assembled from the Python and JS mappings
# Combined mapping by scope
SCOPE_LINK_MAPS = {
    "python": PYTHON_LINK_MAP,
    "js": JS_LINK_MAP,
}
