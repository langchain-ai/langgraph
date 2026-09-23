"""
LangGraph A2A (Agent-to-Agent) Protocol Support.

This module provides native support for Google's A2A protocol, enabling
LangGraph agents to communicate with agents from other frameworks.

The A2A protocol enables:
- Agent discovery via Agent Cards
- Standardized task routing and execution
- Capability negotiation between agents

Example:
    ```python
    from langgraph.graph import StateGraph
    from langgraph.a2a import A2ACapabilities

    capabilities = A2ACapabilities(
        name="research-agent",
        description="An agent that performs research",
        skills=["web_search", "summarization"],
    )

    graph = StateGraph(MyState)
    # ... build graph
    agent = graph.compile(a2a_capabilities=capabilities)
    card = agent.get_agent_card()
    ```
"""

from langgraph.a2a.capabilities import A2ACapabilities
from langgraph.a2a.card import AgentCard
from langgraph.a2a.message import (
    A2AMessage,
    A2AMessageType,
    A2ARequest,
    A2AResponse,
    A2ATaskStatus,
)
from langgraph.a2a.protocol import A2AProtocolHandler

__all__ = [
    "A2ACapabilities",
    "AgentCard",
    "A2AMessage",
    "A2AMessageType",
    "A2ARequest",
    "A2AResponse",
    "A2ATaskStatus",
    "A2AProtocolHandler",
]
