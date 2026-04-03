from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentSkill:
    """A skill that an A2A agent can perform."""

    id: str
    """Slug identifier, e.g. ``web-search``."""
    name: str
    """Human-readable label, e.g. ``Web Search``."""
    description: str
    """What this skill does."""
    tags: list[str] = field(default_factory=list)
    """Freeform tags for categorisation."""
    examples: list[str] = field(default_factory=list)
    """Sample user messages that trigger this skill."""


@dataclass
class AgentCard:
    """Declarative metadata describing an A2A-compatible agent.

    Attach an `AgentCard` to a compiled LangGraph graph via
    `graph.with_agent_card(card)` so it can be served as an A2A endpoint.
    """

    name: str
    """Agent display name."""
    description: str
    """Short description (recommended <= 200 chars)."""
    url: str
    """Base URL where this agent is hosted."""
    version: str = "1.0.0"
    """Semantic version of the agent."""
    org: str = ""
    """Organisation name (shown under ``provider``)."""
    org_url: str = ""
    """Organisation URL."""
    skills: list[AgentSkill] = field(default_factory=list)
    """Capabilities this agent exposes."""
    auth_scheme: Literal["apiKey", "bearer", "none"] = "apiKey"
    """Authentication scheme advertised in the card."""
    streaming: bool = True
    """Whether streaming is supported (always True for LangGraph)."""
    push_notifications: bool = False
    """Whether push notifications are supported."""
    state_transition_history: bool = False
    """Whether state transition history is supported."""

    def to_dict(self) -> dict:
        """Serialize to the canonical A2A agent card JSON shape."""
        card: dict = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": {
                "streaming": self.streaming,
                "pushNotifications": self.push_notifications,
                "stateTransitionHistory": self.state_transition_history,
            },
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["text/plain", "application/json"],
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "tags": s.tags,
                    "examples": s.examples,
                }
                for s in self.skills
            ],
        }
        if self.org:
            card["provider"] = {"organization": self.org, "url": self.org_url}
        if self.auth_scheme != "none":
            card["securitySchemes"] = (
                {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}
                if self.auth_scheme == "apiKey"
                else {"bearer": {"type": "http", "scheme": "bearer"}}
            )
            card["security"] = (
                [{"apiKey": []}] if self.auth_scheme == "apiKey" else [{"bearer": []}]
            )
        return card
