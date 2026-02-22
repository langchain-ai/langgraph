"""
A2A Agent Card implementation.

An Agent Card is a standardized document that describes an agent's identity,
capabilities, and how to interact with it, following the A2A protocol specification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langgraph.a2a.capabilities import A2ACapabilities
from langgraph.types import _DC_KWARGS


@dataclass(**_DC_KWARGS)
class AgentEndpoint:
    """
    Describes an endpoint where an agent can be reached.

    Attributes:
        url: The URL of the endpoint.
        protocol: The protocol used (e.g., "http", "https", "grpc").
        methods: HTTP methods supported (for HTTP endpoints).
    """

    url: str
    """The URL of the endpoint."""

    protocol: str = "https"
    """The protocol used (e.g., 'http', 'https', 'grpc')."""

    methods: tuple[str, ...] = ("POST",)
    """HTTP methods supported (for HTTP endpoints)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert endpoint to dictionary representation."""
        return {
            "url": self.url,
            "protocol": self.protocol,
            "methods": list(self.methods),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentEndpoint:
        """Create endpoint from dictionary."""
        return cls(
            url=data["url"],
            protocol=data.get("protocol", "https"),
            methods=tuple(data.get("methods", ["POST"])),
        )


@dataclass(**_DC_KWARGS)
class AgentCard:
    """
    A2A Agent Card - standardized agent identity and capability descriptor.

    The Agent Card is the primary discovery mechanism in the A2A protocol.
    It describes who the agent is, what it can do, and how to interact with it.

    Attributes:
        agent_id: Unique identifier for this agent instance.
        capabilities: The agent's capabilities definition.
        endpoints: List of endpoints where this agent can be reached.
        created_at: ISO 8601 timestamp when this card was created.
        expires_at: Optional ISO 8601 timestamp when this card expires.
        public_key: Optional public key for secure communication.
        metadata: Additional custom metadata.

    Example:
        ```python
        card = AgentCard(
            agent_id="research-agent-001",
            capabilities=A2ACapabilities(
                name="research-agent",
                description="Performs web research",
                skills=["web_search"],
            ),
            endpoints=[
                AgentEndpoint(url="https://api.example.com/agent")
            ],
        )

        # Export for discovery
        print(card.to_json())
        ```
    """

    agent_id: str
    """Unique identifier for this agent instance."""

    capabilities: A2ACapabilities
    """The agent's capabilities definition."""

    endpoints: tuple[AgentEndpoint, ...] = field(default_factory=tuple)
    """List of endpoints where this agent can be reached."""

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    """ISO 8601 timestamp when this card was created."""

    expires_at: str | None = None
    """Optional ISO 8601 timestamp when this card expires."""

    public_key: str | None = None
    """Optional public key for secure communication."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional custom metadata."""

    # A2A Protocol version this card conforms to
    _protocol_version: str = field(default="1.0", repr=False)

    def __post_init__(self) -> None:
        """Validate the agent card after initialization."""
        if not self.agent_id:
            raise ValueError("AgentCard.agent_id cannot be empty")

    def is_expired(self) -> bool:
        """Check if this agent card has expired.

        Returns:
            True if the card has expired, False otherwise.
        """
        if self.expires_at is None:
            return False
        try:
            expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) > expiry
        except ValueError:
            return False

    def has_endpoint(self, protocol: str = "https") -> bool:
        """Check if the agent has an endpoint with the specified protocol.

        Args:
            protocol: The protocol to check for.

        Returns:
            True if an endpoint with this protocol exists.
        """
        return any(ep.protocol == protocol for ep in self.endpoints)

    def get_endpoint(self, protocol: str = "https") -> AgentEndpoint | None:
        """Get the first endpoint with the specified protocol.

        Args:
            protocol: The protocol to look for.

        Returns:
            The first matching endpoint, or None if not found.
        """
        for ep in self.endpoints:
            if ep.protocol == protocol:
                return ep
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert the agent card to a dictionary representation.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "protocol_version": self._protocol_version,
            "agent_id": self.agent_id,
            "capabilities": self.capabilities.to_dict(),
            "endpoints": [ep.to_dict() for ep in self.endpoints],
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "public_key": self.public_key,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Convert the agent card to a JSON string.

        Args:
            indent: JSON indentation level. Use None for compact output.

        Returns:
            JSON string representation of the agent card.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCard:
        """Create an agent card from a dictionary.

        Args:
            data: Dictionary containing agent card data.

        Returns:
            AgentCard instance.
        """
        capabilities = A2ACapabilities.from_dict(data["capabilities"])
        endpoints = tuple(
            AgentEndpoint.from_dict(ep) for ep in data.get("endpoints", [])
        )
        return cls(
            agent_id=data["agent_id"],
            capabilities=capabilities,
            endpoints=endpoints,
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            expires_at=data.get("expires_at"),
            public_key=data.get("public_key"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> AgentCard:
        """Create an agent card from a JSON string.

        Args:
            json_str: JSON string containing agent card data.

        Returns:
            AgentCard instance.
        """
        return cls.from_dict(json.loads(json_str))
