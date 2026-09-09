"""
A2A Capabilities definition for LangGraph agents.

This module defines the A2ACapabilities class that specifies what an agent
can do and how it should be discovered by other agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from typing_extensions import TypedDict


class SkillDefinition(TypedDict, total=False):
    """Definition of a skill that an agent can perform."""

    name: str
    """Unique identifier for this skill."""

    description: str
    """Human-readable description of what this skill does."""

    input_schema: dict[str, Any] | None
    """JSON Schema for the skill's expected input."""

    output_schema: dict[str, Any] | None
    """JSON Schema for the skill's expected output."""

    tags: list[str]
    """Tags for categorization and discovery."""


@dataclass(frozen=True, slots=True)
class A2ACapabilities:
    """
    Defines the capabilities of a LangGraph agent for A2A protocol discovery.

    A2ACapabilities describes what an agent can do, enabling other agents
    to discover and interact with it through the A2A protocol.

    Attributes:
        name: Unique identifier for this agent. Should be URL-safe.
        description: Human-readable description of this agent's purpose.
        skills: List of skill names or detailed skill definitions.
        version: Semantic version of this agent's capabilities.
        input_schema: JSON Schema for this agent's expected input state.
        output_schema: JSON Schema for this agent's expected output state.
        tags: Tags for categorization and discovery.
        metadata: Additional custom metadata for the agent.

    Example:
        ```python
        capabilities = A2ACapabilities(
            name="research-agent",
            description="An agent that performs web research and summarization",
            skills=["web_search", "summarization", "fact_checking"],
            version="1.0.0",
            tags=["research", "nlp"],
        )
        ```
    """

    name: str
    """Unique identifier for this agent. Should be URL-safe."""

    description: str = ""
    """Human-readable description of this agent's purpose."""

    skills: tuple[str | SkillDefinition, ...] = field(default_factory=tuple)
    """List of skill names or detailed skill definitions."""

    version: str = "1.0.0"
    """Semantic version of this agent's capabilities."""

    input_schema: dict[str, Any] | None = None
    """JSON Schema for this agent's expected input state."""

    output_schema: dict[str, Any] | None = None
    """JSON Schema for this agent's expected output state."""

    tags: tuple[str, ...] = field(default_factory=tuple)
    """Tags for categorization and discovery."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional custom metadata for the agent."""

    def __post_init__(self) -> None:
        """Validate capabilities after initialization."""
        if not self.name:
            raise ValueError("A2ACapabilities.name cannot be empty")
        if not self.name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"A2ACapabilities.name must be URL-safe (alphanumeric, -, _): {self.name}"
            )

    def get_skill_names(self) -> list[str]:
        """Get a list of all skill names.

        Returns:
            List of skill name strings.
        """
        result = []
        for skill in self.skills:
            if isinstance(skill, str):
                result.append(skill)
            else:
                result.append(skill.get("name", ""))
        return [s for s in result if s]

    def has_skill(self, skill_name: str) -> bool:
        """Check if this agent has a specific skill.

        Args:
            skill_name: The name of the skill to check for.

        Returns:
            True if the agent has this skill, False otherwise.
        """
        return skill_name in self.get_skill_names()

    def to_dict(self) -> dict[str, Any]:
        """Convert capabilities to a dictionary representation.

        Returns:
            Dictionary representation suitable for serialization.
        """
        return {
            "name": self.name,
            "description": self.description,
            "skills": list(self.skills),
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2ACapabilities:
        """Create capabilities from a dictionary.

        Args:
            data: Dictionary containing capability data.

        Returns:
            A2ACapabilities instance.
        """
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            skills=tuple(data.get("skills", [])),
            version=data.get("version", "1.0.0"),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            tags=tuple(data.get("tags", [])),
            metadata=data.get("metadata", {}),
        )
