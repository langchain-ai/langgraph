"""Types for setting agent response formats."""

from __future__ import annotations

from dataclasses import dataclass
from types import UnionType
from typing import Any, Generic, Literal, TypeVar, Union, cast, get_args, get_origin

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from pydantic import BaseModel
from typing_extensions import Self

# For now, we support only Pydantic models as schemas.
SchemaT = TypeVar("SchemaT")


@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    """Describes a structured output schema."""

    schema: type[SchemaT]
    """The schema for the response, can be a dict or a Pydantic model."""

    name: str | None = None
    """Name of the schema, used for tool calling.
    
    If not provided, the name will be the model name.
    """

    description: str | None = None
    """Custom description of the schema. 
    
    If not provided, provided will use the model's docstring.
    """
    strict: bool = False
    """Whether to enforce strict validation of the schema."""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool = False,
    ) -> None:
        """Initialize SchemaSpec with schema and optional parameters."""
        self.schema = schema
        self.name = name
        self.description = description
        self.strict = strict


@dataclass(init=False)
class ToolOutput(Generic[SchemaT]):
    """Use a tool calling strategy for model responses."""

    schema: type[SchemaT]
    """Schema for the tool calls."""

    tool_message_content: str
    """The content of the tool message to be returned when the model calls an artificial structured output tool."""

    schema_specs: list[_SchemaSpec[SchemaT]]
    """Schema specs for the tool calls."""

    def __init__(
        self, schema: type[SchemaT], tool_message_content: str = "ok!"
    ) -> None:
        """Initialize ToolOutput with schemas and tool message content."""
        self.schema = schema
        self.tool_message_content = tool_message_content

        if get_origin(schema) in (UnionType, Union):
            self.schema_specs = [_SchemaSpec(s) for s in get_args(schema)]
        else:
            self.schema_specs = [_SchemaSpec(schema)]


@dataclass
class OutputToolBinding(Generic[SchemaT]):
    """Information for tracking structured output tool metadata.

    This contains all necessary information to handle structured responses
    generated via tool calls, including the original schema, its type classification,
    and the corresponding tool implementation used by the tools strategy.
    """

    schema: type[SchemaT]
    """The original schema provided for structured output (Pydantic model or dict schema)."""

    schema_kind: Literal["pydantic"]
    """Classification of the schema type for proper response construction."""

    tool: BaseTool
    """LangChain tool instance created from the schema for model binding."""

    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create an OutputToolBinding instance from a SchemaSpec.

        Args:
            schema_spec: The SchemaSpec to convert

        Returns:
            An OutputToolBinding instance with the appropriate tool created
        """
        # Extract the actual schema from SchemaSpec
        schema = schema_spec.schema
        kwargs = {}
        schema_kind: Literal["pydantic"]

        # Use custom description if provided
        if schema_spec.description:
            kwargs["description"] = schema_spec.description

        # Determine schema kind
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_kind = "pydantic"
        else:
            raise ValueError(
                f"Unsupported schema type: {type(schema)}. "
                f"Only Pydantic models are supported."
            )

        if schema_spec.name is not None:
            tool_creator = create_tool(schema_spec.name)
        else:
            tool_creator = create_tool  # type: ignore[assignment]

        tool = tool_creator(schema, **kwargs)

        return cls(
            schema=cast(type[SchemaT], schema),
            schema_kind=schema_kind,
            tool=tool,
        )

    def parse(self, tool_args: dict[str, Any]) -> SchemaT:
        """Parse tool arguments according to the schema.

        Args:
            tool_args: The arguments from the tool call

        Returns:
            The parsed response according to the schema type

        Raises:
            ValueError: If parsing fails
        """
        if self.schema_kind == "pydantic":
            if not isinstance(self.schema, type) or not issubclass(
                self.schema, BaseModel
            ):
                raise ValueError(
                    f"Expected Pydantic model class for 'pydantic' kind, got {type(self.schema)}"
                )
            try:
                return self.schema(**tool_args)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse tool args to {self.schema.__name__}: {e}"
                ) from e
        else:
            raise ValueError(f"Unsupported schema kind: {self.schema_kind}")


# TODO: Add support for built-in structured responses (e.g., openai, grok)
ResponseFormat = ToolOutput[SchemaT]
