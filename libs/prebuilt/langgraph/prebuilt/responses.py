"""Types for setting agent response formats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, Optional, Sequence, Type, TypeVar

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from pydantic import BaseModel

# For now, we support only Pydantic models as schemas.
SchemaT = TypeVar("SchemaT", bound=type[BaseModel])
Schema = TypeVar("Schema", bound=BaseModel)

# Used to detect default BaseModel docstring
BASE_MODEL_DOC = BaseModel.__doc__

ToolChoice = Literal["required", "auto"]
"""Required: model must call a tool; auto: model may respond with free text."""


@dataclass(init=False)
class SchemaSpec(Generic[SchemaT]):
    """Describes a structured output schema."""

    schema: SchemaT
    """The schema for the response, can be a dict or a Pydantic model."""
    name: Optional[str] = None
    """Name of the schema, used for tool calling. 
    
    If not provided, the name will be the model name.
    """
    description: Optional[str] = None
    """Custom description of the schema. 
    
    If not provided, provided will use the model's docstring.
    """
    strict: bool = False
    """Whether to enforce strict validation of the schema."""

    def __init__(
        self,
        schema: SchemaT,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        """Initialize SchemaSpec with schema and optional parameters."""
        self.schema = schema
        self.name = name
        self.description = description
        self.strict = strict


@dataclass(init=False)
class ToolOutput:
    """Use a tool calling strategy for model responses."""

    schemas: Sequence[SchemaSpec]
    """Schemas for the tool calls."""
    tool_choice: ToolChoice
    """Whether to require tool calling or allow the model to choose.
    
    - "required": The model must use tool calling.
    - "auto": The model can choose whether to use tool calling.
    
    Use `auto` if you want the agent to be able to respond with a non structured
    response to ask clarifying questions.
    """

    def __init__(
        self, schemas: Sequence[SchemaSpec], *, tool_choice: ToolChoice = "required"
    ) -> None:
        """Initialize ToolOutput with schemas and tool choice."""
        self.schemas = schemas
        self.tool_choice = tool_choice


@dataclass
class OutputToolBinding(Generic[Schema]):
    """Information for tracking structured output tool metadata.

    This contains all necessary information to handle structured responses
    generated via tool calls, including the original schema, its type classification,
    and the corresponding tool implementation used by the tools strategy.
    """

    schema: Type[Schema]
    """The original schema provided for structured output (Pydantic model or dict schema)."""
    schema_kind: Literal["pydantic"]
    """Classification of the schema type for proper response construction."""
    tool: BaseTool
    """LangChain tool instance created from the schema for model binding."""

    @classmethod
    def from_schema_spec(cls, schema_spec: SchemaSpec) -> "OutputToolBinding":
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

        # Use custom name if provided, otherwise use schema name
        if schema_spec.name:
            kwargs["name"] = schema_spec.name

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
            tool_creator = create_tool(schema.name)
        else:
            tool_creator = create_tool
        tool = tool_creator(schema, **kwargs)

        return cls(
            schema=schema,
            schema_kind=schema_kind,
            tool=tool,
        )

    def parse_payload(self, tool_args: dict[str, Any]) -> Schema:
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


# TODO: Add support for native structured responses
ResponseFormat = ToolOutput
