"""Types for setting agent response formats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, Optional, Sequence, TypeVar, Union

from langchain_core.tools import BaseTool, tool as create_tool
from pydantic import BaseModel

# For now, we support only Pydantic models as schemas.
SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Used to detect default BaseModel docstring
BASE_MODEL_DOC = BaseModel.__doc__


@dataclass(frozen=True)
class ResponseSchema(Generic[SchemaT]):
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


@dataclass(frozen=True)
class UsingToolStrategy:
    """Use a tool calling strategy for model responses."""

    schemas: Sequence[ResponseSchema]
    """Schemas for the tool calls."""
    tool_choice: Literal["required", "auto"] = "required"
    """Whether to require tool calling or allow the model to choose.
    
    - "required": The model must use tool calling.
    - "auto": The model can choose whether to use tool calling.
    
    Use `auto` if you want the agent to be able to respond with a non structured
    response to ask clarifying questions.
    """


@dataclass(frozen=True)
class StructuredToolInfo:
    """Information for tracking structured output tool metadata.

    This contains all necessary information to handle structured responses
    generated via tool calls, including the original schema, its type classification,
    and the corresponding tool implementation used by the tools strategy.
    """

    schema: Union[type[BaseModel], dict[str, Any]]
    """The original schema provided for structured output (Pydantic model or dict schema)."""
    kind: Literal["pydantic", "dict"]
    """Classification of the schema type for proper response construction."""
    tool: BaseTool
    """LangChain tool instance created from the schema for model binding."""

    @classmethod
    def from_response_schema(
        cls, response_schema: ResponseSchema
    ) -> "StructuredToolInfo":
        """Create a StructuredToolInfo instance from a ResponseSchema.

        Args:
            response_schema: The ResponseSchema to convert

        Returns:
            A StructuredToolInfo instance with the appropriate tool created
        """
        # Extract the actual schema from ResponseSchema
        schema = response_schema.schema
        kwargs = {}

        # Use custom name if provided, otherwise use schema name
        if response_schema.name:
            kwargs["name"] = response_schema.name

        # Use custom description if provided
        if response_schema.description:
            kwargs["description"] = response_schema.description
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            # Fallback to schema docstring if no custom description
            description = "" if schema.__doc__ == BASE_MODEL_DOC else schema.__doc__
            kwargs["description"] = description

        # Determine schema kind
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            kind = "pydantic"
        else:
            kind = "dict"

        # Create the tool
        tool = create_tool(schema, **kwargs)

        return cls(
            schema=schema,
            kind=kind,
            tool=tool,
        )

    def coerce_response(self, tool_args: dict[str, Any]) -> Any:
        """Coerce tool arguments according to the schema.

        Args:
            tool_args: The arguments from the tool call

        Returns:
            The coerced response according to the schema type

        Raises:
            ValueError: If coercion fails
        """
        if self.kind == "pydantic":
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
                    f"Failed to coerce tool args to {self.schema.__name__}: {e}"
                ) from e

        elif self.kind == "dict":
            # For dict schemas, we return the tool args as-is
            # since they're already a dict
            # TODO: Add validation against dict schema if needed
            return tool_args

        else:
            raise ValueError(f"Unsupported schema kind: {self.kind}")


# TODO: This should change to a Union.
ResponseFormat = UsingToolStrategy
