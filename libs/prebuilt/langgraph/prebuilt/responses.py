"""Types for setting agent response formats."""

from __future__ import annotations

from dataclasses import dataclass
from types import UnionType
from typing import Any, Generic, Literal, TypeVar, Union, get_args, get_origin

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
            self.schema_specs = [_SchemaSpec(schema) for schema in get_args(schema)]
        else:
            self.schema_specs = [_SchemaSpec(schema)]


@dataclass(init=False)
class NativeOutput(Generic[SchemaT]):
    """Use the model provider's native structured output method."""

    schema: SchemaSpec[SchemaT]
    """Exactly one schema for native mode."""
    provider: Literal["openai", "grok"] = "openai"
    """Provider hint. Grok uses OpenAI-compatible payload, but other providers 
    may use a different format when native structured output is more widely supported.
    """

    def __init__(
        self,
        schema: SchemaT | SchemaSpec[SchemaT],
        *,
        provider: Literal["openai", "grok"] = "openai",
    ) -> None:
        if isinstance(schema, SchemaSpec):
            self.schema = schema
        else:
            self.schema = SchemaSpec(schema)
        self.provider = provider

    def to_model_kwargs(self) -> dict[str, Any]:
        model_cls = self.schema.schema
        if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
            raise ValueError(
                f"Unsupported schema type: {type(model_cls)}. Only Pydantic models are supported."
            )
            # TODO: add typed dict and dataclass support

        name = self.schema.name or model_cls.__name__
        json_schema = model_cls.model_json_schema()

        # OpenAI: 
        # - see https://platform.openai.com/docs/guides/structured-outputs
        # - response_format is converted to text_format in payload: https://github.com/keenborder786/langchain/blob/69ab335f9559911dc2f38908c39106fe8ba24f06/libs/partners/openai/langchain_openai/chat_models/base.py#L3567
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": json_schema,
            },
        }
        return {"response_format": response_format}


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
            schema=schema,  # type: ignore[arg-type]
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
ResponseFormat = Union[ToolOutput[SchemaT], NativeOutput[SchemaT]]
