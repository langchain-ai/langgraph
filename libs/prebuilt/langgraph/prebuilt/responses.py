"""Types for setting agent response formats."""

from __future__ import annotations

import sys
from dataclasses import dataclass, is_dataclass
from typing import Any, Generic, Literal, TypeVar, Union, get_args, get_origin

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, TypeAdapter
from typing_extensions import Self, is_typeddict

# Supported schema types: Pydantic models, dataclasses, TypedDict, JSON schema dicts
SchemaT = TypeVar("SchemaT")


if sys.version_info >= (3, 10):
    from types import UnionType
else:
    UnionType = Union

SchemaKind = Literal["pydantic", "dataclass", "typeddict", "json_schema"]


def _parse_with_json_schema(
    schema: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any]:
    """Parse data using a JSON schema for validation.

    Args:
        schema: The JSON schema dict
        data: The data to parse

    Returns:
        The validated data

    Raises:
        ValueError: If validation fails
    """
    try:
        # Check required fields
        if "required" in schema and isinstance(schema["required"], list):
            missing_fields = set(schema["required"]) - set(data.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
        # Filter data to only include valid properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            valid_keys = set(schema["properties"].keys())
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            return filtered_data
        return data

    except Exception as e:
        schema_title = schema.get("title", "Schema")
        raise ValueError(
            f"Failed to validate data against JSON schema '{schema_title}': {e}"
        ) from e


def _parse_with_schema(
    schema: Union[type, dict], schema_kind: SchemaKind, data: dict[str, Any]
) -> Any:
    """Parse data using for any supported schema type.

    Args:
        schema: The schema type (Pydantic model, dataclass, or TypedDict)
        data: The data to parse

    Returns:
        The parsed instance according to the schema type

    Raises:
        ValueError: If parsing fails
    """
    if schema_kind == "json_schema":
        return _parse_with_json_schema(schema, data)
    else:
        try:
            adapter = TypeAdapter(schema)
            return adapter.validate_python(data)
        except Exception as e:
            schema_name = getattr(schema, "__name__", str(schema))
            raise ValueError(f"Failed to parse data to {schema_name}: {e}") from e


@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    """Describes a structured output schema."""

    schema: Union[type[SchemaT], dict[str, Any]]
    """The schema for the response, can be a Pydantic model, dataclass, TypedDict, or JSON schema dict."""

    name: str
    """Name of the schema, used for tool calling.
    
    If not provided, the name will be the model name or "structured_output" if it's a JSON schema.
    """

    description: str
    """Custom description of the schema. 
    
    If not provided, provided will use the model's docstring.
    """

    strict: bool = False
    """Whether to enforce strict validation of the schema."""

    schema_kind: SchemaKind
    """The kind of schema."""

    json_schema: dict[str, Any]
    """JSON schema associated with the schema."""

    def __init__(
        self,
        schema: Union[type[SchemaT], dict[str, Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool = False,
    ) -> None:
        """Initialize SchemaSpec with schema and optional parameters."""
        self.schema = schema

        self.name = name or (
            schema.get("title", "structured_output")
            if isinstance(schema, dict)
            else getattr(schema, "__name__", "structured_output")
        )

        self.description = description or (
            schema.get("description", "")
            if isinstance(schema, dict)
            else getattr(schema, "__doc__", "")
        )

        self.strict = strict

        # TODO: do we need to add title to json schema?

        if isinstance(schema, dict):
            # TODO: we could validate json schema here
            self.schema_kind = "json_schema"
            self.json_schema = schema
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            self.schema_kind = "pydantic"
            self.json_schema = schema.model_json_schema()
        elif is_dataclass(schema):
            self.schema_kind = "dataclass"
            self.json_schema = TypeAdapter(schema).json_schema()
        elif is_typeddict(schema):
            self.schema_kind = "typeddict"
            self.json_schema = TypeAdapter(schema).json_schema()
        else:
            raise ValueError(
                f"Unsupported schema type: {type(schema)}. "
                f"Supported types: Pydantic models, dataclasses, TypedDicts, and JSON schema dicts."
            )


@dataclass(init=False)
class ToolOutput(Generic[SchemaT]):
    """Use a tool calling strategy for model responses."""

    schema: Union[type[SchemaT], dict[str, Any]]
    """Schema for the tool calls."""

    tool_message_content: str
    """The content of the tool message to be returned when the model calls an artificial structured output tool."""

    schema_specs: list[_SchemaSpec[SchemaT]]
    """Schema specs for the tool calls."""

    def __init__(
        self,
        schema: Union[type[SchemaT], dict[str, Any]],
        tool_message_content: str = "ok!",
    ) -> None:
        """Initialize ToolOutput with schemas and tool message content."""
        self.schema = schema
        self.tool_message_content = tool_message_content

        if get_origin(schema) in (UnionType, Union):
            self.schema_specs = [_SchemaSpec(s) for s in get_args(schema)]
        else:
            self.schema_specs = [_SchemaSpec(schema)]


@dataclass(init=False)
class NativeOutput(Generic[SchemaT]):
    """Use the model provider's native structured output method."""

    schema: Union[type[SchemaT], dict[str, Any]]
    """Schema for native mode."""

    provider: Literal["openai", "grok"] = "openai"
    """Provider hint. Grok uses OpenAI-compatible payload, but other providers 
    may use a different format when native structured output is more widely supported.
    """

    schema_spec: _SchemaSpec[SchemaT]
    """Schema spec for native mode."""

    def __init__(
        self,
        schema: Union[type[SchemaT], dict[str, Any]],
        *,
        provider: Literal["openai", "grok"] = "openai",
    ) -> None:
        self.schema = schema
        self.provider = provider
        self.schema_spec = _SchemaSpec(schema)

    def to_model_kwargs(self) -> dict[str, Any]:
        # OpenAI:
        # - see https://platform.openai.com/docs/guides/structured-outputs
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": self.schema_spec.name,
                "schema": self.schema_spec.json_schema,
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

    schema: Union[type[SchemaT], dict[str, Any]]
    """The original schema provided for structured output (Pydantic model, dataclass, TypedDict, or JSON schema dict)."""

    schema_kind: SchemaKind
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
        return cls(
            schema=schema_spec.schema,
            schema_kind=schema_spec.schema_kind,
            tool=StructuredTool(
                args_schema=schema_spec.json_schema,
                name=schema_spec.name,
                description=schema_spec.description,
            ),
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
        return _parse_with_schema(self.schema, self.schema_kind, tool_args)


@dataclass
class NativeOutputBinding(Generic[SchemaT]):
    """Information for tracking native structured output metadata.

    This contains all necessary information to handle structured responses
    generated via native provider output, including the original schema,
    its type classification, and parsing logic for provider-enforced JSON.
    """

    schema: Union[type[SchemaT], dict[str, Any]]
    """The original schema provided for structured output (Pydantic model, dataclass, TypedDict, or JSON schema dict)."""

    schema_kind: SchemaKind
    """Classification of the schema type for proper response construction."""

    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create a NativeOutputBinding instance from a SchemaSpec.

        Args:
            schema_spec: The SchemaSpec to convert

        Returns:
            A NativeOutputBinding instance for parsing native structured output
        """
        return cls(
            schema=schema_spec.schema,
            schema_kind=schema_spec.schema_kind,
        )

    def parse(self, response: AIMessage) -> SchemaT:
        """Parse AIMessage content according to the schema.

        Args:
            response: The AI message containing the structured output

        Returns:
            The parsed response according to the schema

        Raises:
            ValueError: If text extraction, JSON parsing or schema validation fails
        """
        # Extract text content from AIMessage and parse as JSON
        raw_text = self._extract_text_content_from_message(response)

        import json

        try:
            data = json.loads(raw_text)
        except Exception as e:
            schema_name = getattr(self.schema, "__name__", "structured_output")
            raise ValueError(
                f"Native structured output expected valid JSON for {schema_name}, but parsing failed: {e}."
            ) from e

        # Parse according to schema
        return _parse_with_schema(self.schema, self.schema_kind, data)

    def _extract_text_content_from_message(self, message: AIMessage) -> str:
        """Extract text content from an AIMessage.

        Args:
            message: The AI message to extract text from

        Returns:
            The extracted text content
        """
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text" and "text" in c:
                        parts.append(str(c["text"]))
                    elif "content" in c and isinstance(c["content"], str):
                        parts.append(c["content"])
                else:
                    parts.append(str(c))
            return "".join(parts)
        return str(content)


ResponseFormat = Union[ToolOutput[SchemaT], NativeOutput[SchemaT]]
