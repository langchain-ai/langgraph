"""Types for setting agent response formats."""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, is_dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

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


class StructuredOutputError(Exception):
    """Base class for structured output errors."""


class MultipleStructuredOutputsError(StructuredOutputError):
    """Raised when model returns multiple structured output tool calls when only one is expected."""

    def __init__(self, tool_names: list[str]):
        self.tool_names = tool_names
        super().__init__(
            f"Model incorrectly returned multiple structured responses ({', '.join(tool_names)}) when only one is expected. "
            "Consider defining a retry policy to handle this behavior: ToolOutput(..., retry_policy=StructuredOutputRetryPolicy(...))"
        )


class StructuredOutputParsingError(StructuredOutputError):
    """Raised when structured output tool call arguments fail to parse according to the schema."""

    def __init__(self, tool_name: str, parse_error: Exception):
        self.tool_name = tool_name
        self.parse_error = parse_error
        super().__init__(
            f"Failed to parse structured output for tool '{tool_name}': {parse_error}. "
            "Consider defining a retry policy to handle this behavior: ToolOutput(..., retry_policy=StructuredOutputRetryPolicy(...))"
        )


def _default_error_message_generator(
    exception: Exception, tool_name: str | None
) -> str:
    """Generate a default error message for structured output failures."""
    if isinstance(exception, MultipleStructuredOutputsError):
        return (
            "Error: You called multiple structured output tools, but only one is expected. "
            "Please call only one."
        )
    elif isinstance(exception, StructuredOutputParsingError):
        return (
            f"Error: The arguments provided to '{tool_name}' don't match the expected format. "
            f"Please check the tool schema and provide valid arguments. Details: {exception.parse_error}"
        )
    else:
        return f"Error processing structured output: {str(exception)}. Please try again with the correct format."


@dataclass
class StructuredOutputRetryPolicy:
    """Policy for handling structured output errors and retries."""

    retry_on: Union[
        type[Exception],
        Union[type[Exception]],
        Callable[[Exception], bool],
        bool,
    ] = True
    """What exceptions to retry on. Default is True (catch all exceptions)."""

    tool_message_content: str | Callable[[Exception, str], str] = (
        _default_error_message_generator
    )
    """Content for tool message sent to model on error. Can be static string or callable that takes (exception, tool_name)."""

    def should_retry(self, exception: Exception) -> bool:
        """Check if the exception should trigger a retry."""
        if isinstance(self.retry_on, bool):
            return self.retry_on

        if isinstance(self.retry_on, type) and issubclass(self.retry_on, Exception):
            return isinstance(exception, self.retry_on)

        if get_origin(self.retry_on) in (UnionType, Union):
            exception_types = get_args(self.retry_on)
            return any(isinstance(exception, exc_type) for exc_type in exception_types)

        if isinstance(self.retry_on, (list, tuple)):
            return any(isinstance(exception, exc_type) for exc_type in self.retry_on)

        if callable(self.retry_on):
            return self.retry_on(exception)  # type: ignore[call-arg]

        return False

    def get_tool_message_content(self, exception: Exception, tool_name: str) -> str:
        """Get the tool message content for the given exception."""
        if callable(self.tool_message_content):
            return self.tool_message_content(exception, tool_name)
        return self.tool_message_content


def _parse_with_schema(
    schema: Union[type[SchemaT], dict], schema_kind: SchemaKind, data: dict[str, Any]
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
        return data
    else:
        try:
            adapter: TypeAdapter[SchemaT] = TypeAdapter(schema)
            return adapter.validate_python(data)
        except Exception as e:
            schema_name = getattr(schema, "__name__", str(schema))
            raise ValueError(f"Failed to parse data to {schema_name}: {e}") from e


@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    """Describes a structured output schema."""

    schema: type[SchemaT]
    """The schema for the response, can be a Pydantic model, dataclass, TypedDict, or JSON schema dict."""

    name: str
    """Name of the schema, used for tool calling.
    
    If not provided, the name will be the model name or "response_format" if it's a JSON schema.
    """

    description: str
    """Custom description of the schema. 
    
    If not provided, provided will use the model's docstring.
    """

    schema_kind: SchemaKind
    """The kind of schema."""

    json_schema: dict[str, Any]
    """JSON schema associated with the schema."""

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

        # Schema names must be unique so we use a shortened UUID suffix
        if name:
            self.name = name
        elif isinstance(schema, dict):
            self.name = str(
                schema.get("title", f"response_format_{str(uuid.uuid4())[:4]}")
            )
        else:
            self.name = str(
                getattr(schema, "__name__", f"response_format_{str(uuid.uuid4())[:4]}")
            )

        self.description = description or (
            schema.get("description", "")
            if isinstance(schema, dict)
            else getattr(schema, "__doc__", None) or ""
        )

        self.strict = strict

        if isinstance(schema, dict):
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

    schema: type[SchemaT]
    """Schema for the tool calls."""

    schema_specs: list[_SchemaSpec[SchemaT]]
    """Schema specs for the tool calls."""

    tool_message_content: str | None
    """The content of the tool message to be returned when the model calls an artificial structured output tool."""

    retry_policy: StructuredOutputRetryPolicy | None
    """Policy for handling structured output errors and retries."""

    def __init__(
        self,
        schema: type[SchemaT],
        tool_message_content: str | None = None,
        retry_policy: StructuredOutputRetryPolicy | None = None,
    ) -> None:
        """Initialize ToolOutput with schemas, tool message content, and retry policy."""
        self.schema = schema
        self.tool_message_content = tool_message_content
        self.retry_policy = retry_policy

        def _iter_variants(schema: Any) -> Iterable[Any]:
            """Yield leaf variants from Union and JSON Schema oneOf."""

            if get_origin(schema) in (UnionType, Union):
                for arg in get_args(schema):
                    yield from _iter_variants(arg)
                return

            if isinstance(schema, dict) and "oneOf" in schema:
                for sub in schema.get("oneOf", []):
                    yield from _iter_variants(sub)
                return

            yield schema

        self.schema_specs = [_SchemaSpec(s) for s in _iter_variants(schema)]


@dataclass(init=False)
class NativeOutput(Generic[SchemaT]):
    """Use the model provider's native structured output method."""

    schema: type[SchemaT]
    """Schema for native mode."""

    schema_spec: _SchemaSpec[SchemaT]
    """Schema spec for native mode."""

    def __init__(
        self,
        schema: type[SchemaT],
    ) -> None:
        self.schema = schema
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

    schema: type[SchemaT]
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

    schema: type[SchemaT]
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
            schema_name = getattr(self.schema, "__name__", "response_format")
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
