"""Types for setting agent response formats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Literal, TypeVar, Generic

from pydantic import BaseModel

# For now, we support only Pydantic models as schemas.
SchemaT = TypeVar("SchemaT", bound=BaseModel)


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
