# Iterate on prompts

## Overview

LangGraph Studio supports two methods for modifying prompts in your graph: direct node editing and the LangSmith Playground interface.

## Direct Node Editing

Studio allows you to edit prompts used inside individual nodes, directly from the graph interface.

!!! info "Prerequisites"

    - [Assistants overview](../../concepts/assistants.md)

### Graph Configuration

Define your [configuration](https://langchain-ai.github.io/langgraph/how-tos/configuration/) to specify prompt fields and their associated nodes using `langgraph_nodes` and `langgraph_type` keys.

#### Configuration Reference

##### `langgraph_nodes`

- **Description**: Specifies which nodes of the graph a configuration field is associated with.
- **Value Type**: Array of strings, where each string is the name of a node in your graph.
- **Usage Context**: Include in the `json_schema_extra` dictionary for Pydantic models or the `metadata["json_schema_extra"]` dictionary for dataclasses.
- **Example**:
  ```python
  system_prompt: str = Field(
      default="You are a helpful AI assistant.",
      json_schema_extra={"langgraph_nodes": ["call_model", "other_node"]},
  )
  ```

##### `langgraph_type`

- **Description**: Specifies the type of configuration field, which determines how it's handled in the UI.
- **Value Type**: String
- **Supported Values**:
  - `"prompt"`: Indicates the field contains prompt text that should be treated specially in the UI.
- **Usage Context**: Include in the `json_schema_extra` dictionary for Pydantic models or the `metadata["json_schema_extra"]` dictionary for dataclasses.
- **Example**:
  ```python
  system_prompt: str = Field(
      default="You are a helpful AI assistant.",
      json_schema_extra={
          "langgraph_nodes": ["call_model"],
          "langgraph_type": "prompt",
      },
  )
  ```

#### Example Configuration

```python
## Using Pydantic
from pydantic import BaseModel, Field
from typing import Annotated, Literal

class Configuration(BaseModel):
    """The configuration for the agent."""

    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="The system prompt to use for the agent's interactions. "
        "This prompt sets the context and behavior for the agent.",
        json_schema_extra={
            "langgraph_nodes": ["call_model"],
            "langgraph_type": "prompt",
        },
    )

    model: Annotated[
        Literal[
            "anthropic/claude-3-7-sonnet-latest",
            "anthropic/claude-3-5-haiku-latest",
            "openai/o1",
            "openai/gpt-4o-mini",
            "openai/o1-mini",
            "openai/o3-mini",
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="openai/gpt-4o-mini",
        description="The name of the language model to use for the agent's main interactions. "
        "Should be in the form: provider/model-name.",
        json_schema_extra={"langgraph_nodes": ["call_model"]},
    )

## Using Dataclasses
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default="You are a helpful AI assistant.",
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent.",
            "json_schema_extra": {"langgraph_nodes": ["call_model"]},
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name.",
            "json_schema_extra": {"langgraph_nodes": ["call_model"]},
        },
    )

```

### Editing prompts in UI

1. Locate the gear icon on nodes with associated configuration fields
2. Click to open the configuration modal
3. Edit the values
4. Save to update the current assistant version or create a new one

## LangSmith Playground

The [LangSmith Playground](https://
docs.smith.langchain.com/prompt_engineering/how_to_guides#playground) interface allows testing individual LLM calls without running the full graph:

1. Select a thread
2. Click "View LLM Runs" on a node. This lists all the LLM calls (if any) made inside the node.
3. Select an LLM run to open in Playground
4. Modify prompts and test different model and tool settings
5. Copy updated prompts back to your graph

For advanced Playground features, click the expand button in the top right corner.
