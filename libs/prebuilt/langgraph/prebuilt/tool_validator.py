"""This module provides a ValidationNode class that can be used to validate tool calls
in a langchain graph. It applies a pydantic schema to tool_calls in the models' outputs,
and returns a ToolMessage with the validated content. If the schema is not valid, it
returns a ToolMessage with the error message. The ValidationNode can be used in a
StateGraph with a "messages" key. If multiple tool calls are requested, they will be run in parallel.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import (
    Any,
    cast,
)

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import (
    RunnableConfig,
)
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool, create_schema_from_function
from langchain_core.utils.pydantic import is_basemodel_subclass
from langgraph._internal._runnable import RunnableCallable
from langgraph.warnings import LangGraphDeprecatedSinceV10
from pydantic import BaseModel, ValidationError
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1
from typing_extensions import deprecated

logger = logging.getLogger(__name__)


def _default_format_error(
    error: BaseException,
    call: ToolCall,
    schema: type[BaseModel] | type[BaseModelV1],
) -> str:
    """Default error formatting function."""
    return f"{repr(error)}\n\nRespond after fixing all validation errors."


def _filter_output(content: str, schema: type) -> str:
    """Apply output data minimisation: only return fields explicitly declared on the schema."""
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            if issubclass(schema, BaseModel):
                allowed_fields = set(schema.model_fields.keys())
            elif issubclass(schema, BaseModelV1):
                allowed_fields = set(schema.__fields__.keys())
            else:
                allowed_fields = set(data.keys())
            filtered = {k: v for k, v in data.items() if k in allowed_fields}
            return json.dumps(filtered)
    except (json.JSONDecodeError, TypeError):
        pass
    return content


def _emit_audit_record(
    *,
    event: str,
    tool_name: str,
    tool_call_id: str,
    input_args: Any,
    outcome: str,
    error: str | None = None,
    config: RunnableConfig | None = None,
) -> None:
    """Emit a structured audit log record for a tool validation decision."""
    try:
        input_hash = hashlib.sha256(
            json.dumps(input_args, sort_keys=True, default=str).encode()
        ).hexdigest()
    except Exception:
        input_hash = "unhashable"

    record: dict[str, Any] = {
        "audit_event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "input_hash": input_hash,
        "outcome": outcome,
    }
    if error is not None:
        record["error"] = error
    if config is not None:
        record["run_id"] = str(config.get("run_id", ""))
        record["tags"] = config.get("tags", [])

    logger.info("AUDIT tool_validation %s", json.dumps(record))


@deprecated(
    "ValidationNode is deprecated. Please use `create_agent` from `langchain.agents` with custom tool error handling.",
    category=LangGraphDeprecatedSinceV10,
)
class ValidationNode(RunnableCallable):
    """A node that validates all tools requests from the last `AIMessage`.

    It can be used either in `StateGraph` with a `'messages'` key.

    !!! note

        This node does not actually **run** the tools, it only validates the tool calls,
        which is useful for extraction and other use cases where you need to generate
        structured output that conforms to a complex schema without losing the original
        messages and tool IDs (for use in multi-turn conversations).

    Returns:
        (Union[Dict[str, List[ToolMessage]], Sequence[ToolMessage]]): A list of
            `ToolMessage` objects with the validated content or error messages.

    Example:
        ```python title="Example usage for re-prompting the model to generate a valid response:"
        from typing import Literal, Annotated
        from typing_extensions import TypedDict

        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, field_validator

        from langgraph.graph import END, START, StateGraph
        from langgraph.prebuilt import ValidationNode
        from langgraph.graph.message import add_messages

        class SelectNumber(BaseModel):
            a: int

            @field_validator("a")
            def a_must_be_meaningful(cls, v):
                if v != 37:
                    raise ValueError("Only 37 is allowed")
                return v

        builder = StateGraph(Annotated[list, add_messages])
        llm = ChatOpenAI(model="gpt-4o").bind_tools([SelectNumber])
        builder.add_node("model", llm)
        builder.add_node("validation", ValidationNode([SelectNumber]))
        builder.add_edge(START, "model")

        def should_validate(state: list) -> Literal["validation", "__end__"]:
            if state[-1].tool_calls:
                return "validation"
            return END

        builder.add_conditional_edges("model", should_validate)

        def should_reprompt(state: list) -> Literal["model", "__end__"]:
            for msg in state[::-1]:
                # None of the tool calls were errors
                if msg.type == "ai":
                    return END
                if msg.additional_kwargs.get("is_error"):
                    return "model"
            return END

        builder.add_conditional_edges("validation", should_reprompt)

        graph = builder.compile()
        res = graph.invoke(("user", "Select a number, any number"))
        # Show the retry logic
        for msg in res:
            msg.pretty_print()
        ```
    """

    def __init__(
        self,
        schemas: Sequence[BaseTool | type[BaseModel] | Callable],
        *,
        format_error: Callable[[BaseException, ToolCall, type[BaseModel]], str]
        | None = None,
        name: str = "validation",
        tags: list[str] | None = None,
    ) -> None:
        """Initialize the ValidationNode.

        Args:
            schemas: A list of schemas to validate the tool calls with. These can be
                any of the following:
                - A pydantic BaseModel class
                - A BaseTool instance (the args_schema will be used)
                - A function (a schema will be created from the function signature)
            format_error: A function that takes an exception, a ToolCall, and a schema
                and returns a formatted error string. By default, it returns the
                exception repr and a message to respond after fixing validation errors.
            name: The name of the node.
            tags: A list of tags to add to the node.
        """
        super().__init__(self._func, None, name=name, tags=tags, trace=False)
        self._format_error = format_error or _default_format_error
        self.schemas_by_name: dict[str, type[BaseModel]] = {}
        for schema in schemas:
            if isinstance(schema, BaseTool):
                if schema.args_schema is None:
                    raise ValueError(
                        f"Tool {schema.name} does not have an args_schema defined."
                    )
                elif not isinstance(
                    schema.args_schema, type
                ) or not is_basemodel_subclass(schema.args_schema):
                    raise ValueError(
                        "Validation node only works with tools that have a pydantic BaseModel args_schema. "
                        f"Got {schema.name} with args_schema: {schema.args_schema}."
                    )
                self.schemas_by_name[schema.name] = schema.args_schema
            elif isinstance(schema, type) and issubclass(
                schema, (BaseModel, BaseModelV1)
            ):
                self.schemas_by_name[schema.__name__] = cast(type[BaseModel], schema)
            elif callable(schema):
                base_model = create_schema_from_function("Validation", schema)
                self.schemas_by_name[schema.__name__] = base_model
            else:
                raise ValueError(
                    f"Unsupported input to ValidationNode. Expected BaseModel, tool or function. Got: {type(schema)}."
                )

    def _get_message(
        self, input: list[AnyMessage] | dict[str, Any]
    ) -> tuple[str, AIMessage]:
        """Extract the last AIMessage from the input."""
        if isinstance(input, list):
            output_type = "list"
            messages: list = input
        elif messages := input.get("messages", []):
            output_type = "dict"
        else:
            raise ValueError("No message found in input")
        message: AnyMessage = messages[-1]
        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")
        return output_type, message

    def _func(
        self, input: list[AnyMessage] | dict[str, Any], config: RunnableConfig
    ) -> Any:
        """Validate and run tool calls synchronously."""
        output_type, message = self._get_message(input)

        # Build the allow list from registered schemas
        allow_list: set[str] = set(self.schemas_by_name.keys())

        def run_one(call: ToolCall) -> ToolMessage:
            tool_name = call["name"]
            tool_call_id = cast(str, call["id"])

            # Enforce explicit tool allow list
            if tool_name not in allow_list:
                _emit_audit_record(
                    event="tool_validation_denied",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    input_args=call.get("args"),
                    outcome="denied",
                    error=f"Tool '{tool_name}' is not in the approved allow list.",
                    config=config,
                )
                logger.warning(
                    "AUDIT tool_validation_denied: tool '%s' (call_id=%s) is not in the allow list.",
                    tool_name,
                    tool_call_id,
                )
                return ToolMessage(
                    content=f"Error: tool '{tool_name}' is not permitted.",
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    additional_kwargs={"is_error": True},
                )

            schema = self.schemas_by_name[tool_name]
            try:
                if issubclass(schema, BaseModel):
                    output = schema.model_validate(call["args"])
                    raw_content = output.model_dump_json()
                elif issubclass(schema, BaseModelV1):
                    output = schema.validate(call["args"])
                    raw_content = output.json()
                else:
                    raise ValueError(
                        f"Unsupported schema type: {type(schema)}. Expected BaseModel or BaseModelV1."
                    )
                # Apply output data minimisation
                content = _filter_output(raw_content, schema)
                _emit_audit_record(
                    event="tool_validation_success",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    input_args=call.get("args"),
                    outcome="success",
                    config=config,
                )
                return ToolMessage(
                    content=content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            except (ValidationError, ValidationErrorV1) as e:
                _emit_audit_record(
                    event="tool_validation_error",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    input_args=call.get("args"),
                    outcome="validation_error",
                    error=repr(e),
                    config=config,
                )
                return ToolMessage(
                    content=self._format_error(e, call, schema),
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    additional_kwargs={"is_error": True},
                )

        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one, message.tool_calls)]
            if output_type == "list":
                return outputs
            else:
                return {"messages": outputs}