"""This module provides a ValidationNode class that can be used to validate tool calls
in a langchain graph. It applies a pydantic schema to tool_calls in the models' outputs,
and returns a ToolMessage with the validated content. If the schema is not valid, it
returns a ToolMessage with the error message. The ValidationNode can be used in a
StateGraph with a "messages" key or in a MessageGraph. If multiple tool calls are
requested, they will be run in parallel.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
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
from pydantic import BaseModel, ValidationError
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1

from langgraph.utils.runnable import RunnableCallable


def _default_format_error(
    error: BaseException,
    call: ToolCall,
    schema: Union[Type[BaseModel], Type[BaseModelV1]],
) -> str:
    """Default error formatting function."""
    return f"{repr(error)}\n\nRespond after fixing all validation errors."


class ValidationNode(RunnableCallable):
    """A node that validates all tools requests from the last AIMessage.

    It can be used either in StateGraph with a "messages" key or in MessageGraph.

    !!! note

        This node does not actually **run** the tools, it only validates the tool calls,
        which is useful for extraction and other use cases where you need to generate
        structured output that conforms to a complex schema without losing the original
        messages and tool IDs (for use in multi-turn conversations).

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

    Returns:
        (Union[Dict[str, List[ToolMessage]], Sequence[ToolMessage]]): A list of ToolMessages with the validated content or error messages.

    Examples:
        Example usage for re-prompting the model to generate a valid response:
        >>> from typing import Literal, Annotated, TypedDict
        ...
        >>> from langchain_anthropic import ChatAnthropic
        >>> from pydantic import BaseModel, validator
        ...
        >>> from langgraph.graph import END, START, StateGraph
        >>> from langgraph.prebuilt import ValidationNode
        >>> from langgraph.graph.message import add_messages
        ...
        ...
        >>> class SelectNumber(BaseModel):
        ...     a: int
        ...
        ...     @validator("a")
        ...     def a_must_be_meaningful(cls, v):
        ...         if v != 37:
        ...             raise ValueError("Only 37 is allowed")
        ...         return v
        ...
        ...
        >>> class State(TypedDict):
        ...     messages: Annotated[list, add_messages]
        ...
        >>> builder = StateGraph(State)
        >>> llm = ChatAnthropic(model="claude-3-haiku-20240307").bind_tools([SelectNumber])
        >>> builder.add_node("model", llm)
        >>> builder.add_node("validation", ValidationNode([SelectNumber]))
        >>> builder.add_edge(START, "model")
        ...
        ...
        >>> def should_validate(state: list) -> Literal["validation", "__end__"]:
        ...     if state[-1].tool_calls:
        ...         return "validation"
        ...     return END
        ...
        ...
        >>> builder.add_conditional_edges("model", should_validate)
        ...
        ...
        >>> def should_reprompt(state: list) -> Literal["model", "__end__"]:
        ...     for msg in state[::-1]:
        ...         # None of the tool calls were errors
        ...         if msg.type == "ai":
        ...             return END
        ...         if msg.additional_kwargs.get("is_error"):
        ...             return "model"
        ...     return END
        ...
        ...
        >>> builder.add_conditional_edges("validation", should_reprompt)
        ...
        ...
        >>> graph = builder.compile()
        >>> res = graph.invoke(("user", "Select a number, any number"))
        >>> # Show the retry logic
        >>> for msg in res:
        ...     msg.pretty_print()
        ================================ Human Message =================================
        Select a number, any number
        ================================== Ai Message ==================================
        [{'id': 'toolu_01JSjT9Pq8hGmTgmMPc6KnvM', 'input': {'a': 42}, 'name': 'SelectNumber', 'type': 'tool_use'}]
        Tool Calls:
        SelectNumber (toolu_01JSjT9Pq8hGmTgmMPc6KnvM)
        Call ID: toolu_01JSjT9Pq8hGmTgmMPc6KnvM
        Args:
            a: 42
        ================================= Tool Message =================================
        Name: SelectNumber
        ValidationError(model='SelectNumber', errors=[{'loc': ('a',), 'msg': 'Only 37 is allowed', 'type': 'value_error'}])
        Respond after fixing all validation errors.
        ================================== Ai Message ==================================
        [{'id': 'toolu_01PkxSVxNxc5wqwCPW1FiSmV', 'input': {'a': 37}, 'name': 'SelectNumber', 'type': 'tool_use'}]
        Tool Calls:
        SelectNumber (toolu_01PkxSVxNxc5wqwCPW1FiSmV)
        Call ID: toolu_01PkxSVxNxc5wqwCPW1FiSmV
        Args:
            a: 37
        ================================= Tool Message =================================
        Name: SelectNumber
        {"a": 37}

    """

    def __init__(
        self,
        schemas: Sequence[Union[BaseTool, Type[BaseModel], Callable]],
        *,
        format_error: Optional[
            Callable[[BaseException, ToolCall, Type[BaseModel]], str]
        ] = None,
        name: str = "validation",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, None, name=name, tags=tags, trace=False)
        self._format_error = format_error or _default_format_error
        self.schemas_by_name: Dict[str, Type[BaseModel]] = {}
        for schema in schemas:
            if isinstance(schema, BaseTool):
                if schema.args_schema is None:
                    raise ValueError(
                        f"Tool {schema.name} does not have an args_schema defined."
                    )
                self.schemas_by_name[schema.name] = schema.args_schema
            elif isinstance(schema, type) and issubclass(
                schema, (BaseModel, BaseModelV1)
            ):
                self.schemas_by_name[schema.__name__] = cast(Type[BaseModel], schema)
            elif callable(schema):
                base_model = create_schema_from_function("Validation", schema)
                self.schemas_by_name[schema.__name__] = base_model
            else:
                raise ValueError(
                    f"Unsupported input to ValidationNode. Expected BaseModel, tool or function. Got: {type(schema)}."
                )

    def _get_message(
        self, input: Union[list[AnyMessage], dict[str, Any]]
    ) -> Tuple[str, AIMessage]:
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
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        """Validate and run tool calls synchronously."""
        output_type, message = self._get_message(input)

        def run_one(call: ToolCall):
            schema = self.schemas_by_name[call["name"]]
            try:
                if issubclass(schema, BaseModel):
                    output = schema.model_validate(call["args"])
                    content = output.model_dump_json()
                elif issubclass(schema, BaseModelV1):
                    output = schema.validate(call["args"])
                    content = output.json()
                else:
                    raise ValueError(
                        f"Unsupported schema type: {type(schema)}. Expected BaseModel or BaseModelV1."
                    )
                return ToolMessage(
                    content=content,
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                )
            except (ValidationError, ValidationErrorV1) as e:
                return ToolMessage(
                    content=self._format_error(e, call, schema),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True},
                )

        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one, message.tool_calls)]
            if output_type == "list":
                return outputs
            else:
                return {"messages": outputs}
