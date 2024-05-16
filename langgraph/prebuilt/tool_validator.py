"""This module provides a ValidationNode class that can be used to validate tool calls
in a langchain graph. It applies a pydantic schema to tool_calls in the models' outputs,
and returns a ToolMessage with the validated content. If the schema is not valid, it
returns a ToolMessage with the error message. The ValidationNode can be used in a
StateGraph with a "messages" key or in a MessageGraph. If multiple tool calls are
requested, they will be run in parallel.
"""

import asyncio
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
from langchain_core.pydantic_v1 import BaseModel, ValidationError
from langchain_core.runnables import (
    RunnableConfig,
)
from langchain_core.runnables import (
    chain as as_runnable,
)
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool, create_schema_from_function
from pydantic import BaseModel as BaseModelV2

from langgraph.utils import RunnableCallable


def _default_format_error(
    error: BaseException, call: ToolCall, schema: Type[BaseModel]
) -> str:
    """Default error formatting function."""
    return f"{repr(error)}\n\nRespond after fixing all validation errors."


class ValidationNode(RunnableCallable):
    """A node that validates and runs the tools requested in the last AIMessage.

    It can be used either in StateGraph with a "messages" key or in MessageGraph.
    If multiple tool calls are requested, they will be run in parallel. The output
    will be a list of ToolMessages, one for each tool call.

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

    Examples:
        You can use this for things like re-prompting the model when it generates
        invalid output.
        >>> from typing import Literal
        ... 
        >>> from langchain_anthropic import ChatAnthropic
        >>> from langchain_core.pydantic_v1 import BaseModel, validator
        ... 
        >>> from langgraph.graph import END, START, MessageGraph
        >>> from langgraph.prebuilt import ValidationNode
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
        >>> builder = MessageGraph()
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
        >>> def get_ai_message(state: list):
        ...     for msg in state[::-1]:
        ...         if msg.type == "ai":
        ...             return msg
        ...     raise ValueError("No AI message found")
        ... 
        ... 
        >>> graph = builder.compile()
        >>> res = graph.invoke(("user", "Select a number, any number"))
        >>> res[-2].pretty_print()
        ================================== Ai Message ==================================

        [{'text': 'Apologies, it seems the `SelectNumber` function only accepts the number 37. Let me try that again:', 'type': 'text'}, {'id': 'toolu_01G4ivTrLGZnYdQ4MY8pL2cc', 'input': {'a': 37}, 'name': 'SelectNumber', 'type': 'tool_use'}]
        Tool Calls:
        SelectNumber (toolu_01G4ivTrLGZnYdQ4MY8pL2cc)
        Call ID: toolu_01G4ivTrLGZnYdQ4MY8pL2cc
        Args:
            a: 37

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
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
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
                schema, (BaseModel, BaseModelV2)
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

        @as_runnable
        def run_one(call: ToolCall):
            schema = self.schemas_by_name[call["name"]]
            try:
                output = schema.validate(call["args"])
                return ToolMessage(
                    content=output.json(),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                )
            except ValidationError as e:
                return ToolMessage(
                    content=self._format_error(e, call, schema),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True},
                )

        with get_executor_for_config(config) as executor:
            outputs = [
                *executor.map(lambda x: run_one.invoke(x, config), message.tool_calls)
            ]
            if output_type == "list":
                return outputs
            else:
                return {"messages": outputs}

    async def _afunc(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        """Validate and run tool calls asynchronously."""
        output_type, message = self._get_message(input)

        @as_runnable
        async def run_one(call: ToolCall):
            schema = self.schemas_by_name[call["name"]]
            try:
                output = schema.validate(call["args"])
                return ToolMessage(
                    content=output.json(),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                )
            except ValidationError as e:
                return ToolMessage(
                    content=self._format_error(e, call, schema),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True},
                )

        outputs = await asyncio.gather(
            *(run_one.ainvoke(call, config) for call in message.tool_calls)
        )
        if output_type == "list":
            return outputs
        else:
            return {"messages": outputs}
