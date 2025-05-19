from copy import deepcopy
from typing import Any, Literal, Optional, Union, cast

from langchain_core.messages import ToolCall, ToolMessage
from typing_extensions import TypedDict

from langgraph.types import Command, interrupt
from langgraph.utils.runnable import RunnableCallable


class HumanInterruptConfig(TypedDict):
    """Configuration that defines what actions are allowed for a human interrupt.

    This controls the available interaction options when the graph is paused for human input.

    Attributes:
        allow_ignore: Whether the human can choose to ignore/skip the current step
        allow_respond: Whether the human can provide a text response/feedback
        allow_edit: Whether the human can edit the provided content/state
        allow_accept: Whether the human can accept/approve the current state
    """

    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool


class ActionRequest(TypedDict):
    """Represents a request for human action within the graph execution.

    Contains the action type and any associated arguments needed for the action.

    Attributes:
        action: The type or name of action being requested (e.g., "Approve XYZ action")
        args: Key-value pairs of arguments needed for the action
    """

    action: str
    args: dict


class HumanInterrupt(TypedDict):
    """Represents an interrupt triggered by the graph that requires human intervention.

    This is passed to the `interrupt` function when execution is paused for human input.

    Attributes:
        action_request: The specific action being requested from the human
        config: Configuration defining what actions are allowed
        description: Optional detailed description of what input is needed

    Example:
        ```python
        # Extract a tool call from the state and create an interrupt request
        request = HumanInterrupt(
            action_request=ActionRequest(
                action="run_command",  # The action being requested
                args={"command": "ls", "args": ["-l"]}  # Arguments for the action
            ),
            config=HumanInterruptConfig(
                allow_ignore=True,    # Allow skipping this step
                allow_respond=True,   # Allow text feedback
                allow_edit=False,     # Don't allow editing
                allow_accept=True     # Allow direct acceptance
            ),
            description="Please review the command before execution"
        )
        # Send the interrupt request and get the response
        response = interrupt([request])[0]
        ```
    """

    action_request: ActionRequest
    config: HumanInterruptConfig
    description: Optional[str]


class HumanResponse(TypedDict):
    """The response provided by a human to an interrupt, which is returned when graph execution resumes.

    Attributes:
        type: The type of response:
            - "accept": Approves the current state without changes
            - "ignore": Skips/ignores the current step
            - "response": Provides text feedback or instructions
            - "edit": Modifies the current state/content
        arg: The response payload:
            - None: For ignore/accept actions
            - str: For text responses
            - ActionRequest: For edit actions with updated content
    """

    type: Literal["accept", "ignore", "response", "edit"]
    args: Union[None, str, ActionRequest]


class InterruptToolNode(RunnableCallable):
    """Prebuilt post model hook node used to enable common patterns for tool interrupts.

    For any tools with specified policies, an interrupt will be raised when the LLM returns
    a tool call for said tool. The interrupt policy will be used to determine what sort of resume logic is allowed.
    Any of the following resume patterns are supported:

    * accept: the tool call is executed as planned
    * edit: the args for the tool call are edited and then the tool call is executed
    * response: text response/feedback is fed back into the LLM
    * ignore: the current tool call is ignored / skipped

    Args:
        **interrupt_policy: a mapping of tool names to [`HumanInterruptConfig`][prebuilt.interrupt.HumanInterruptConfig] dictionaries
            specifying which interrupt patterns to enable for said tool.

        Example:
        ```python
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.prebuilt.interrupt import HumanInterruptConfig, InterruptToolNode
        from langgraph.types import Command


        def book_hotel(hotel_name: str) -> str:
            '''Book a room at the provided hotel.'''
            # Some hotel API calls, a sensitive / expensive operation
            return f"Booked a hotel at {hotel_name}."


        agent = create_react_agent(
            "openai:gpt-4.1",
            tools=[book_hotel],
            prompt="You are a hotel booking assistant.",
            post_model_hook=InterruptToolNode(
                book_hotel=HumanInterruptConfig(
                    allow_accept=True,
                    allow_edit=True,
                    allow_ignore=True,
                    allow_respond=True,
                )
            ),
            checkpointer=InMemorySaver(),
        )

        config = {"configurable": {"thread_id": 1}}

        response = agent.invoke(
            {"messages": [{"role": "user", "content": "please book a hotel at the hilton inn in boston."}]},
            config=config,
        )

        response = agent.invoke(Command(resume={"type": "accept"}), config=config)
        ```
    """

    def __init__(self, **interrupt_policy: HumanInterruptConfig):
        super().__init__(self._func, self._afunc)
        self.interrupt_policy = interrupt_policy

    def _interrupt(
        self,
        tool_call: ToolCall,
        interrupt_config: HumanInterruptConfig,
    ) -> Union[ToolCall, ToolMessage]:
        """Interrupt before a tool call and ask for human input."""
        call_id = tool_call["id"]
        tool_name = tool_call["name"]

        request = HumanInterrupt(
            action_request=ActionRequest(
                action=tool_name,
                args=tool_call["args"],
            ),
            config=interrupt_config,
            description=f"Please review tool call for `{tool_name}` before execution.",
        )
        response = interrupt([request])

        # resume provided by agent inbox as a list
        response = response[0] if isinstance(response, list) else response

        try:
            response_type = response.get("type")
        except AttributeError:
            raise TypeError(
                f"Unexpected resume value: {response}."
                f"Expected a dict with `'type'` key."
            )

        if response_type == "accept" and interrupt_config["allow_accept"]:
            return tool_call
        elif response_type == "edit" and interrupt_config["allow_edit"]:
            return ToolCall(
                args=cast(ActionRequest, response)["args"]["args"],
                name=tool_name,
                id=call_id,
                type="tool_call",
            )
        elif response_type == "response" and interrupt_config["allow_respond"]:
            return ToolMessage(
                content=cast(str, response["args"]),
                name=tool_name,
                tool_call_id=call_id,
                status="error",
            )
        elif response_type == "ignore" and interrupt_config["allow_ignore"]:
            return ToolMessage(
                content=f"User ignored the tool call for `{tool_name}` with id {call_id}",
                name=tool_name,
                tool_call_id=call_id,
                status="success",
            )

        allowed_types = [
            type_name
            for type_name, is_allowed in {
                "accept": interrupt_config["allow_accept"],
                "edit": interrupt_config["allow_edit"],
                "response": interrupt_config["allow_respond"],
                "ignore": interrupt_config["allow_ignore"],
            }.items()
            if is_allowed
        ]

        raise ValueError(
            f"Unexpected human response: {response}. "
            f"Expected one with `'type'` in {allowed_types} based on {tool_name}'s interrupt configuration."
        )

    def _func(self, input: dict[str, Any]) -> Command:
        ai_msg = input["messages"][-1]
        tool_calls: list[ToolCall] = deepcopy(ai_msg.tool_calls) or []
        tool_messages: list[ToolMessage] = []

        for idx, tool_call in enumerate(tool_calls):
            if interrupt_config := self.interrupt_policy.get(tool_call["name"]):
                interrupt_result = self._interrupt(
                    tool_call=tool_call, interrupt_config=interrupt_config
                )

                if isinstance(interrupt_result, ToolMessage):
                    tool_messages.append(interrupt_result)
                else:
                    tool_calls[idx] = interrupt_result

        updated_ai_msg = ai_msg.copy(update={"tool_calls": tool_calls})

        # conditional routing logic for post_model_hook will direct to the tools node
        # or agent node depending on if there are pending tool calls
        return {"messages": [updated_ai_msg, *tool_messages]}

    async def _afunc(self, input: dict[str, Any]) -> Command:
        return self._func(input)
