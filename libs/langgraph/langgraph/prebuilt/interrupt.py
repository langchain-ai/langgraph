from typing import (
    Literal,
    Optional,
    Union,
)

from typing_extensions import TypedDict


class HumanInterruptConfig(TypedDict):
    """Configuration that defines what actions are allowed for a human interrupt.

    This controls the available interaction options when the graph is paused for human input.

    Attributes:
        allow_ignore (bool): Whether the human can choose to ignore/skip the current step
        allow_respond (bool): Whether the human can provide a text response/feedback
        allow_edit (bool): Whether the human can edit the provided content/state
        allow_accept (bool): Whether the human can accept/approve the current state
    """

    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool


class ActionRequest(TypedDict):
    """Represents a request for human action within the graph execution.

    Contains the action type and any associated arguments needed for the action.

    Attributes:
        action (str): The type or name of action being requested (e.g., "Approve XYZ action")
        args (dict): Key-value pairs of arguments needed for the action
    """

    action: str
    args: dict


class HumanInterrupt(TypedDict):
    """Represents an interrupt triggered by the graph that requires human intervention.

    This is passed to the `interrupt` function when execution is paused for human input.

    Attributes:
        action_request (ActionRequest): The specific action being requested from the human
        config (HumanInterruptConfig): Configuration defining what actions are allowed
        description (Optional[str]): Optional detailed description of what input is needed
    """

    action_request: ActionRequest
    config: HumanInterruptConfig
    description: Optional[str]


class HumanResponse(TypedDict):
    """The response provided by a human to an interrupt, which is returned when graph execution resumes.

    Attributes:
        type (Literal['accept', 'ignore', 'response', 'edit']): The type of response:
            - "accept": Approves the current state without changes
            - "ignore": Skips/ignores the current step
            - "response": Provides text feedback or instructions
            - "edit": Modifies the current state/content
        args (Union[None, str, ActionRequest]): The response payload:
            - None: For ignore/accept actions
            - str: For text responses
            - ActionRequest: For edit actions with updated content
    """

    type: Literal["accept", "ignore", "response", "edit"]
    args: Union[None, str, ActionRequest]
