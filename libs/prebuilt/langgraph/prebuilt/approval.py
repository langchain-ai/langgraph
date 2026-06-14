from __future__ import annotations

from typing import Any, Literal, TypedDict, Union

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langgraph._internal._runnable import RunnableCallable


class ApprovalRequested(BaseModel):
    """Event fired when an approval is requested."""
    prompt: str = Field(..., description="The prompt shown to the human.")
    data: Any = Field(None, description="Additional data associated with the approval.")
    timeout: int | None = Field(None, description="Optional timeout for the approval in seconds.")


class ApprovalOutcome(BaseModel):
    """The outcome of an approval request."""
    action: Literal["approve", "reject", "modify"] = Field(..., description="The action taken by the human.")
    data: Any | None = Field(None, description="Additional data provided by the human.")
    reason: str | None = Field(None, description="The reason for rejection.")


class ApprovalGranted(ApprovalOutcome):
    """Event fired when an approval is granted."""
    action: Literal["approve"] = "approve"


class ApprovalRejected(ApprovalOutcome):
    """Event fired when an approval is rejected."""
    action: Literal["reject"] = "reject"


class ApprovalModified(ApprovalOutcome):
    """Event fired when an approval is modified."""
    action: Literal["modify"] = "modify"


class ApprovalResponse(TypedDict):
    """The response from the human."""
    action: Literal["approve", "reject", "modify"]
    data: Any | None
    reason: str | None


class ApprovalNode(RunnableCallable):
    """A node that interrupts graph execution to wait for human approval.

    This node simplifies the common pattern of pausing a workflow for human
    intervention. It uses the `interrupt` function to halt execution and
    surfaces an `ApprovalRequested` event to the client.

    When resumed via a `Command(resume=...)`, the node processes the response
    and returns a state update containing the outcome.

    Attributes:
        prompt (str): The prompt to show to the human.
        state_key (str): The key in the state where the result will be stored.
            Defaults to "approval_result".
        timeout (int | None): Optional timeout in seconds to include in the request.
    """

    def __init__(
        self,
        prompt: str,
        *,
        state_key: str = "approval_result",
        timeout: int | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the ApprovalNode.

        Args:
            prompt: The message to display to the human.
            state_key: The state key to update with the approval outcome.
            timeout: Optional timeout for the approval.
            name: Optional name for the node.
            tags: Optional tags for the node.
            metadata: Optional metadata for the node.
        """
        super().__init__(
            self._call,
            name=name or "ApprovalNode",
            tags=tags,
            metadata=metadata,
            trace=True,
        )
        self.prompt = prompt
        self.state_key = state_key
        self.timeout = timeout

    def _call(
        self, state: Any, config: RunnableConfig, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute the node logic."""
        # Create the interrupt request
        request = ApprovalRequested(
            prompt=self.prompt, data=state, timeout=self.timeout
        )

        # This will raise GraphInterrupt on the first call
        # and return the resume value on subsequent calls.
        response: ApprovalResponse = interrupt(request.model_dump())

        if not isinstance(response, dict) or "action" not in response:
            raise ValueError(
                f"Invalid approval response. Expected a dict with an 'action' key, "
                f"got {type(response).__name__}: {response}"
            )

        action = response.get("action")
        if action == "approve":
            outcome = ApprovalGranted(data=response.get("data"))
        elif action == "reject":
            outcome = ApprovalRejected(reason=response.get("reason"))
        elif action == "modify":
            outcome = ApprovalModified(data=response.get("data"))
        else:
            raise ValueError(f"Invalid approval action: {action}")

        return {self.state_key: outcome.model_dump(exclude_none=True)}
