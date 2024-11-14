from typing import Optional

from langchain_core.tools import StructuredTool
from langgraph.graph import GraphCommand


class HandoffTool(StructuredTool):
    destination_node: str


# This handoff is too limited. We want to make sure that we can update
# graph states as part of the handoff or add a human-in-the-loop step w/ an interrupt.
# THe primitive needs to look more like a regular tool.
# We either push information into type annotation or we use a separate class.


def handoff(
    destination_node: str,
    *,
    # This API does not work right now -- inconsistent with how we handle tools
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> HandoffTool:
    """Create a tool that can hand off control to another node / agent."""

    def func():
        return f"Transferred to '{destination_node}'!", GraphCommand(
            update={"active_agent": destination_node}
        )

    if description is None:
        description = f"Transfer to '{destination_node}'. Do not ask any details."

    if name is None:
        name = destination_node

    transfer_tool = HandoffTool.from_function(
        func,
        name=name,
        description=description,
        response_format="content_and_artifact",
        args_schema=None,
        infer_schema=True,
        destination_node=destination_node,
    )
    return transfer_tool


#
# class Handoff:
#     def __init__(
#             self,
#             *,
#             destination_node: str,
#             parent_state_update: Optional[Any] = None,
#             internal_state_update: Optional[Any] = None,
#     ) -> None:
#         self.destination_node = destination_node
#         self.parent_state_update = parent_state_update
#         self.internal_state_update = internal_state_update
