import dataclasses
from typing import Optional, Any


#
# class HandoffTool(StructuredTool):
#     destination_node: str


# This handoff is too limited. We want to make sure that we can update
# graph states as part of the handoff or add a human-in-the-loop step w/ an interrupt.
# THe primitive needs to look more like a regular tool.
# We either push information into type annotation or we use a separate class.

@dataclasses.dataclass(frozen=True, kw_only=True)
class Handoff:
    node: str
    parent_update: Optional[Any] = None
    internal_update: Optional[Any] = None


# def handoff(
#     destination_node: str,
#     *,
#     # This API does not work right now -- inconsistent with how we handle tools
#     name: Optional[str] = None,
#     description: Optional[str] = None,
# ) -> HandoffTool:
#     """Create a tool that can hand off control to another node / agent."""
#
#     def func():
#         return f"Transferred to '{destination_node}'!", HandoffTool(
#             destination_node=destination_node
#         )
#
#     if description is None:
#         description = f"Transfer to '{destination_node}'. Do not ask any details."
#
#     if name is None:
#         name = destination_node
#
#     transfer_tool = HandoffTool.from_function(
#         func,
#         name=name,
#         description=description,
#         response_format="content_and_artifact",
#         args_schema=None,
#         infer_schema=True,
#         destination_node=destination_node,
#         # The return direct is very hacky
#         return_direct=True,
#     )
#     return transfer_tool
#
#
