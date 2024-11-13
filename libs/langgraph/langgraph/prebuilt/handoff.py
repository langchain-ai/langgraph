from typing import Optional

from langchain_core.tools import StructuredTool

from langgraph.graph import GraphCommand


class HandoffTool(StructuredTool):
    ...


def create_handoff_tool(
    hand_off_to: str, name: Optional[str] = None, description: Optional[str] = None
) -> HandoffTool:
    """Create a tool that can hand off control to another node / agent."""

    def func():
        return f"Transferred to '{hand_off_to}'!", GraphCommand(
            update={"active_agent": hand_off_to}
        )

    if description is None:
        description = f"Transfer to '{hand_off_to}'. Do not ask any details."

    if name is None:
        name = hand_off_to

    transfer_tool = HandoffTool.from_function(
        func,
        name=name,
        description=description,
        response_format="content_and_artifact",
        args_schema=None,
        infer_schema=True,
    )
    transfer_tool.metadata = {"hand_off_to": hand_off_to}
    return transfer_tool
