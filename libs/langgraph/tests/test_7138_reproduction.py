from unittest.mock import Mock

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
import sys
import os

sys.path.append(os.path.abspath("libs/langgraph"))
sys.path.append(os.path.abspath("libs/prebuilt"))


def test_tool_node_error_metadata_inclusion():
    def error_tool(x: int):
        """Tool that always raises to test error metadata propagation."""
        raise ValueError("Simulated tool error")

    node = ToolNode(tools=[error_tool], handle_tool_errors=True)

    tool_call = {
        "name": "error_tool",
        "args": {"x": 1},
        "id": "call_123",
        "type": "tool_call",
        "metadata": {"user_id": "test_user", "session_id": "101"},
    }

    input_state = [tool_call]

    mock_runtime = Mock()
    mock_runtime.store = None
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda *args, **kwargs: None

    output = node.invoke(
        input_state,
        config={"configurable": {"__pregel_runtime": mock_runtime}},
    )

    message = output["messages"][-1]

    assert isinstance(message, ToolMessage)
    assert message.status == "error"

    assert "metadata" in message.additional_kwargs, (
        "Metadata missing in additional_kwargs"
    )
    assert message.additional_kwargs["metadata"] == tool_call["metadata"]
