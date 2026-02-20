import pytest
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

# Mock LLM that always requests the failing tool
class MockLLM(Runnable):
    def __init__(self):
        self.i = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, input, config: RunnableConfig = None, **kwargs):
        self.i += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fails_always",
                    "args": {},
                    "id": f"call_{self.i}",
                    "type": "tool_call",
                }
            ],
        )

@tool
def fails_always():
    """A tool that always returns an error message."""
    return "Error: Token exchange failed. Please try again."

@pytest.mark.asyncio
async def test_infinite_recursion_on_tool_failure():
    """Test that create_react_agent detects infinite tool call loops early."""
    tools = [fails_always]
    model = MockLLM()
    agent = create_react_agent(model, tools)

    inputs = {"messages": [HumanMessage(content="Swap 1 ETH for USDC")]}
    config = {"recursion_limit": 10}

    with pytest.raises(GraphRecursionError, match="Recursion limit hit for tool call"):
        async for _ in agent.astream(inputs, config=config, stream_mode="values"):
            pass
