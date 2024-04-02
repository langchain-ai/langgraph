import json
from typing import List

from langchain_core.agents import AgentFinish
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from syrupy import SnapshotAssertion

from langgraph.prebuilt.agent_executor import create_agent_executor


def _make_agent_runnable(tools: List[BaseTool], response_messages: List[BaseMessage]) -> Runnable:
    """Make fake agent runnable."""
    from langchain.agents.format_scratchpad.openai_tools import (
        format_to_openai_tool_messages,
    )
    from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
    from langchain.chat_models.fake import FakeMessagesListChatModel

    chat_model = FakeMessagesListChatModel(responses=response_messages)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm_with_tools = chat_model.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            )
        )
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )


def test_agent_executor(snapshot: SnapshotAssertion) -> None:
    from langchain.agents import tool

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]
    responses = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "tool_call123",
                        "type": "function",
                        "function": {
                            "name": "search_api",
                            "arguments": json.dumps("query"),
                        },
                    }
                ]
            },
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "tool_call234",
                        "type": "function",
                        "function": {
                            "name": "search_api",
                            "arguments": json.dumps("another one"),
                        },
                    }
                ]
            },
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "tool_call345",
                        "type": "function",
                        "function": {
                            "name": "search_api",
                            "arguments": json.dumps("another one"),
                        },
                    }
                ]
            },
        ),
        AIMessage(content="answer"),
    ]
    agent_runnable = _make_agent_runnable(tools, responses)
    app = create_agent_executor(agent_runnable, tools)

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_ascii() == snapshot

    result = app.invoke(
        {
            "input": "what is the weather in sf?",
            "chat_history": [],
            "max_iterations": None,
            "iteration_count": 0,
            # "max_execution_time": None,
            # "start_time": time.time()
        }
    )
    assert result["agent_outcome"] == AgentFinish(return_values={'output': 'answer'}, log='answer')
    result = app.invoke(
        {
            "input": "what is the weather in sf?",
            "chat_history": [],
            "max_iterations": 2,
            "iteration_count": 0,
            # "max_execution_time": None,
            # "start_time": time.time()
        }
    )
    assert result["agent_outcome"] == AgentFinish(return_values={'output': 'Agent stopped due to max iterations.'}, log='Agent stopped due to max iterations.')
