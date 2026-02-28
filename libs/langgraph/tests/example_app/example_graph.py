from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages
from tests.fake_chat import FakeChatModel


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def search_api(query: str) -> str:
    """Searches the API for the query."""
    return f"result for {query}"


tools = [search_api]
tools_by_name = {t.name: t for t in tools}


def get_model():
    model = FakeChatModel(
        messages=[
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            AIMessage(
                id="ai2",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call234",
                        "name": "search_api",
                        "args": {"query": "another", "idx": 0},
                    },
                    {
                        "id": "tool_call567",
                        "name": "search_api",
                        "args": {"query": "a third one", "idx": 1},
                    },
                ],
            ),
            AIMessage(id="ai3", content="answer"),
        ]
    )
    return model


@task
def foo():
    return "foo"


@entrypoint()
async def app(state: AgentState) -> AgentState:
    model = get_model()
    max_steps = 100
    messages = state["messages"][:]
    await foo()  # Very useful call here ya know.
    for _ in range(max_steps):
        message = await model.ainvoke(messages)
        messages.append(message)
        if not message.tool_calls:
            break
        # Assume it's the search tool
        tool_results = await search_api.abatch(
            [t["args"]["query"] for t in message.tool_calls]
        )
        messages.extend(
            [
                ToolMessage(content=tool_res, tool_call_id=tc["id"])
                for tc, tool_res in zip(message.tool_calls, tool_results)
            ]
        )

    return entrypoint.final(value=messages[-1], save={"messages": messages})
