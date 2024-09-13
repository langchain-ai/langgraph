from typing import Any, Optional
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.pregel import Pregel


def react_agent(n_tools: int, checkpointer: BaseCheckpointSaver) -> Pregel:
    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_tools(self, functions: list):
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = self.responses[self.i].copy()
            if self.i < len(self.responses) - 1:
                self.i += 1
            else:
                self.i = 0
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])

    tool = StructuredTool.from_function(
        lambda query: f"result for query: {query}" * 10,
        name=str(uuid4()),
        description="",
    )

    model = FakeFuntionChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": str(uuid4()),
                        "name": tool.name,
                        "args": {"query": str(uuid4()) * 100},
                    }
                ],
                id=str(uuid4()),
            )
            for _ in range(n_tools)
        ]
        + [
            AIMessage(content="answer" * 100, id=str(uuid4())),
        ]
    )

    return create_react_agent(model, [tool], checkpointer=checkpointer)


if __name__ == "__main__":
    import asyncio

    import uvloop

    from langgraph.checkpoint.memory import MemorySaver

    graph = react_agent(100, checkpointer=MemorySaver())
    input = {"messages": [HumanMessage("hi?")]}
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 20000000000}

    async def run():
        len([c async for c in graph.astream(input, config=config)])

    uvloop.install()
    asyncio.run(run())
