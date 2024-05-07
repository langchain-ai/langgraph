from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool

from langgraph.prebuilt import create_react_executor


class FakeToolCallingModel(BaseChatModel):
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        messages_string = "-".join([m.content for m in messages])
        message = AIMessage(content=messages_string, id="0")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if len(tools) > 0:
            raise ValueError("Not supported yet!")
        return self


def test_no_modifier():
    model = FakeToolCallingModel()
    agent = create_react_executor(model, [])
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="hi?", id="0")]}
    assert response == expected_response


def test_system_message_modifier():
    model = FakeToolCallingModel()
    messages_modifier = SystemMessage(content="Foo")
    agent = create_react_executor(model, [], messages_modifier=messages_modifier)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Foo-hi?", id="0")]}
    assert response == expected_response


def test_system_message_string_modifier():
    model = FakeToolCallingModel()
    messages_modifier = "Foo"
    agent = create_react_executor(model, [], messages_modifier=messages_modifier)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Foo-hi?", id="0")]}
    assert response == expected_response


def test_callable_modifier():
    model = FakeToolCallingModel()

    def messages_modifier(messages):
        return [HumanMessage(content="Bar")]

    agent = create_react_executor(model, [], messages_modifier=messages_modifier)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Bar", id="0")]}
    assert response == expected_response


def test_runnable_modifier():
    model = FakeToolCallingModel()

    messages_modifier = RunnableLambda(lambda x: [HumanMessage(content="Baz")])

    agent = create_react_executor(model, [], messages_modifier=messages_modifier)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs})
    expected_response = {"messages": inputs + [AIMessage(content="Baz", id="0")]}
    assert response == expected_response
