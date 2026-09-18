"""LangGraph + aimlapi.com: a two-turn tool loop, end to end.

Turn 1 asks the model a question and it answers with a tool call.  Turn 2 feeds
the tool result back and the model answers in words.  Turn 2 is the turn that
breaks for OpenAI-compatible gateways that clear tools by sending
``"tools": null`` - aimlapi.com rejects a null ``tools`` field with a 400 on
several models - so it is the shape worth exercising, not a single-shot call.

Run it::

    pip install langgraph langchain-aimlapi
    export AIMLAPI_API_KEY=...        # your aimlapi.com key
    python two_turn_tool_loop.py [model-id]

The default model id is ``openai/gpt-4o-mini``, one of the strictest models on
aimlapi.com with respect to null request fields.
"""

from __future__ import annotations

import os
import sys

from langchain_aimlapi import ChatAimlapi
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

try:  # LangChain v1 moved the prebuilt agent out of langgraph
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - LangChain 0.3
    from langgraph.prebuilt import create_react_agent as create_agent

from aimlapi_attribution import aimlapi_default_headers

BASE_URL = "https://api.aimlapi.com/v1/"
DEFAULT_MODEL = "openai/gpt-4o-mini"


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"{city}: sunny, 21C, wind 8 km/h"


def build_agent(model: str = DEFAULT_MODEL):
    """A prebuilt ReAct agent backed by aimlapi.com.

    LangGraph resolves a ``"<provider>:<model>"`` string through LangChain's
    ``init_chat_model``, which has no aimlapi.com entry, so pass a
    ``ChatAimlapi`` **instance** instead - any ``BaseChatModel`` instance skips
    that lookup entirely.
    """
    llm = ChatAimlapi(
        model=model,
        base_url=BASE_URL,
        api_key=os.environ["AIMLAPI_API_KEY"],
        # merge, never assign: passing default_headers= without this helper
        # replaces the attribution that langchain-aimlapi ships
        default_headers=aimlapi_default_headers(base_url=BASE_URL),
    )
    return create_agent(llm, [get_weather])


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    agent = build_agent(model)
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "What is the weather in Paris right now? "
                    "Use the tool, then answer in one sentence."
                )
            ]
        }
    )
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
