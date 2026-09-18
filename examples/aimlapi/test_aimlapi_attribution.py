"""Offline tests for the aimlapi.com example. No network, no API key needed.

Run with::

    pytest examples/aimlapi
"""

from __future__ import annotations

import re

import pytest
from aimlapi_attribution import (
    AIMLAPI_PARTNER_ID,
    LANGGRAPH_ATTRIBUTION_HEADERS,
    aimlapi_default_headers,
)

PARTNER_ID_PATTERN = re.compile(r"^part_[A-Za-z0-9]{1,64}$")
SOURCE_PATTERN = re.compile(r"^(web|agent|mcp)/[a-z0-9-]{1,32}$")

AIMLAPI_BASE_URL = "https://api.aimlapi.com/v1/"


def test_partner_id_is_empty_or_well_formed() -> None:
    """A malformed partner id is never rejected by the API - it just earns
    nothing, silently. Empty is the honest value until one is registered."""
    assert AIMLAPI_PARTNER_ID == "" or PARTNER_ID_PATTERN.match(AIMLAPI_PARTNER_ID)


def test_partner_id_header_is_omitted_while_unregistered() -> None:
    headers = aimlapi_default_headers(base_url=AIMLAPI_BASE_URL)
    if not AIMLAPI_PARTNER_ID:
        # never send an empty header; fall back to whatever langchain-aimlapi ships
        assert headers.get("X-AIMLAPI-Partner-ID", "") != ""
    else:
        assert headers["X-AIMLAPI-Partner-ID"] == AIMLAPI_PARTNER_ID


def test_source_header_shape() -> None:
    """`<channel>/<client>`; an unrecognised channel makes the whole value drop."""
    assert SOURCE_PATTERN.match(LANGGRAPH_ATTRIBUTION_HEADERS["X-AIMLAPI-Source"])


def test_referer_and_title_name_the_host_project() -> None:
    """HTTP-Referer / X-Title identify the calling app, not the API vendor."""
    assert "langgraph" in LANGGRAPH_ATTRIBUTION_HEADERS["HTTP-Referer"]
    assert "aimlapi" not in LANGGRAPH_ATTRIBUTION_HEADERS["HTTP-Referer"]
    assert LANGGRAPH_ATTRIBUTION_HEADERS["X-Title"] == "LangGraph"


def test_caller_headers_are_merged_not_dropped() -> None:
    headers = aimlapi_default_headers(
        {"X-App": "my-graph", "X-Title": "My Graph"}, base_url=AIMLAPI_BASE_URL
    )
    assert headers["X-App"] == "my-graph"
    assert headers["X-Title"] == "My Graph"  # caller wins on a clash
    assert headers["X-AIMLAPI-Source"] == "agent/langgraph"  # attribution survives


def test_shared_constants_are_never_mutated() -> None:
    before = dict(LANGGRAPH_ATTRIBUTION_HEADERS)
    first = aimlapi_default_headers(base_url=AIMLAPI_BASE_URL)
    first["X-Title"] = "mutated"
    second = aimlapi_default_headers(base_url=AIMLAPI_BASE_URL)
    assert second["X-Title"] == "LangGraph"
    assert first is not second
    assert dict(LANGGRAPH_ATTRIBUTION_HEADERS) == before


def test_attribution_is_scoped_to_our_origin() -> None:
    """Headers must not ride a request to another provider, or to a proxy."""
    headers = aimlapi_default_headers(
        {"X-App": "my-graph"}, base_url="https://api.openai.com/v1/"
    )
    assert headers == {"X-App": "my-graph"}
    assert not any(k.lower().startswith("x-aimlapi") for k in headers)


@pytest.mark.parametrize("stage", ["first_turn", "second_turn"])
def test_request_body_never_carries_null_fields(stage: str) -> None:
    """aimlapi.com 400s on `null` for temperature, top_p, seed, tools,
    tool_choice, response_format, stream, stream_options, parallel_tool_calls,
    max_tokens and max_completion_tokens on its strictest models. `tools: null`
    is the one that matters here: it succeeds on turn 1 and fails on turn 2 of
    every agent loop. Unset options must be omitted, never sent as null."""
    langchain_aimlapi = pytest.importorskip("langchain_aimlapi")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Return the current weather for a city."""
        return "sunny"

    llm = langchain_aimlapi.ChatAimlapi(
        model="openai/gpt-4o-mini", api_key="not-a-real-key-no-request-is-made"
    )
    bound_kwargs = llm.bind_tools([get_weather]).kwargs

    messages = [HumanMessage("What is the weather in Paris?")]
    if stage == "second_turn":
        messages += [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="sunny", tool_call_id="call_1"),
        ]

    payload = llm._get_request_payload(messages, stop=None, **bound_kwargs)

    nulls = sorted(key for key, value in payload.items() if value is None)
    assert nulls == [], f"null request fields would 400 on aimlapi.com: {nulls}"
    assert payload["tools"], "tools must stay populated on every turn"
