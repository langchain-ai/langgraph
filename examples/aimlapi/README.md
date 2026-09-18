# Using aimlapi.com with LangGraph

[aimlapi.com](https://aimlapi.com) is an OpenAI-compatible gateway that serves
353 chat models behind one key. This directory shows how to drive a LangGraph
graph with it, and documents the sharp edges that a green test suite will not
catch.

Everything below was verified against a live key on 2026-09-03 with
`langgraph 1.0.1`, `langgraph-prebuilt 1.0.1`, `langchain-core 0.3.86`,
`langchain-openai 0.3.35`, `openai 2.54.0` and `langchain-aimlapi 0.1.2`.

## LangGraph has no provider layer - and does not need one

LangGraph has no provider registry, no model adapters and no LLM client code.
The only place a model is resolved is
[`libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py`](../../libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py),
and it delegates:

```python
if isinstance(model, str):
    from langchain.chat_models import init_chat_model

    model = cast(BaseChatModel, init_chat_model(model))
```

So there are two paths, and only one of them works today:

| Path | Works? | Notes |
| --- | --- | --- |
| `create_react_agent(ChatAimlapi(...), tools)` - **model instance** | yes | any `BaseChatModel` instance skips `init_chat_model` entirely |
| `create_react_agent("aimlapi:openai/gpt-4o-mini", tools)` - **string syntax** | no | `ValueError: Unable to infer model provider for model='aimlapi:openai/gpt-4o-mini'` |

The string form needs a one-line entry in LangChain's provider table
(`langchain/chat_models/base.py`), which lives in `langchain-ai/langchain`, not
here. Nothing about aimlapi.com belongs in the LangGraph source tree.

## Quickstart

```bash
pip install langgraph langchain-aimlapi
export AIMLAPI_API_KEY=...
python two_turn_tool_loop.py                       # openai/gpt-4o-mini
python two_turn_tool_loop.py anthropic/claude-sonnet-4.6
```

```python
from langchain_aimlapi import ChatAimlapi
from langgraph.prebuilt import create_react_agent

llm = ChatAimlapi(model="openai/gpt-4o-mini")  # reads AIMLAPI_API_KEY
agent = create_react_agent(llm, [get_weather])
agent.invoke({"messages": [("user", "Weather in Paris?")]})
```

`ChatAimlapi` subclasses `BaseChatOpenAI`, so `bind_tools`, `with_structured_output`,
`.stream()` and `.astream()` all behave the way LangGraph expects. Requests go to
`POST https://api.aimlapi.com/v1/chat/completions`.

Files here:

- [`two_turn_tool_loop.py`](two_turn_tool_loop.py) - runnable two-turn agent loop.
- [`aimlapi_attribution.py`](aimlapi_attribution.py) - merge-safe attribution headers.
- [`test_aimlapi_attribution.py`](test_aimlapi_attribution.py) - offline tests
  (`pytest examples/aimlapi`), no key required.

## The turn that actually breaks: turn 2

aimlapi.com returns `400` for `null` on `tools`, `tool_choice`, `temperature`,
`top_p`, `seed`, `response_format`, `stream`, `stream_options`,
`parallel_tool_calls`, `max_tokens` and `max_completion_tokens` - and the exact
set differs per model. `tools: null` is the dangerous one: a client that clears
tools between turns by nulling the field succeeds on turn 1 and fails on turn 2
of every agent loop, which single-shot tests never reach.

This stack is clean. `langchain-openai` omits unset optionals rather than sending
`None`, and the OpenAI SDK drops `NOT_GIVEN`, so the body on the wire is only
`{messages, model, stream, tools}` on **both** turns. Verified end to end, two
HTTP requests per run, zero null fields:

| Model | turn 1 | turn 2 | `model` echoed back |
| --- | --- | --- | --- |
| `openai/gpt-4o-mini` | tool call | final answer | `gpt-4o-mini-2024-07-18` |
| `gpt-5.6-terra` (package default) | tool call | final answer | `gpt-5.6-terra` |
| `anthropic/claude-sonnet-4.6` | tool call | final answer | `anthropic/claude-sonnet-4.6` |
| `deepseek/deepseek-chat` | tool call | final answer | **`deepseek-v4-flash`** |

The same holds for a hand-built `StateGraph` with an explicit tool node, for
`.stream()`, and for `with_structured_output`.

If you write your own model node, keep the rule: **omit unset keys, never send
`None`.** `test_aimlapi_attribution.py` asserts the request body carries no null
fields on both turns.

> Note the `deepseek/deepseek-chat` row: aimlapi.com serves that id with a
> different model (`deepseek-v4-flash`). If your graph records or pins "the model
> that answered", read `response_metadata["model_name"]` rather than assuming it
> matches what you asked for.

## Embeddings: do not point `OpenAIEmbeddings` at aimlapi.com

`OpenAIEmbeddings` tokenises input client-side by default
(`check_embedding_ctx_length=True`) and sends `input` as an array of token ids.
aimlapi.com rejects that:

```text
400 PublicApiValidationException
details: [{"path": "input", "reason": "Invalid input", "code": "invalid_union"}]
```

Two working options:

```python
from langchain_aimlapi import AimlapiEmbeddings  # recommended: sends strings

AimlapiEmbeddings(model="text-embedding-3-small")

from langchain_openai import OpenAIEmbeddings  # or disable tokenisation

OpenAIEmbeddings(
    base_url="https://api.aimlapi.com/v1/",
    model="text-embedding-3-small",
    check_embedding_ctx_length=False,
)
```

Both return 1536-dimension vectors. This bites RAG graphs on the first real call
with a fully green test suite.

## Attribution headers

`langchain-aimlapi` already sends `HTTP-Referer`, `X-Title`, `X-AIMLAPI-Source`
and `X-AIMLAPI-Partner-ID` on every request. **Passing your own
`default_headers=` replaces them wholesale**, so an application that adds a
single tracing header silently drops all four. Use the helper in
[`aimlapi_attribution.py`](aimlapi_attribution.py), which merges instead of
assigning, builds a new dict per call, and refuses to attach the headers when
the base URL is not aimlapi.com:

```python
from aimlapi_attribution import aimlapi_default_headers

ChatAimlapi(
    model="openai/gpt-4o-mini",
    default_headers=aimlapi_default_headers({"X-App": "my-graph"}),
)
```

`AIMLAPI_PARTNER_ID` in that module now holds the partner id registered for
LangGraph. Note what this changes: the helper overrides the id that
`langchain-aimlapi` carries by default, so requests made through this example
are attributed to LangGraph rather than to LangChain. Clear the constant to go
back to the previous behaviour - the helper then leaves langchain's id in place
and overrides only `HTTP-Referer`/`X-Title`/`X-AIMLAPI-Source`.

## Other things worth knowing

- **Model ids.** Check `GET https://api.aimlapi.com/v1/models?include=all` and
  accept an id that appears either as an `id` *or* in another model's `aliases`.
  Chat models are `type == "openai/chat-completions"` (353 of 936 entries).
  Ids that look plausible and are dead: `gpt-5.5`, `anthropic/claude-sonnet-4-5`,
  `meta-llama/llama-4-maverick`. Correct forms: `openai/gpt-5-5`,
  `anthropic/claude-sonnet-4.5`.
- **`GET /v1/models` returns 200 for any key, including a bogus one.** Do not use
  it to validate a key in a graph's startup check.
- **`POST /v1/completions` does not exist** (404). `POST /v1/responses` exists but
  serves only 50 of the 353 chat models; `langchain-openai` uses
  `/v1/chat/completions` here, which is the right endpoint.
- **Token accounting.** On some models `completion_tokens` excludes reasoning
  tokens, and `max_tokens` does not reliably bound them. Do not meter spend from
  `usage_metadata["output_tokens"]` alone.
- **Capability flags are unreliable.** 15 chat models publish
  `capabilities: ["streaming"]` and nothing else while demonstrably doing tool
  calling; do not gate a graph's tool node on them.