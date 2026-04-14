import pytest

from langgraph.stream.chat_model_stream import AsyncChatModelStream, ChatModelStream


def _text_delta(text: str) -> dict:
    return {"content_block": {"type": "text", "text": text}}


def _reasoning_delta(text: str) -> dict:
    return {"content_block": {"type": "reasoning", "reasoning": text}}


# ---------------------------------------------------------------------------
# Sync ChatModelStream tests
# ---------------------------------------------------------------------------


def test_sync_text_accumulates():
    stream = ChatModelStream()
    stream._push_content_block_delta(_text_delta("Hello"))
    stream._push_content_block_delta(_text_delta(", world"))
    stream._finish({"reason": "stop"})

    assert stream.text == "Hello, world"
    assert isinstance(stream.text, str)


def test_sync_reasoning_accumulates():
    stream = ChatModelStream()
    stream._push_content_block_delta(_reasoning_delta("step 1"))
    stream._push_content_block_delta(_reasoning_delta(" -> step 2"))
    stream._finish({"reason": "stop"})

    assert stream.reasoning == "step 1 -> step 2"
    assert isinstance(stream.reasoning, str)


def test_sync_usage():
    stream = ChatModelStream()
    usage = {"input_tokens": 10, "output_tokens": 5}
    stream._finish({"reason": "stop", "usage": usage})
    assert stream.usage == usage


def test_sync_mixed_blocks():
    stream = ChatModelStream()
    stream._push_content_block_delta(_text_delta("answer"))
    stream._push_content_block_delta(
        {"content_block": {"type": "tool_call", "name": "search"}}
    )
    stream._push_content_block_delta(_text_delta(" here"))
    stream._finish({"reason": "stop"})

    assert stream.text == "answer here"


def test_sync_tool_call_only_text_empty():
    stream = ChatModelStream()
    stream._push_content_block_delta(
        {"content_block": {"type": "tool_call", "name": "search"}}
    )
    stream._finish({"reason": "stop"})
    assert stream.text == ""


def test_sync_fail_marks_done():
    stream = ChatModelStream()
    assert not stream.done
    stream._fail(RuntimeError("err"))
    assert stream.done


def test_sync_namespace_and_node():
    stream = ChatModelStream(
        namespace=["agent:0", "tools:1"],
        node="chat_model",
        message_id="msg-123",
    )
    assert stream.namespace == ["agent:0", "tools:1"]
    assert stream.node == "chat_model"
    assert stream.message_id == "msg-123"


def test_sync_content_block_finish_authoritative():
    """content-block-finish with authoritative text overrides accumulated."""
    stream = ChatModelStream()
    stream._push_content_block_delta(_text_delta("partial"))
    stream._push_content_block_finish(
        {"content_block": {"type": "text", "text": "full text"}}
    )
    assert stream.text == "full text"


# ---------------------------------------------------------------------------
# Async ChatModelStream tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_text_iterable_yields_deltas():
    stream = AsyncChatModelStream()
    stream._push_content_block_delta(_text_delta("Hello"))
    stream._push_content_block_delta(_text_delta(", world"))
    stream._finish({"reason": "stop"})

    collected = []
    async for delta in stream.text:
        collected.append(delta)
    assert collected == ["Hello", ", world"]


@pytest.mark.anyio
async def test_async_text_awaitable_returns_full():
    stream = AsyncChatModelStream()
    stream._push_content_block_delta(_text_delta("Hello"))
    stream._push_content_block_delta(_text_delta(", world"))
    stream._finish({"reason": "stop"})

    result = await stream.text
    assert result == "Hello, world"


@pytest.mark.anyio
async def test_async_reasoning_dual_pattern():
    stream = AsyncChatModelStream()
    stream._push_content_block_delta(_reasoning_delta("step 1"))
    stream._push_content_block_delta(_reasoning_delta(" -> step 2"))
    stream._finish({"reason": "stop"})

    collected = []
    async for delta in stream.reasoning:
        collected.append(delta)
    assert collected == ["step 1", " -> step 2"]

    stream2 = AsyncChatModelStream()
    stream2._push_content_block_delta(_reasoning_delta("thinking"))
    stream2._finish({"reason": "stop"})
    full = await stream2.reasoning
    assert full == "thinking"


@pytest.mark.anyio
async def test_async_usage_resolves():
    stream = AsyncChatModelStream()
    usage = {"input_tokens": 10, "output_tokens": 5}
    stream._finish({"reason": "stop", "usage": usage})
    result = await stream.usage
    assert result == usage


@pytest.mark.anyio
async def test_async_mixed_blocks_text_only():
    stream = AsyncChatModelStream()
    stream._push_content_block_delta(_text_delta("answer"))
    stream._push_content_block_delta(
        {"content_block": {"type": "tool_call", "name": "search"}}
    )
    stream._push_content_block_delta(_text_delta(" here"))
    stream._finish({"reason": "stop"})

    collected = []
    async for delta in stream.text:
        collected.append(delta)
    assert collected == ["answer", " here"]


@pytest.mark.anyio
async def test_async_tool_call_only_text_empty():
    stream = AsyncChatModelStream()
    stream._push_content_block_delta(
        {"content_block": {"type": "tool_call", "name": "search"}}
    )
    stream._finish({"reason": "stop"})
    result = await stream.text
    assert result == ""


@pytest.mark.anyio
async def test_async_fail_raises_on_text_await():
    stream = AsyncChatModelStream()
    stream._push_content_block_delta(_text_delta("partial"))
    stream._fail(RuntimeError("model error"))

    with pytest.raises(RuntimeError, match="model error"):
        await stream.text


@pytest.mark.anyio
async def test_async_fail_raises_on_reasoning_await():
    stream = AsyncChatModelStream()
    stream._push_content_block_delta(_reasoning_delta("thinking"))
    stream._fail(RuntimeError("model error"))

    with pytest.raises(RuntimeError, match="model error"):
        await stream.reasoning


@pytest.mark.anyio
async def test_async_fail_raises_on_usage_await():
    stream = AsyncChatModelStream()
    stream._fail(RuntimeError("model error"))

    with pytest.raises(RuntimeError, match="model error"):
        await stream.usage


@pytest.mark.anyio
async def test_async_fail_raises_during_text_iteration():
    stream = AsyncChatModelStream()
    stream._push_content_block_delta(_text_delta("partial"))
    stream._fail(RuntimeError("model error"))

    collected = []
    with pytest.raises(RuntimeError, match="model error"):
        async for delta in stream.text:
            collected.append(delta)
    assert collected == ["partial"]


@pytest.mark.anyio
async def test_async_fail_marks_done():
    stream = AsyncChatModelStream()
    assert not stream.done
    stream._fail(RuntimeError("err"))
    assert stream.done


@pytest.mark.anyio
async def test_async_namespace_and_node():
    stream = AsyncChatModelStream(
        namespace=["agent:0", "tools:1"],
        node="chat_model",
        message_id="msg-123",
    )
    assert stream.namespace == ["agent:0", "tools:1"]
    assert stream.node == "chat_model"
    assert stream.message_id == "msg-123"


@pytest.mark.anyio
async def test_async_inherits_from_sync():
    """AsyncChatModelStream is a subclass of ChatModelStream."""
    stream = AsyncChatModelStream()
    assert isinstance(stream, ChatModelStream)
