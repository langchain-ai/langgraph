from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult

from langgraph.constants import TAG_HIDDEN, TAG_NOSTREAM
from langgraph.pregel._messages_v2 import StreamProtocolMessagesHandler
from langgraph.types import Command

META = {"langgraph_checkpoint_ns": "root:", "langgraph_node": "agent"}


def make_handler(subgraphs=True):
    events = []
    handler = StreamProtocolMessagesHandler(events.append, subgraphs)
    return handler, events


def test_streamed_text():
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[]
    )

    for token_text in ("Hello", " ", "world"):
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content=token_text, id=f"run-{run_id}")
        )
        handler.on_llm_new_token(token_text, chunk=chunk, run_id=run_id)

    final_msg = AIMessage(content="Hello world", id=f"run-{run_id}")
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    assert data_events[0]["event"] == "message-start"
    assert data_events[1]["event"] == "content-block-start"
    assert data_events[1]["index"] == 0

    deltas = [d for d in data_events if d["event"] == "content-block-delta"]
    assert len(deltas) == 3
    assert deltas[0]["content_block"]["text"] == "Hello"
    assert deltas[1]["content_block"]["text"] == " "
    assert deltas[2]["content_block"]["text"] == "world"

    finish_blocks = [d for d in data_events if d["event"] == "content-block-finish"]
    assert len(finish_blocks) == 1
    assert finish_blocks[0]["content_block"]["text"] == "Hello world"

    assert data_events[-1]["event"] == "message-finish"
    assert data_events[-1]["reason"] == "stop"


def test_tool_calls():
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[]
    )

    chunk1 = ChatGenerationChunk(
        message=AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "search", "args": '{"q', "id": "call_1", "index": 0}
            ],
            id=f"run-{run_id}",
        )
    )
    handler.on_llm_new_token("", chunk=chunk1, run_id=run_id)

    chunk2 = ChatGenerationChunk(
        message=AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": None, "args": 'uery":"hi"}', "id": None, "index": 0}
            ],
            id=f"run-{run_id}",
        )
    )
    handler.on_llm_new_token("", chunk=chunk2, run_id=run_id)

    final_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"query": "hi"}, "id": "call_1"}],
        id=f"run-{run_id}",
    )
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    finish_blocks = [d for d in data_events if d["event"] == "content-block-finish"]
    assert len(finish_blocks) == 1
    fb = finish_blocks[0]["content_block"]
    assert fb["type"] == "tool_call"
    assert fb["args"] == {"query": "hi"}
    assert fb["name"] == "search"
    assert fb["id"] == "call_1"


def test_invalid_tool_call_json():
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[]
    )

    chunk = ChatGenerationChunk(
        message=AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "search",
                    "args": "{not valid json",
                    "id": "call_2",
                    "index": 0,
                }
            ],
            id=f"run-{run_id}",
        )
    )
    handler.on_llm_new_token("", chunk=chunk, run_id=run_id)

    final_msg = AIMessage(content="", id=f"run-{run_id}")
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    finish_blocks = [d for d in data_events if d["event"] == "content-block-finish"]
    assert len(finish_blocks) == 1
    fb = finish_blocks[0]["content_block"]
    assert fb["type"] == "invalid_tool_call"
    assert "Failed to parse" in fb["error"]


def test_reasoning_blocks():
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[]
    )

    chunk = ChatGenerationChunk(
        message=AIMessageChunk(
            content=[{"type": "reasoning_content", "reasoning_content": "thinking..."}],
            id=f"run-{run_id}",
        )
    )
    handler.on_llm_new_token("", chunk=chunk, run_id=run_id)

    final_msg = AIMessage(content="", id=f"run-{run_id}")
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    block_starts = [d for d in data_events if d["event"] == "content-block-start"]
    assert len(block_starts) == 1
    assert block_starts[0]["content_block"]["type"] == "reasoning"

    deltas = [d for d in data_events if d["event"] == "content-block-delta"]
    assert len(deltas) == 1
    assert deltas[0]["content_block"]["reasoning"] == "thinking..."


def test_multiple_content_blocks():
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[]
    )

    chunk1 = ChatGenerationChunk(
        message=AIMessageChunk(content="hello", id=f"run-{run_id}")
    )
    handler.on_llm_new_token("hello", chunk=chunk1, run_id=run_id)

    chunk2 = ChatGenerationChunk(
        message=AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "lookup", "args": '{"x":1}', "id": "call_3", "index": 1}
            ],
            id=f"run-{run_id}",
        )
    )
    handler.on_llm_new_token("", chunk=chunk2, run_id=run_id)

    final_msg = AIMessage(
        content="hello",
        tool_calls=[{"name": "lookup", "args": {"x": 1}, "id": "call_3"}],
        id=f"run-{run_id}",
    )
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    finish_blocks = [d for d in data_events if d["event"] == "content-block-finish"]
    assert len(finish_blocks) == 2


def test_usage_metadata():
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[]
    )

    chunk = ChatGenerationChunk(
        message=AIMessageChunk(content="hi", id=f"run-{run_id}")
    )
    handler.on_llm_new_token("hi", chunk=chunk, run_id=run_id)

    final_msg = AIMessage(
        content="hi",
        id=f"run-{run_id}",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    finish_event = [d for d in data_events if d["event"] == "message-finish"][0]
    assert "usage" in finish_event
    assert finish_event["usage"]["input_tokens"] == 10


@pytest.mark.parametrize(
    "raw_reason,expected",
    [
        ("stop", "stop"),
        ("tool_calls", "tool_use"),
        ("length", "length"),
        ("content_filter", "content_filter"),
        ("end_turn", "stop"),
    ],
)
def test_finish_reason_normalization(raw_reason, expected):
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[]
    )
    chunk = ChatGenerationChunk(message=AIMessageChunk(content="x", id=f"run-{run_id}"))
    handler.on_llm_new_token("x", chunk=chunk, run_id=run_id)

    final_msg = AIMessage(
        content="x",
        id=f"run-{run_id}",
        response_metadata={"finish_reason": raw_reason},
    )
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    finish_event = [d for d in data_events if d["event"] == "message-finish"][0]
    assert finish_event["reason"] == expected


def test_tag_nostream():
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[TAG_NOSTREAM]
    )
    chunk = ChatGenerationChunk(
        message=AIMessageChunk(content="secret", id=f"run-{run_id}")
    )
    handler.on_llm_new_token("secret", chunk=chunk, run_id=run_id)

    final_msg = AIMessage(content="secret", id=f"run-{run_id}")
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id,
    )
    assert events == []


def test_tag_hidden_chain():
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chain_start(
        serialized={},
        inputs={},
        run_id=run_id,
        metadata=META,
        tags=[TAG_HIDDEN],
        name="agent",
    )
    handler.on_chain_end(
        {"messages": [AIMessage(content="hidden", id="msg-1")]},
        run_id=run_id,
    )
    assert events == []


def test_subgraph_filtering():
    handler, events = make_handler(subgraphs=False)
    run_id = uuid4()

    subgraph_meta = {
        "langgraph_checkpoint_ns": "root:|child:",
        "langgraph_node": "agent",
    }
    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=subgraph_meta, tags=[]
    )
    chunk = ChatGenerationChunk(
        message=AIMessageChunk(content="sub", id=f"run-{run_id}")
    )
    handler.on_llm_new_token("sub", chunk=chunk, run_id=run_id)

    final_msg = AIMessage(content="sub", id=f"run-{run_id}")
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id,
    )
    assert events == []


def test_chain_emits_messages():
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chain_start(
        serialized={}, inputs={}, run_id=run_id, metadata=META, tags=[], name="agent"
    )
    handler.on_chain_end(
        {"messages": [AIMessage(content="hello", id="msg-chain-1")]},
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    assert len(data_events) > 0
    assert data_events[0]["event"] == "message-start"
    assert data_events[-1]["event"] == "message-finish"


def test_llm_error_after_start():
    """on_llm_error should emit a message-error event for a started stream."""
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[]
    )

    chunk = ChatGenerationChunk(
        message=AIMessageChunk(content="partial", id=f"run-{run_id}")
    )
    handler.on_llm_new_token("partial", chunk=chunk, run_id=run_id)

    handler.on_llm_error(RuntimeError("connection lost"), run_id=run_id)

    data_events = [e[2] for e in events]
    assert data_events[0]["event"] == "message-start"
    error_events = [d for d in data_events if d["event"] == "error"]
    assert len(error_events) == 1
    assert "connection lost" in error_events[0]["message"]


def test_llm_error_before_start_no_emit():
    """on_llm_error before any tokens should not emit error events."""
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[]
    )

    # Error before any token — state.started is False
    handler.on_llm_error(RuntimeError("immediate fail"), run_id=run_id)

    data_events = [e[2] for e in events]
    error_events = [d for d in data_events if d.get("event") == "error"]
    assert len(error_events) == 0


def test_non_streamed_model():
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id, metadata=META, tags=[]
    )

    final_msg = AIMessage(
        content="full response",
        id=f"run-{run_id}",
        response_metadata={"finish_reason": "stop"},
    )
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    assert len(data_events) > 0
    assert data_events[0]["event"] == "message-start"

    deltas = [d for d in data_events if d["event"] == "content-block-delta"]
    assert len(deltas) == 1
    assert deltas[0]["content_block"]["text"] == "full response"

    assert data_events[-1]["event"] == "message-finish"
    assert data_events[-1]["reason"] == "stop"


def test_chain_emits_command_with_message():
    """on_chain_end should emit protocol events for messages inside a Command."""
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chain_start(
        serialized={}, inputs={}, run_id=run_id, metadata=META, tags=[], name="agent"
    )
    handler.on_chain_end(
        Command(update={"messages": [AIMessage(content="from command", id="cmd-1")]}),
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    assert len(data_events) > 0
    assert data_events[0]["event"] == "message-start"
    deltas = [d for d in data_events if d["event"] == "content-block-delta"]
    assert len(deltas) == 1
    assert deltas[0]["content_block"]["text"] == "from command"
    assert data_events[-1]["event"] == "message-finish"


def test_chain_emits_command_in_list():
    """on_chain_end should handle a list containing Command objects."""
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chain_start(
        serialized={}, inputs={}, run_id=run_id, metadata=META, tags=[], name="agent"
    )
    handler.on_chain_end(
        [Command(update={"messages": [AIMessage(content="listed", id="cmd-2")]})],
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    starts = [d for d in data_events if d["event"] == "message-start"]
    assert len(starts) == 1


def test_chain_deduplicates_seen_messages():
    """Messages already seen from LLM streaming should not be re-emitted by chain end."""
    handler, events = make_handler()
    run_id_llm = uuid4()
    run_id_chain = uuid4()
    msg_id = f"run-{run_id_llm}"

    # Simulate LLM streaming
    handler.on_chat_model_start(
        serialized={}, messages=[[]], run_id=run_id_llm, metadata=META, tags=[]
    )
    chunk = ChatGenerationChunk(message=AIMessageChunk(content="hello", id=msg_id))
    handler.on_llm_new_token("hello", chunk=chunk, run_id=run_id_llm)

    final_msg = AIMessage(content="hello", id=msg_id)
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=final_msg)]]),
        run_id=run_id_llm,
    )

    events_before = len(events)

    # Now chain end with the same message ID
    handler.on_chain_start(
        serialized={},
        inputs={},
        run_id=run_id_chain,
        metadata=META,
        tags=[],
        name="agent",
    )
    handler.on_chain_end(
        {"messages": [AIMessage(content="hello", id=msg_id)]},
        run_id=run_id_chain,
    )

    # No new events should have been emitted for the duplicate
    data_events_after = [e[2] for e in events[events_before:]]
    starts = [d for d in data_events_after if d.get("event") == "message-start"]
    assert len(starts) == 0


def test_chain_emits_human_message_role():
    """Non-AI messages from chain output should have the correct role."""
    handler, events = make_handler()
    run_id = uuid4()

    handler.on_chain_start(
        serialized={}, inputs={}, run_id=run_id, metadata=META, tags=[], name="agent"
    )
    handler.on_chain_end(
        {"messages": [HumanMessage(content="user msg", id="hmsg-1")]},
        run_id=run_id,
    )

    data_events = [e[2] for e in events]
    starts = [d for d in data_events if d["event"] == "message-start"]
    assert len(starts) == 1
    assert starts[0]["role"] == "human"
