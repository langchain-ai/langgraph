"""Tests for subgraph persistence behavior (sync).

Covers three checkpointer settings for subgraph state:
- checkpointer=False: no persistence, even when parent has a checkpointer
- checkpointer=None (default): "stateless" — inherits parent checkpointer for
  interrupt support, but state resets each invocation. This is the common case
  when an agent is invoked from inside a tool used by another agent.
- checkpointer=True: "stateful" — state accumulates across invocations on the same thread id
"""

import hashlib
import hmac
import logging
import re
import time
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command, Interrupt, interrupt
from tests.any_str import AnyStr

# ---------------------------------------------------------------------------
# Audit logging setup
# ---------------------------------------------------------------------------
_audit_logger = logging.getLogger("ai_audit")
if not _audit_logger.handlers:
    _audit_handler = logging.StreamHandler()
    _audit_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    _audit_logger.addHandler(_audit_handler)
_audit_logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Inter-agent authentication
# ---------------------------------------------------------------------------
_AGENT_SECRET = b"test-inter-agent-secret-key"
_APPROVED_AGENTS = {"parent", "inner", "fruit", "veggie", "wrapper"}


def _sign_agent_token(caller: str, callee: str) -> str:
    """Return an HMAC token authenticating a caller→callee invocation."""
    msg = f"{caller}:{callee}:{int(time.time() // 60)}".encode()
    return hmac.new(_AGENT_SECRET, msg, hashlib.sha256).hexdigest()


def _verify_agent_token(caller: str, callee: str, token: str) -> None:
    """Raise ValueError if the token does not authenticate the call."""
    if caller not in _APPROVED_AGENTS or callee not in _APPROVED_AGENTS:
        raise ValueError(
            f"Inter-agent authentication failed: unknown agent '{caller}' or '{callee}'"
        )
    expected_now = hmac.new(
        _AGENT_SECRET,
        f"{caller}:{callee}:{int(time.time() // 60)}".encode(),
        hashlib.sha256,
    ).hexdigest()
    expected_prev = hmac.new(
        _AGENT_SECRET,
        f"{caller}:{callee}:{int(time.time() // 60) - 1}".encode(),
        hashlib.sha256,
    ).hexdigest()
    if not (
        hmac.compare_digest(token, expected_now)
        or hmac.compare_digest(token, expected_prev)
    ):
        raise ValueError(
            f"Inter-agent authentication failed: invalid token for {caller}→{callee}"
        )


# ---------------------------------------------------------------------------
# Input sanitization / validation
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS = re.compile(
    r"(base64\s*:|"
    r"(?:^|[\s;|&`])\s*(?:rm|curl|wget|bash|sh|python|exec|eval|os\.|subprocess)\b|"
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|"
    r"<script|javascript:|data:text/html)",
    re.IGNORECASE,
)


def _sanitize_input(content: str, field: str = "content") -> str:
    """Validate and sanitize a prompt string before passing to an agent."""
    if not isinstance(content, str):
        raise ValueError(f"Input field '{field}' must be a string.")
    if len(content) > 4096:
        raise ValueError(f"Input field '{field}' exceeds maximum allowed length.")
    if _DANGEROUS_PATTERNS.search(content):
        raise ValueError(
            f"Input field '{field}' contains potentially malicious content."
        )
    return content


def _make_human_message(content: str) -> HumanMessage:
    """Return a HumanMessage after sanitizing the content."""
    return HumanMessage(content=_sanitize_input(content))


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------
def _audit_invoke(
    graph_name: str,
    input_data: object,
    output_data: object,
    trace_id: str,
    *,
    event: str = "invoke",
) -> None:
    input_repr = repr(input_data)
    input_hash = hashlib.sha256(input_repr.encode()).hexdigest()
    output_repr = repr(output_data)
    _audit_logger.info(
        "event=%s graph=%s trace_id=%s timestamp=%s "
        "input_hash=%s output_preview=%.120s",
        event,
        graph_name,
        trace_id,
        time.time(),
        input_hash,
        output_repr,
    )


def _audit_interrupt(graph_name: str, trace_id: str, interrupt_value: object) -> None:
    _audit_logger.info(
        "event=interrupt graph=%s trace_id=%s timestamp=%s interrupt_value=%.120s",
        graph_name,
        trace_id,
        time.time(),
        repr(interrupt_value),
    )


def _audit_resume(graph_name: str, trace_id: str) -> None:
    _audit_logger.info(
        "event=resume graph=%s trace_id=%s timestamp=%s",
        graph_name,
        trace_id,
        time.time(),
    )


def _audit_state_accumulation(
    graph_name: str, trace_id: str, messages: list
) -> None:
    _audit_logger.info(
        "event=state_accumulation graph=%s trace_id=%s timestamp=%s messages_count=%d",
        graph_name,
        trace_id,
        time.time(),
        len(messages),
    )


# ---------------------------------------------------------------------------
# Authenticated invoke wrappers
# ---------------------------------------------------------------------------
def _authenticated_invoke(
    graph,
    graph_name: str,
    caller: str,
    input_data: object,
    config: dict | None = None,
    trace_id: str | None = None,
) -> object:
    """Authenticate, audit, and invoke a graph."""
    token = _sign_agent_token(caller, graph_name)
    _verify_agent_token(caller, graph_name, token)
    _trace_id = trace_id or str(uuid4())
    _audit_invoke(graph_name, input_data, None, _trace_id, event="invoke_start")
    if config is not None:
        result = graph.invoke(input_data, config)
    else:
        result = graph.invoke(input_data)
    _audit_invoke(graph_name, input_data, result, _trace_id, event="invoke_end")
    return result


class ParentState(TypedDict):
    result: str


# -- checkpointer=None (stateless) --


def test_stateless_interrupt_resume(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a subgraph compiled with checkpointer=None (the default) can
    still support interrupt/resume when invoked from inside a parent graph that
    has a checkpointer. This is the "stateless" pattern — the subgraph inherits
    the parent's checkpointer just enough to pause and resume, but does not
    retain any state across separate parent invocations. This pattern commonly
    appears when an agent is invoked from inside a tool used by another agent.
    """
    trace_id = str(uuid4())

    # Build a subgraph that interrupts before echoing.
    # Two nodes: "process" interrupts then echoes, "respond" returns "Done".
    def process(state: MessagesState) -> dict:
        interrupt("continue?")
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    def respond(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="Done")]}

    inner = (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_node("respond", respond)
        .add_edge(START, "process")
        .add_edge("process", "respond")
        .compile()
    )

    def call_inner(state: ParentState) -> dict:
        token = _sign_agent_token("parent", "inner")
        _verify_agent_token("parent", "inner", token)
        input_data = {"messages": [_make_human_message("apples")]}
        _audit_invoke("inner", input_data, None, trace_id, event="invoke_start")
        resp = inner.invoke(input_data)
        _audit_invoke("inner", input_data, resp, trace_id, event="invoke_end")
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    # First invoke hits the interrupt
    input1 = {"result": ""}
    _audit_invoke("parent", input1, None, trace_id, event="invoke_start")
    result = parent.invoke(input1, config)
    _audit_invoke("parent", input1, result, trace_id, event="invoke_end")
    if "__interrupt__" in result:
        _audit_interrupt("parent", trace_id, result["__interrupt__"])
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }

    # Resume completes the subgraph
    resume_cmd = Command(resume=True)
    _audit_resume("parent", trace_id)
    _audit_invoke("parent", resume_cmd, None, trace_id, event="invoke_start")
    result = parent.invoke(resume_cmd, config)
    _audit_invoke("parent", resume_cmd, result, trace_id, event="invoke_end")
    assert result == {"result": "Done"}


def test_stateless_state_resets(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a subgraph compiled with checkpointer=None (the default) does
    not retain any message history between separate parent invocations. Each time
    the parent graph invokes the subgraph, it starts with a clean slate. This
    confirms the "stateless" behavior: even though the parent has a checkpointer,
    the subgraph state is not persisted across calls.
    """
    trace_id = str(uuid4())

    # Build a simple echo subgraph: echoes "Processing: <input>"
    def echo(state: MessagesState) -> dict:
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    inner = (
        StateGraph(MessagesState)
        .add_node("echo", echo)
        .add_edge(START, "echo")
        .compile()
    )

    subgraph_messages: list[list[str]] = []
    call_count = 0

    def call_inner(state: ParentState) -> dict:
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        token = _sign_agent_token("parent", "inner")
        _verify_agent_token("parent", "inner", token)
        input_data = {
            "messages": [_make_human_message(f"tell me about {topic}")]
        }
        _audit_invoke("inner", input_data, None, trace_id, event="invoke_start")
        resp = inner.invoke(input_data)
        _audit_invoke("inner", input_data, resp, trace_id, event="invoke_end")
        subgraph_messages.append([m.text for m in resp["messages"]])
        _audit_state_accumulation("inner", trace_id, resp["messages"])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    input1 = {"result": ""}
    _audit_invoke("parent", input1, None, trace_id, event="invoke_start")
    result1 = parent.invoke(input1, config)
    _audit_invoke("parent", input1, result1, trace_id, event="invoke_end")
    assert result1 == {"result": "Processing: tell me about apples"}

    input2 = {"result": ""}
    _audit_invoke("parent", input2, None, trace_id, event="invoke_start")
    result2 = parent.invoke(input2, config)
    _audit_invoke("parent", input2, result2, trace_id, event="invoke_end")
    assert result2 == {"result": "Processing: tell me about bananas"}

    # Both invocations produce fresh history — no memory of prior call
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
    ]
    assert subgraph_messages[1] == [
        "tell me about bananas",
        "Processing: tell me about bananas",
    ]


def test_stateless_state_resets_with_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a subgraph compiled with checkpointer=None resets its state
    between parent invocations even when interrupt/resume is used. The subgraph
    is invoked twice from the parent, each time with an interrupt that must be
    resumed. After both invoke+resume cycles, each subgraph run should only
    contain its own messages — no bleed-over from the previous run.
    """
    trace_id = str(uuid4())

    # Build a subgraph that interrupts before echoing, then responds "Done"
    def process(state: MessagesState) -> dict:
        interrupt("continue?")
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    def respond(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="Done")]}

    inner = (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_node("respond", respond)
        .add_edge(START, "process")
        .add_edge("process", "respond")
        .compile()
    )

    subgraph_messages: list[list[str]] = []
    call_count = 0

    def call_inner(state: ParentState) -> dict:
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        token = _sign_agent_token("parent", "inner")
        _verify_agent_token("parent", "inner", token)
        input_data = {
            "messages": [_make_human_message(f"tell me about {topic}")]
        }
        _audit_invoke("inner", input_data, None, trace_id, event="invoke_start")
        resp = inner.invoke(input_data)
        _audit_invoke("inner", input_data, resp, trace_id, event="invoke_end")
        subgraph_messages.append([m.text for m in resp["messages"]])
        _audit_state_accumulation("inner", trace_id, resp["messages"])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    # First invoke+resume cycle
    input1 = {"result": ""}
    _audit_invoke("parent", input1, None, trace_id, event="invoke_start")
    result = parent.invoke(input1, config)
    _audit_invoke("parent", input1, result, trace_id, event="invoke_end")
    if "__interrupt__" in result:
        _audit_interrupt("parent", trace_id, result["__interrupt__"])
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }
    resume_cmd = Command(resume=True)
    _audit_resume("parent", trace_id)
    _audit_invoke("parent", resume_cmd, None, trace_id, event="invoke_start")
    result = parent.invoke(resume_cmd, config)
    _audit_invoke("parent", resume_cmd, result, trace_id, event="invoke_end")
    assert result == {"result": "Done"}

    # Second invoke+resume cycle
    input2 = {"result": ""}
    _audit_invoke("parent", input2, None, trace_id, event="invoke_start")
    result = parent.invoke(input2, config)
    _audit_invoke("parent", input2, result, trace_id, event="invoke_end")
    if "__interrupt__" in result:
        _audit_interrupt("parent", trace_id, result["__interrupt__"])
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }
    resume_cmd2 = Command(resume=True)
    _audit_resume("parent", trace_id)
    _audit_invoke("parent", resume_cmd2, None, trace_id, event="invoke_start")
    result = parent.invoke(resume_cmd2, config)
    _audit_invoke("parent", resume_cmd2, result, trace_id, event="invoke_end")
    assert result == {"result": "Done"}

    # Both invocations produce fresh history — no memory of prior call
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
    ]
    assert subgraph_messages[1] == [
        "tell me about bananas",
        "Processing: tell me about bananas",
        "Done",
    ]


# -- checkpointer=False --


def test_checkpointer_false_no_persistence(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a subgraph compiled with checkpointer=False gets no
    persistence at all, even when the parent graph has a checkpointer. Unlike
    the default (checkpointer=None) which inherits just enough from the parent
    to support interrupt/resume, checkpointer=False explicitly opts out of all
    checkpoint behavior. Each invocation starts completely fresh.
    """
    trace_id = str(uuid4())

    # Build a simple echo subgraph with checkpointer=False
    def echo(state: MessagesState) -> dict:
        return {
            "messages": [AIMessage(content=f"Processed: {state['messages'][-1].text}")]
        }

    inner = (
        StateGraph(MessagesState)
        .add_node("echo", echo)
        .add_edge(START, "echo")
        .compile(checkpointer=False)
    )

    subgraph_messages: list[list[str]] = []
    call_count = 0

    def call_inner(state: ParentState) -> dict:
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        token = _sign_agent_token("parent", "inner")
        _verify_agent_token("parent", "inner", token)
        input_data = {
            "messages": [_make_human_message(f"tell me about {topic}")]
        }
        _audit_invoke("inner", input_data, None, trace_id, event="invoke_start")
        resp = inner.invoke(input_data)
        _audit_invoke("inner", input_data, resp, trace_id, event="invoke_end")
        subgraph_messages.append([m.text for m in resp["messages"]])
        _audit_state_accumulation("inner", trace_id, resp["messages"])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    input1 = {"result": ""}
    _audit_invoke("parent", input1, None, trace_id, event="invoke_start")
    result1 = parent.invoke(input1, config)
    _audit_invoke("parent", input1, result1, trace_id, event="invoke_end")
    assert result1 == {"result": "Processed: tell me about apples"}

    input2 = {"result": ""}
    _audit_invoke("parent", input2, None, trace_id, event="invoke_start")
    result2 = parent.invoke(input2, config)
    _audit_invoke("parent", input2, result2, trace_id, event="invoke_end")
    assert result2 == {"result": "Processed: tell me about bananas"}

    # Both start fresh — no history from first call
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processed: tell me about apples",
    ]
    assert subgraph_messages[1] == [
        "tell me about bananas",
        "Processed: tell me about bananas",
    ]


# -- checkpointer=True (stateful) --


def test_stateful_state_accumulates(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a subgraph compiled with checkpointer=True ("stateful")
    retains its message history across separate parent invocations. To enable
    this, the subgraph is wrapped in an outer graph compiled with
    checkpointer=True — this wrapper gives the inner subgraph its own persistent
    checkpoint namespace. After two parent calls, the second subgraph invocation
    should see messages from both the first and second calls.
    """
    trace_id = str(uuid4())

    # Build a simple echo subgraph
    def echo(state: MessagesState) -> dict:
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    inner = (
        StateGraph(MessagesState)
        .add_node("echo", echo)
        .add_edge(START, "echo")
        .compile()
    )

    # Wrap the inner subgraph with checkpointer=True to enable stateful.
    # The wrapper graph gives the subgraph its own persistent checkpoint
    # namespace, keyed by the node name ("agent").
    wrapper = (
        StateGraph(MessagesState)
        .add_node("agent", inner)
        .add_edge(START, "agent")
        .compile(checkpointer=True)
    )

    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    def call_inner(state: ParentState) -> dict:
        topic = topics[len(subgraph_messages)]
        token = _sign_agent_token("parent", "wrapper")
        _verify_agent_token("parent", "wrapper", token)
        input_data = {
            "messages": [_make_human_message(f"tell me about {topic}")]
        }
        _audit_invoke("wrapper", input_data, None, trace_id, event="invoke_start")
        resp = wrapper.invoke(input_data)
        _audit_invoke("wrapper", input_data, resp, trace_id, event="invoke_end")
        subgraph_messages.append([m.text for m in resp["messages"]])
        _audit_state_accumulation("wrapper", trace_id, resp["messages"])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    input1 = {"result": ""}
    _audit_invoke("parent", input1, None, trace_id, event="invoke_start")
    result1 = parent.invoke(input1, config)
    _audit_invoke("parent", input1, result1, trace_id, event="invoke_end")
    assert result1 == {"result": "Processing: tell me about apples"}

    input2 = {"result": ""}
    _audit_invoke("parent", input2, None, trace_id, event="invoke_start")
    result2 = parent.invoke(input2, config)
    _audit_invoke("parent", input2, result2, trace_id, event="invoke_end")
    assert result2 == {"result": "Processing: tell me about bananas"}

    # First call: fresh history
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
    ]
    # Second call: retains messages from first call
    assert subgraph_messages[1] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "tell me about bananas",
        "Processing: tell me about bananas",
    ]


def test_stateful_state_accumulates_with_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a stateful subgraph (checkpointer=True) retains its
    message history across parent invocations even when interrupt/resume is
    involved. The subgraph interrupts before echoing, then responds "Done".
    After two invoke+resume cycles, the second run should contain the full
    accumulated history from both calls.
    """
    trace_id = str(uuid4())

    # Build a subgraph that interrupts before echoing, then responds "Done"
    def process(state: MessagesState) -> dict:
        interrupt("continue?")
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    def respond(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="Done")]}

    inner = (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_node("respond", respond)
        .add_edge(START, "process")
        .add_edge("process", "respond")
        .compile()
    )

    # Wrap with checkpointer=True for stateful
    wrapper = (
        StateGraph(MessagesState)
        .add_node("agent", inner)
        .add_edge(START, "agent")
        .compile(checkpointer=True)
    )

    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    def call_inner(state: ParentState) -> dict:
        topic = topics[len(subgraph_messages)]
        token = _sign_agent_token("parent", "wrapper")
        _verify_agent_token("parent", "wrapper", token)
        input_data = {
            "messages": [_make_human_message(f"tell me about {topic}")]
        }
        _audit_invoke("wrapper", input_data, None, trace_id, event="invoke_start")
        resp = wrapper.invoke(input_data)
        _audit_invoke("wrapper", input_data, resp, trace_id, event="invoke_end")
        subgraph_messages.append([m.text for m in resp["messages"]])
        _audit_state_accumulation("wrapper", trace_id, resp["messages"])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    # First invoke+resume cycle
    input1 = {"result": ""}
    _audit_invoke("parent", input1, None, trace_id, event="invoke_start")
    result = parent.invoke(input1, config)
    _audit_invoke("parent", input1, result, trace_id, event="invoke_end")
    if "__interrupt__" in result:
        _audit_interrupt("parent", trace_id, result["__interrupt__"])
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }
    resume_cmd = Command(resume=True)
    _audit_resume("parent", trace_id)
    _audit_invoke("parent", resume_cmd, None, trace_id, event="invoke_start")
    result = parent.invoke(resume_cmd, config)
    _audit_invoke("parent", resume_cmd, result, trace_id, event="invoke_end")
    assert result == {"result": "Done"}

    # Second invoke+resume cycle
    input2 = {"result": ""}
    _audit_invoke("parent", input2, None, trace_id, event="invoke_start")
    result = parent.invoke(input2, config)
    _audit_invoke("parent", input2, result, trace_id, event="invoke_end")
    if "__interrupt__" in result:
        _audit_interrupt("parent", trace_id, result["__interrupt__"])
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }
    resume_cmd2 = Command(resume=True)
    _audit_resume("parent", trace_id)
    _audit_invoke("parent", resume_cmd2, None, trace_id, event="invoke_start")
    result = parent.invoke(resume_cmd2, config)
    _audit_invoke("parent", resume_cmd2, result, trace_id, event="invoke_end")
    assert result == {"result": "Done"}

    # First call: fresh history
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
    ]
    # Second call: retains messages from first call
    assert subgraph_messages[1] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
        "tell me about bananas",
        "Processing: tell me about bananas",
        "Done",
    ]


def test_stateful_interrupt_resume(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a stateful subgraph (checkpointer=True) correctly
    supports interrupt/resume while also accumulating state. Each invoke+resume
    pair triggers the subgraph, and after the second pair completes we verify
    both the per-step invoke outputs and the accumulated message history. This
    exercises the full lifecycle: interrupt, resume, state accumulation.
    """
    trace_id = str(uuid4())

    # Build a subgraph that interrupts before echoing, then responds "Done"
    def process(state: MessagesState) -> dict:
        interrupt("continue?")
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    def respond(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="Done")]}

    inner = (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_node("respond", respond)
        .add_edge(START, "process")
        .add_edge("process", "respond")
        .compile()
    )

    # Wrap with checkpointer=True for stateful
    wrapper = (
        StateGraph(MessagesState)
        .add_node("agent", inner)
        .add_edge(START, "agent")
        .compile(checkpointer=True)
    )

    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    def call_inner(state: ParentState) -> dict:
        topic = topics[len(subgraph_messages)]
        token = _sign_agent_token("parent", "wrapper")
        _verify_agent_token("parent", "wrapper", token)
        input_data = {
            "messages": [_make_human_message(f"tell me about {topic}")]
        }
        _audit_invoke("wrapper", input_data, None, trace_id, event="invoke_start")
        resp = wrapper.invoke(input_data)
        _audit_invoke("wrapper", input_data, resp, trace_id, event="invoke_end")
        subgraph_messages.append([m.text for m in resp["messages"]])
        _audit_state_accumulation("wrapper", trace_id, resp["messages"])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    # First invocation: hits interrupt
    input1 = {"result": ""}
    _audit_invoke("parent", input1, None, trace_id, event="invoke_start")
    result = parent.invoke(input1, config)
    _audit_invoke("parent", input1, result, trace_id, event="invoke_end")
    if "__interrupt__" in result:
        _audit_interrupt("parent", trace_id, result["__interrupt__"])
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }

    # Resume: completes first call
    resume_cmd = Command(resume=True)
    _audit_resume("parent", trace_id)
    _audit_invoke("parent", resume_cmd, None, trace_id, event="invoke_start")
    result = parent.invoke(resume_cmd, config)
    _audit_invoke("parent", resume_cmd, result, trace_id, event="invoke_end")
    assert result == {"result": "Done"}
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
    ]

    # Second invocation: hits interrupt, state accumulated from first call
    input2 = {"result": ""}
    _audit_invoke("parent", input2, None, trace_id, event="invoke_start")
    result = parent.invoke(input2, config)
    _audit_invoke("parent", input2, result, trace_id, event="invoke_end")
    if "__interrupt__" in result:
        _audit_interrupt("parent", trace_id, result["__interrupt__"])
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }

    # Resume: completes second call with accumulated state
    resume_cmd2 = Command(resume=True)
    _audit_resume("parent", trace_id)
    _audit_invoke("parent", resume_cmd2, None, trace_id, event="invoke_start")
    result = parent.invoke(resume_cmd2, config)
    _audit_invoke("parent", resume_cmd2, result, trace_id, event="invoke_end")
    assert result == {"result": "Done"}
    assert subgraph_messages[1] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
        "tell me about bananas",
        "Processing: tell me about bananas",
        "Done",
    ]


def test_stateful_namespace_isolation(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that two different stateful subgraphs (checkpointer=True)
    maintain completely independent state when they use different wrapper node
    names. A "fruit_agent" and "veggie_agent" are each wrapped in their own
    stateful graph. After two parent invocations, each agent should only
    see its own accumulated history with no cross-contamination between them.
    """
    trace_id = str(uuid4())

    # Build two simple echo subgraphs with different prefixes
    def fruit_echo(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content=f"Fruit: {state['messages'][-1].text}")]}

    def veggie_echo(state: MessagesState) -> dict:
        return {
            "messages": [AIMessage(content=f"Veggie: {state['messages'][-1].text}")]
        }

    fruit_inner = (
        StateGraph(MessagesState)
        .add_node("echo", fruit_echo)
        .add_edge(START, "echo")
        .compile()
    )
    veggie_inner = (
        StateGraph(MessagesState)
        .add_node("echo", veggie_echo)
        .add_edge(START, "echo")
        .compile()
    )

    # Wrap each with checkpointer=True, using different node names to get
    # independent checkpoint namespaces
    fruit = (
        StateGraph(MessagesState)
        .add_node("fruit_agent", fruit_inner)
        .add_edge(START, "fruit_agent")
        .compile(checkpointer=True)
    )
    veggie = (
        StateGraph(MessagesState)
        .add_node("veggie_agent", veggie_inner)
        .add_edge(START, "veggie_agent")
        .compile(checkpointer=True)
    )

    fruit_msgs: list[list[str]] = []
    veggie_msgs: list[list[str]] = []
    call_count = 0

    def call_both(state: ParentState) -> dict:
        nonlocal call_count
        call_count += 1
        suffix = "round 1" if call_count == 1 else "round 2"

        fruit_token = _sign_agent_token("parent", "fruit")
        _verify_agent_token("parent", "fruit", fruit_token)
        fruit_input = {"messages": [_make_human_message(f"cherries {suffix}")]}
        _audit_invoke("fruit", fruit_input, None, trace_id, event="invoke_start")
        f = fruit.invoke(fruit_input)
        _audit_invoke("fruit", fruit_input, f, trace_id, event="invoke_end")
        _audit_state_accumulation("fruit", trace_id, f["messages"])

        veggie_token = _sign_agent_token("parent", "veggie")
        _verify_agent_token("parent", "veggie", veggie_token)
        veggie_input = {"messages": [_make_human_message(f"broccoli {suffix}")]}
        _audit_invoke("veggie", veggie_input, None, trace_id, event="invoke_start")
        v = veggie.invoke(veggie_input)
        _audit_invoke("veggie", veggie_input, v, trace_id, event="invoke_end")
        _audit_state_accumulation("veggie", trace_id, v["messages"])

        fruit_msgs.append([m.text for m in f["messages"]])
        veggie_msgs.append([m.text for m in v["messages"]])
        return {"result": f["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_both", call_both)
        .add_edge(START, "call_both")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    input1 = {"result": ""}
    _audit_invoke("parent", input1, None, trace_id, event="invoke_start")
    result1 = parent.invoke(input1, config)
    _audit_invoke("parent", input1, result1, trace_id, event="invoke_end")
    assert result1 == {"result": "Fruit: cherries round 1"}

    input2 = {"result": ""}
    _audit_invoke("parent", input2, None, trace_id, event="invoke_start")
    result2 = parent.invoke(input2, config)
    _audit_invoke("parent", input2, result2, trace_id, event="invoke_end")
    assert result2 == {"result": "Fruit: cherries round 2"}

    # First call: each agent sees only its own history
    assert fruit_msgs[0] == ["cherries round 1", "Fruit: cherries round 1"]
    assert veggie_msgs[0] == ["broccoli round 1", "Veggie: broccoli round 1"]

    # Second call: each accumulated independently — no cross-contamination
    assert fruit_msgs[1] == [
        "cherries round 1",
        "Fruit: cherries round 1",
        "cherries round 2",
        "Fruit: cherries round 2",
    ]
    assert veggie_msgs[1] == [
        "broccoli round 1",
        "Veggie: broccoli round 1",
        "broccoli round 2",
        "Veggie: broccoli round 2",
    ]