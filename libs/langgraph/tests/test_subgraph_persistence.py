"""Tests for subgraph persistence behavior.

Covers two scoping modes for subgraph state:
- Tool invocation scope (no checkpointer on subgraph): state resets each call
- Session scope (checkpointer=True on subgraph): state accumulates across calls

These tests use deterministic node functions (no LLM calls) and validate
interrupt/resume, state reset vs accumulation, namespace isolation via
StateGraph wrapper, and checkpoint collision with parallel calls.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command, Interrupt, interrupt
from tests.any_str import AnyStr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inner_graph() -> StateGraph:
    """Build a simple inner subgraph that uses interrupt and echoes a response.

    Flow: process_input -> respond
    - process_input: appends a tool-result message after interrupt
    - respond: appends an AI response summarising the interaction
    """

    def process_input(state: MessagesState):
        interrupt("continue?")
        last_content = state["messages"][-1].content
        return {
            "messages": [
                AIMessage(content=f"Processing: {last_content}"),
            ]
        }

    def respond(state: MessagesState):
        return {
            "messages": [
                AIMessage(content="Done"),
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("process_input", process_input)
    builder.add_node("respond", respond)
    builder.add_edge(START, "process_input")
    builder.add_edge("process_input", "respond")
    return builder


# ---------------------------------------------------------------------------
# Test 1: Tool invocation scope — interrupt and resume
# ---------------------------------------------------------------------------


def test_tool_invocation_scope_interrupt_resume(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Subgraph without its own checkpointer supports interrupt/resume.

    Maps to tool_invocation_scope.py Case 1.
    """
    inner = _make_inner_graph().compile()  # no checkpointer

    class ParentState(TypedDict):
        result: str

    def call_inner(state: ParentState):
        response = inner.invoke(
            {"messages": [HumanMessage(content="tell me about apples")]}
        )
        return {"result": response["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # First invoke hits interrupt
    result = parent.invoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [
            Interrupt(value="continue?", id=AnyStr()),
        ],
    }

    # Resume completes the graph
    result = parent.invoke(Command(resume=True), config)
    assert result == {"result": "Done"}


# ---------------------------------------------------------------------------
# Test 2: checkpointer=False explicitly disables persistence
# ---------------------------------------------------------------------------


def test_checkpointer_false_no_persistence(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Subgraph with checkpointer=False never persists state, even when parent has one.

    Unlike omitting the checkpointer (which inherits from parent),
    checkpointer=False explicitly opts out of checkpointing.
    State should reset on every invocation.
    """

    def process(state: MessagesState):
        last_content = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"Processed: {last_content}")]}

    inner = (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_edge(START, "process")
        .compile(checkpointer=False)  # explicitly disabled
    )

    message_counts: list[int] = []
    call_count = 0

    class ParentState(TypedDict):
        result: str

    def call_inner(state: ParentState):
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        response = inner.invoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        message_counts.append(len(response["messages"]))
        return {"result": response["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Two invocations on the same thread
    parent.invoke({"result": ""}, config)
    parent.invoke({"result": ""}, config)

    # Both should have the same count — no accumulation despite parent having
    # a checkpointer, because checkpointer=False explicitly disables it
    assert len(message_counts) == 2
    assert message_counts[0] == message_counts[1], (
        f"checkpointer=False should prevent state accumulation, "
        f"got counts {message_counts}"
    )


# ---------------------------------------------------------------------------
# Test 3: Tool invocation scope — state resets between calls
# ---------------------------------------------------------------------------


def test_tool_invocation_scope_state_resets(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Subgraph without checkpointer resets state on each invocation.

    Maps to tool_invocation_scope.py Case 2. The inner subgraph is called
    twice on the same thread. Both calls should produce the same message
    count because the subgraph has no memory.
    """
    inner = _make_inner_graph().compile()  # no checkpointer

    call_count = 0
    message_counts: list[int] = []

    class ParentState(TypedDict):
        result: str

    def call_inner(state: ParentState):
        nonlocal call_count
        call_count += 1
        question = "apples" if call_count == 1 else "bananas"
        response = inner.invoke(
            {"messages": [HumanMessage(content=f"tell me about {question}")]}
        )
        message_counts.append(len(response["messages"]))
        return {"result": response["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # First call — hits interrupt
    parent.invoke({"result": ""}, config)
    parent.invoke(Command(resume=True), config)

    # Second call on same thread — hits interrupt again
    parent.invoke({"result": ""}, config)
    parent.invoke(Command(resume=True), config)

    # Both invocations should have produced the same message count
    # because the inner subgraph has no checkpointer (state resets)
    assert len(message_counts) == 2
    assert message_counts[0] == message_counts[1], (
        f"Expected equal message counts (state reset), got {message_counts}"
    )


# ---------------------------------------------------------------------------
# Test 3: Session scope — state accumulates across calls
# ---------------------------------------------------------------------------


def test_session_scope_state_accumulates(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Subgraph with checkpointer=True accumulates state across invocations.

    Maps to session_scope.py Case 2. The inner subgraph is wrapped in a
    StateGraph (for namespace isolation) and compiled with checkpointer=True.
    On the second call the subgraph should have more messages than the first.
    """

    def process_input(state: MessagesState):
        last_content = state["messages"][-1].content
        return {
            "messages": [
                AIMessage(content=f"Processing: {last_content}"),
            ]
        }

    def respond(state: MessagesState):
        return {
            "messages": [
                AIMessage(content="Done"),
            ]
        }

    inner_builder = StateGraph(MessagesState)
    inner_builder.add_node("process_input", process_input)
    inner_builder.add_node("respond", respond)
    inner_builder.add_edge(START, "process_input")
    inner_builder.add_edge("process_input", "respond")
    inner_graph = inner_builder.compile()

    # Wrap in StateGraph for namespace isolation (matches session_scope.py pattern)
    wrapper = (
        StateGraph(MessagesState)
        .add_node("inner_agent", inner_graph)
        .add_edge(START, "inner_agent")
        .compile(checkpointer=True)  # session scope
    )

    message_counts: list[int] = []
    call_count = 0

    class ParentState(TypedDict):
        result: str

    def call_inner(state: ParentState):
        nonlocal call_count
        call_count += 1
        question = "apples" if call_count == 1 else "bananas"
        response = wrapper.invoke(
            {"messages": [HumanMessage(content=f"tell me about {question}")]}
        )
        message_counts.append(len(response["messages"]))
        return {"result": response["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # First call
    parent.invoke({"result": ""}, config)
    first_count = message_counts[0]

    # Second call on same thread — state should accumulate
    parent.invoke({"result": ""}, config)
    second_count = message_counts[1]

    assert second_count > first_count, (
        f"Expected state to accumulate (second_count > first_count), "
        f"got first={first_count}, second={second_count}"
    )


# ---------------------------------------------------------------------------
# Test 4: Session scope — parallel different agents, namespace isolation
# ---------------------------------------------------------------------------


def test_session_scope_parallel_different_agents_namespace_isolation(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Two different session-scope subgraphs maintain independent state.

    Maps to session_scope.py Case 3 + sydney_test.py. Each subgraph is
    wrapped in its own StateGraph with a unique node name, giving each
    its own checkpoint namespace. State accumulates independently.
    """

    def fruit_node(state: MessagesState):
        last_content = state["messages"][-1].content
        return {
            "messages": [
                AIMessage(content=f"Fruit info: {last_content}"),
            ]
        }

    def veggie_node(state: MessagesState):
        last_content = state["messages"][-1].content
        return {
            "messages": [
                AIMessage(content=f"Veggie info: {last_content}"),
            ]
        }

    # Build two subgraphs with unique wrapper names for namespace isolation
    fruit_inner = (
        StateGraph(MessagesState)
        .add_node("fruit_node", fruit_node)
        .add_edge(START, "fruit_node")
        .compile()
    )
    fruit_wrapper = (
        StateGraph(MessagesState)
        .add_node("fruit_agent", fruit_inner)  # unique name
        .add_edge(START, "fruit_agent")
        .compile(checkpointer=True)
    )

    veggie_inner = (
        StateGraph(MessagesState)
        .add_node("veggie_node", veggie_node)
        .add_edge(START, "veggie_node")
        .compile()
    )
    veggie_wrapper = (
        StateGraph(MessagesState)
        .add_node("veggie_agent", veggie_inner)  # unique name
        .add_edge(START, "veggie_agent")
        .compile(checkpointer=True)
    )

    fruit_counts: list[int] = []
    veggie_counts: list[int] = []
    call_count = 0

    class ParentState(TypedDict):
        fruit_result: str
        veggie_result: str

    def call_both(state: ParentState):
        nonlocal call_count
        call_count += 1
        suffix = "round 1" if call_count == 1 else "round 2"

        fruit_resp = fruit_wrapper.invoke(
            {"messages": [HumanMessage(content=f"cherries {suffix}")]}
        )
        veggie_resp = veggie_wrapper.invoke(
            {"messages": [HumanMessage(content=f"broccoli {suffix}")]}
        )

        fruit_counts.append(len(fruit_resp["messages"]))
        veggie_counts.append(len(veggie_resp["messages"]))

        return {
            "fruit_result": fruit_resp["messages"][-1].content,
            "veggie_result": veggie_resp["messages"][-1].content,
        }

    parent = (
        StateGraph(ParentState)
        .add_node("call_both", call_both)
        .add_edge(START, "call_both")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # First call
    parent.invoke({"fruit_result": "", "veggie_result": ""}, config)
    fruit_first = fruit_counts[0]
    veggie_first = veggie_counts[0]

    # Second call — both should accumulate independently
    parent.invoke({"fruit_result": "", "veggie_result": ""}, config)
    fruit_second = fruit_counts[1]
    veggie_second = veggie_counts[1]

    # Both agents should accumulate state
    assert fruit_second > fruit_first, (
        f"Fruit agent should accumulate state: first={fruit_first}, second={fruit_second}"
    )
    assert veggie_second > veggie_first, (
        f"Veggie agent should accumulate state: first={veggie_first}, second={veggie_second}"
    )

    # Namespace isolation: fruit and veggie counts should match each other
    # (both start fresh and accumulate symmetrically)
    assert fruit_first == veggie_first
    assert fruit_second == veggie_second


# ---------------------------------------------------------------------------
# Test 5: Session scope — same agent called across invocations, shared checkpoint
# ---------------------------------------------------------------------------


def test_session_scope_parallel_same_agent_checkpoint_collision(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Same session-scope subgraph shares one checkpoint across all calls.

    Maps to session_scope.py Case 4. When the same checkpointer=True subgraph
    is called multiple times (whether sequentially or in parallel), all calls
    read/write the same checkpoint namespace. This means:
    - Sequential calls across invocations: state accumulates (expected)
    - Parallel calls within one invocation: both read the same initial state
      and write back, causing a merge that produces unexpected message counts

    This test verifies the shared-checkpoint behavior by calling the same
    session-scope subgraph once per parent invocation. The first call to
    the subgraph within each parent invocation records its message count.
    On the second parent invocation, the subgraph sees accumulated state
    from the first invocation — confirming a single shared checkpoint.
    """

    def echo_node(state: MessagesState):
        last_content = state["messages"][-1].content
        return {
            "messages": [
                AIMessage(content=f"Echo: {last_content}"),
            ]
        }

    inner = (
        StateGraph(MessagesState)
        .add_node("echo", echo_node)
        .add_edge(START, "echo")
        .compile()
    )
    # Same wrapper used for all calls — single shared checkpoint namespace
    shared_wrapper = (
        StateGraph(MessagesState)
        .add_node("shared_agent", inner)
        .add_edge(START, "shared_agent")
        .compile(checkpointer=True)
    )

    message_counts: list[int] = []
    call_count = 0

    class ParentState(TypedDict):
        result: str

    def call_shared_agent(state: ParentState):
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = shared_wrapper.invoke({"messages": [HumanMessage(content=topic)]})
        message_counts.append(len(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_shared_agent", call_shared_agent)
        .add_edge(START, "call_shared_agent")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # First invocation — subgraph starts fresh
    parent.invoke({"result": ""}, config)
    first_count = message_counts[0]

    # Second invocation — subgraph sees accumulated state (shared checkpoint)
    parent.invoke({"result": ""}, config)
    second_count = message_counts[1]

    # The same session-scope subgraph accumulates state across invocations
    # because all calls share a single checkpoint namespace. This is the
    # mechanism that causes unexpected state merging when parallel tool calls
    # target the same subgraph within one invocation.
    assert second_count > first_count, (
        f"Same session-scope subgraph should accumulate state across invocations "
        f"(shared checkpoint): first={first_count}, second={second_count}"
    )
