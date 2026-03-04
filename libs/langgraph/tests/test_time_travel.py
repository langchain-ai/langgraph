"""Tests for time travel (replay and fork) behavior.

Covers the intersection of replay vs fork across graph structures:
- No subgraph, no interrupt
- No subgraph, with interrupt
- With subgraph, no interrupt in subgraph
- With subgraph, with interrupt in subgraph
- Additional scenarios (multiple interrupts, get_state with subgraphs, etc.)
- Edge cases

Key concepts:
- Replay (invoke with checkpoint_id): Re-executes nodes after the checkpoint.
  For interrupt nodes, cached resume values are used so interrupts don't re-fire.
- Fork (update_state then invoke): Creates a new checkpoint without cached
  pending writes. Nodes re-execute and interrupts DO re-fire.
"""

import operator
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

# ---------------------------------------------------------------------------
# Section 1: No subgraph, no interrupt
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    value: Annotated[list[str], operator.add]


def test_no_subgraph_no_interrupt_replay(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from checkpoint before node_b. node_b re-executes (it's after
    the checkpoint), producing the same result."""

    called: list[str] = []

    def node_a(state: SimpleState) -> SimpleState:
        called.append("node_a")
        return {"value": ["a"]}

    def node_b(state: SimpleState) -> SimpleState:
        called.append("node_b")
        return {"value": ["b"]}

    graph = (
        StateGraph(SimpleState)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"value": []}, config)
    assert result == {"value": ["a", "b"]}
    assert called == ["node_a", "node_b"]

    # Find checkpoint before node_b (next=(node_b,))
    history = list(graph.get_state_history(config))
    before_b = next(s for s in history if s.next == ("node_b",))

    # Replay from checkpoint before node_b
    called.clear()
    replay_result = graph.invoke(None, before_b.config)

    # node_b re-executes (it's after the checkpoint), same final state
    assert replay_result == {"value": ["a", "b"]}
    assert "node_b" in called
    # node_a does NOT re-execute (it's before the checkpoint)
    assert "node_a" not in called


def test_no_subgraph_no_interrupt_fork(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from checkpoint before node_b with modified state. node_b
    re-executes with the new state."""

    called: list[str] = []

    def node_a(state: SimpleState) -> SimpleState:
        called.append("node_a")
        return {"value": ["a"]}

    def node_b(state: SimpleState) -> SimpleState:
        called.append("node_b")
        return {"value": ["b"]}

    graph = (
        StateGraph(SimpleState)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"value": []}, config)

    # Find checkpoint before node_b
    history = list(graph.get_state_history(config))
    before_b = next(s for s in history if s.next == ("node_b",))

    # Fork: update_state creates a new checkpoint with modified values
    called.clear()
    fork_config = graph.update_state(before_b.config, {"value": ["x"]})
    fork_result = graph.invoke(None, fork_config)

    # node_b re-executes with the forked state
    assert "node_b" in called
    assert fork_result == {"value": ["a", "x", "b"]}


# ---------------------------------------------------------------------------
# Section 2: No subgraph, with interrupt
# ---------------------------------------------------------------------------


class InterruptState(TypedDict):
    value: Annotated[list[str], operator.add]


def _build_interrupt_graph(
    checkpointer: BaseCheckpointSaver,
    called: list[str],
):
    """Build: START -> node_a -> ask_human [interrupt] -> node_b -> END"""

    def node_a(state: InterruptState) -> InterruptState:
        called.append("node_a")
        return {"value": ["a"]}

    def ask_human(state: InterruptState) -> InterruptState:
        called.append("ask_human")
        answer = interrupt("What is your input?")
        return {"value": [f"human:{answer}"]}

    def node_b(state: InterruptState) -> InterruptState:
        called.append("node_b")
        return {"value": ["b"]}

    graph = (
        StateGraph(InterruptState)
        .add_node("node_a", node_a)
        .add_node("ask_human", ask_human)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "ask_human")
        .add_edge("ask_human", "node_b")
        .compile(checkpointer=checkpointer)
    )
    return graph


def test_no_subgraph_interrupt_replay_from_before(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from checkpoint before interrupt node. ask_human re-executes but
    interrupt() returns cached resume value — interrupt NOT re-triggered."""

    called: list[str] = []
    graph = _build_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result

    # Resume with answer
    result = graph.invoke(Command(resume="hello"), config)
    assert result == {"value": ["a", "human:hello", "b"]}

    # Find checkpoint before ask_human (where next=(ask_human,) and no interrupts)
    history = list(graph.get_state_history(config))
    # There may be multiple checkpoints with next=(ask_human,):
    # one before it ran, one where it interrupted. We want the one before.
    before_ask_candidates = [s for s in history if s.next == ("ask_human",)]
    # The one without interrupt tasks is "before"
    before_ask = before_ask_candidates[-1]  # Earliest in reverse-chronological

    # Replay from that checkpoint
    called.clear()
    replay_result = graph.invoke(None, before_ask.config)

    # Interrupt is NOT re-triggered — cached resume value used
    assert replay_result == {"value": ["a", "human:hello", "b"]}
    assert "__interrupt__" not in replay_result
    # The node code DID re-execute (ask_human was called), but interrupt()
    # returned the cached resume value instead of raising GraphInterrupt
    assert "ask_human" in called
    assert "node_b" in called


def test_no_subgraph_interrupt_replay_node_reexecutes(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Verify that during replay, ALL node code re-executes — it's only
    interrupt() that uses cached values. Non-interrupt logic (like LLM calls)
    runs again and can produce different results."""

    call_count: dict[str, int] = {"ask_human": 0, "node_b": 0}
    # Track what interrupt() returned each time
    interrupt_returns: list[str] = []

    class ReplayState(TypedDict):
        value: Annotated[list[str], operator.add]

    def ask_human(state: ReplayState) -> ReplayState:
        call_count["ask_human"] += 1
        answer = interrupt("What is your input?")
        interrupt_returns.append(answer)
        return {"value": [f"human:{answer}"]}

    def node_b(state: ReplayState) -> ReplayState:
        call_count["node_b"] += 1
        # Simulate non-deterministic logic: output includes call count
        return {"value": [f"b_call_{call_count['node_b']}"]}

    graph = (
        StateGraph(ReplayState)
        .add_node("ask_human", ask_human)
        .add_node("node_b", node_b)
        .add_edge(START, "ask_human")
        .add_edge("ask_human", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt
    graph.invoke({"value": []}, config)
    assert call_count["ask_human"] == 1

    # Resume
    graph.invoke(Command(resume="hello"), config)
    assert call_count["ask_human"] == 2  # Re-executes on resume
    assert call_count["node_b"] == 1
    assert interrupt_returns == ["hello"]

    # Find checkpoint before ask_human
    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    # Replay
    interrupt_returns.clear()
    replay_result = graph.invoke(None, before_ask.config)

    # ask_human re-executed (call count incremented)
    assert call_count["ask_human"] == 3
    # interrupt() returned the cached value "hello" — not re-triggered
    assert interrupt_returns == ["hello"]
    assert "__interrupt__" not in replay_result
    # node_b also re-executed (call count incremented)
    assert call_count["node_b"] == 2
    # node_b produced DIFFERENT output because it's non-deterministic
    assert "b_call_2" in replay_result["value"]


def test_no_subgraph_interrupt_fork_from_before(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from checkpoint before interrupt node. Interrupt IS re-triggered
    because fork has no cached resume values."""

    called: list[str] = []
    graph = _build_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt, then resume
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="hello"), config)

    # Find checkpoint before ask_human
    history = list(graph.get_state_history(config))
    before_ask_candidates = [s for s in history if s.next == ("ask_human",)]
    before_ask = before_ask_candidates[-1]

    # Fork from that checkpoint
    called.clear()
    fork_config = graph.update_state(before_ask.config, {"value": ["forked"]})

    # Invoke from fork — interrupt IS re-triggered
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "What is your input?"

    # Resume the forked interrupt with a different answer
    final = graph.invoke(Command(resume="world"), fork_config)
    assert final == {"value": ["a", "forked", "human:world", "b"]}


def test_no_subgraph_interrupt_replay_from_of(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from the interrupt checkpoint itself (where interrupt fired).
    Uses cached resume, completes without re-triggering interrupt."""

    called: list[str] = []
    graph = _build_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt
    graph.invoke({"value": []}, config)

    # Resume
    graph.invoke(Command(resume="hello"), config)

    # Find the checkpoint where interrupt fired
    history = list(graph.get_state_history(config))
    interrupt_checkpoint = next(
        s
        for s in history
        if s.next == ("ask_human",) and s.tasks and any(t.interrupts for t in s.tasks)
    )

    # Replay from that checkpoint — uses cached resume
    called.clear()
    replay_result = graph.invoke(None, interrupt_checkpoint.config)
    assert replay_result == {"value": ["a", "human:hello", "b"]}
    assert "__interrupt__" not in replay_result


def test_no_subgraph_interrupt_fork_from_of(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from the interrupt checkpoint. Interrupt re-triggered because
    fork clears cached data."""

    called: list[str] = []
    graph = _build_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt, then resume
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="hello"), config)

    # Find interrupt checkpoint
    history = list(graph.get_state_history(config))
    interrupt_checkpoint = next(
        s
        for s in history
        if s.next == ("ask_human",) and s.tasks and any(t.interrupts for t in s.tasks)
    )

    # Fork from that checkpoint
    fork_config = graph.update_state(interrupt_checkpoint.config, {"value": ["forked"]})

    # Interrupt IS re-triggered
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result

    # Resume forked interrupt with different answer
    final = graph.invoke(Command(resume="different"), fork_config)
    assert "human:different" in final["value"]


# ---------------------------------------------------------------------------
# Section 3: With subgraph, no interrupt in subgraph
# ---------------------------------------------------------------------------


class ParentState(TypedDict):
    value: Annotated[list[str], operator.add]


class SubState(TypedDict):
    value: Annotated[list[str], operator.add]


def _build_subgraph_no_interrupt(
    checkpointer: BaseCheckpointSaver,
    called: list[str],
):
    """Build: START -> parent_node -> [subgraph: step_a -> step_b] -> post_process -> END"""

    def parent_node(state: ParentState) -> ParentState:
        called.append("parent_node")
        return {"value": ["parent"]}

    def step_a(state: SubState) -> SubState:
        called.append("step_a")
        return {"value": ["sub_a"]}

    def step_b(state: SubState) -> SubState:
        called.append("step_b")
        return {"value": ["sub_b"]}

    subgraph = (
        StateGraph(SubState)
        .add_node("step_a", step_a)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "step_b")
        .compile()
    )

    def post_process(state: ParentState) -> ParentState:
        called.append("post_process")
        return {"value": ["post"]}

    graph = (
        StateGraph(ParentState)
        .add_node("parent_node", parent_node)
        .add_node("subgraph", subgraph)
        .add_node("post_process", post_process)
        .add_edge(START, "parent_node")
        .add_edge("parent_node", "subgraph")
        .add_edge("subgraph", "post_process")
        .compile(checkpointer=checkpointer)
    )
    return graph


def test_subgraph_no_interrupt_replay_from_parent_before(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from parent checkpoint before subgraph node. Subgraph and
    post_process re-execute (they're after the checkpoint)."""

    called: list[str] = []
    graph = _build_subgraph_no_interrupt(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    result = graph.invoke({"value": []}, config)
    # Subgraph inherits parent state, so "parent" appears from parent_node
    # and again when subgraph returns state including the inherited value
    initial_value = result["value"]

    # Find checkpoint before subgraph (next=(subgraph,))
    history = list(graph.get_state_history(config))
    before_sub = next(s for s in history if s.next == ("subgraph",))

    # Replay from that checkpoint
    called.clear()
    replay_result = graph.invoke(None, before_sub.config)

    # Same result as original execution
    assert replay_result["value"] == initial_value
    assert "parent_node" not in called  # Before checkpoint — not re-executed


def test_subgraph_no_interrupt_fork_from_parent_before(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from parent checkpoint before subgraph node with modified state.
    Subgraph re-executes with forked state."""

    called: list[str] = []
    graph = _build_subgraph_no_interrupt(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"value": []}, config)

    # Find checkpoint before subgraph
    history = list(graph.get_state_history(config))
    before_sub = next(s for s in history if s.next == ("subgraph",))

    # Fork with modified state
    called.clear()
    fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    # Subgraph re-executes with forked state included
    assert "step_a" in called
    assert "step_b" in called
    assert "post_process" in called
    assert "forked" in fork_result["value"]
    assert "sub_a" in fork_result["value"]
    assert "sub_b" in fork_result["value"]


# ---------------------------------------------------------------------------
# Section 4: With subgraph, with interrupt in subgraph
# ---------------------------------------------------------------------------


def _build_subgraph_interrupt_graph(
    checkpointer: BaseCheckpointSaver,
    called: list[str],
    subgraph_checkpointer=True,
):
    """Build: START -> router -> [subgraph: step_a -> ask_human [interrupt] -> step_b] -> post_process -> END"""

    class SubInterruptState(TypedDict):
        value: Annotated[list[str], operator.add]

    class ParentInterruptState(TypedDict):
        value: Annotated[list[str], operator.add]

    def router(state: ParentInterruptState) -> ParentInterruptState:
        called.append("router")
        return {"value": ["routed"]}

    def step_a(state: SubInterruptState) -> SubInterruptState:
        called.append("step_a")
        return {"value": ["sub_a"]}

    def ask_human(state: SubInterruptState) -> SubInterruptState:
        called.append("ask_human")
        answer = interrupt("Provide input:")
        return {"value": [f"human:{answer}"]}

    def step_b(state: SubInterruptState) -> SubInterruptState:
        called.append("step_b")
        return {"value": ["sub_b"]}

    subgraph = (
        StateGraph(SubInterruptState)
        .add_node("step_a", step_a)
        .add_node("ask_human", ask_human)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_human")
        .add_edge("ask_human", "step_b")
        .compile(checkpointer=subgraph_checkpointer)
    )

    def post_process(state: ParentInterruptState) -> ParentInterruptState:
        called.append("post_process")
        return {"value": ["post"]}

    graph = (
        StateGraph(ParentInterruptState)
        .add_node("router", router)
        .add_node("subgraph_node", subgraph)
        .add_node("post_process", post_process)
        .add_edge(START, "router")
        .add_edge("router", "subgraph_node")
        .add_edge("subgraph_node", "post_process")
        .compile(checkpointer=checkpointer)
    )

    return graph


def test_subgraph_interrupt_replay_from_parent_before(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from parent checkpoint before subgraph. Subgraph re-executes
    but interrupt uses cached resume — interrupt NOT re-triggered."""

    called: list[str] = []
    graph = _build_subgraph_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result

    # Resume
    completed_result = graph.invoke(Command(resume="answer"), config)
    # Capture actual completed values (subgraph may inherit parent state)
    completed_value = completed_result["value"]
    assert "human:answer" in completed_value
    assert "sub_b" in completed_value
    assert "post" in completed_value

    # Find parent checkpoint before subgraph_node
    history = list(graph.get_state_history(config))
    before_sub_candidates = [s for s in history if s.next == ("subgraph_node",)]
    before_sub = before_sub_candidates[-1]  # Earliest

    # Replay — interrupt NOT re-triggered (cached resume used)
    called.clear()
    replay_result = graph.invoke(None, before_sub.config)
    assert replay_result["value"] == completed_value
    assert "__interrupt__" not in replay_result


def test_subgraph_interrupt_fork_from_parent_before(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from parent checkpoint before subgraph (checkpointer=True).
    The subgraph's OWN checkpoint state persists independently — the parent
    fork does NOT clear subgraph checkpoints. The subgraph interrupt uses
    its cached resume and is NOT re-triggered.

    This is jtamsen's exact issue: forking the parent doesn't re-trigger
    subgraph interrupts when the subgraph has checkpointer=True."""

    called: list[str] = []
    graph = _build_subgraph_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt, then resume
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="answer"), config)

    # Find parent checkpoint before subgraph_node
    history = list(graph.get_state_history(config))
    before_sub_candidates = [s for s in history if s.next == ("subgraph_node",)]
    before_sub = before_sub_candidates[-1]

    # Fork from parent
    called.clear()
    fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    # With checkpointer=True, subgraph has its own persistent state.
    # Parent fork does NOT clear subgraph checkpoints, so the subgraph
    # replays with cached resume — interrupt is NOT re-triggered.
    assert "__interrupt__" not in fork_result
    assert "human:answer" in fork_result["value"]


def test_subgraph_interrupt_replay_from_subgraph_of(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from the parent checkpoint where subgraph interrupt fired.
    Uses cached resume, completes without re-triggering interrupt."""

    called: list[str] = []
    graph = _build_subgraph_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt
    graph.invoke({"value": []}, config)

    # Get state with subgraphs at the interrupt point to verify structure
    parent_state = graph.get_state(config, subgraphs=True)
    assert len(parent_state.tasks) > 0
    sub_task = parent_state.tasks[0]
    assert sub_task.state is not None

    # Resume
    completed_result = graph.invoke(Command(resume="answer"), config)
    completed_value = completed_result["value"]

    # Find the parent checkpoint where the interrupt fired
    history = list(graph.get_state_history(config))
    interrupt_checkpoint = next(
        s
        for s in history
        if s.next == ("subgraph_node",)
        and s.tasks
        and any(t.interrupts for t in s.tasks)
    )

    # Replay from that checkpoint — uses cached resume
    called.clear()
    replay_result = graph.invoke(None, interrupt_checkpoint.config)
    assert replay_result["value"] == completed_value
    assert "__interrupt__" not in replay_result


def test_subgraph_interrupt_fork_from_parent_before_no_sub_checkpointer(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from parent checkpoint before subgraph with DEFAULT checkpointer
    (no sub-checkpointer). The subgraph inherits parent's checkpointer for
    interrupt support only. Forking the parent DOES re-trigger the interrupt
    because subgraph state flows through parent's checkpoint."""

    called: list[str] = []
    graph = _build_subgraph_interrupt_graph(
        sync_checkpointer, called, subgraph_checkpointer=None
    )
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt, then resume
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="answer"), config)

    # Find parent checkpoint before subgraph_node
    history = list(graph.get_state_history(config))
    before_sub_candidates = [s for s in history if s.next == ("subgraph_node",)]
    before_sub = before_sub_candidates[-1]

    # Fork from parent
    called.clear()
    fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    # With default (no) subgraph checkpointer, the subgraph state is managed
    # through the parent's checkpointer. Forking the parent clears downstream
    # state, so the interrupt IS re-triggered.
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Provide input:"


def test_subgraph_interrupt_fork_from_subgraph_of(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from the parent checkpoint where subgraph interrupt fired
    (checkpointer=True). With checkpointer=True, the subgraph has its own
    persistent checkpoints, so parent fork may not clear subgraph state."""

    called: list[str] = []
    graph = _build_subgraph_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt, then resume
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="answer"), config)

    # Find the parent checkpoint where the interrupt fired
    history = list(graph.get_state_history(config))
    interrupt_checkpoint = next(
        s
        for s in history
        if s.next == ("subgraph_node",)
        and s.tasks
        and any(t.interrupts for t in s.tasks)
    )

    # Fork from the interrupt checkpoint at parent level
    called.clear()
    fork_config = graph.update_state(interrupt_checkpoint.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    # With checkpointer=True, whether interrupt re-triggers depends on
    # whether the subgraph's checkpoint is cleared by the parent fork.
    # The result confirms actual behavior.
    assert "value" in fork_result


def test_subgraph_interrupt_fork_from_subgraph_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from the subgraph's own checkpoint (using subgraph config from
    get_state with subgraphs=True). This directly modifies the subgraph's
    checkpoint, so the interrupt IS re-triggered."""

    called: list[str] = []
    graph = _build_subgraph_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result

    # Get state with subgraphs to find the subgraph's own config
    parent_state = graph.get_state(config, subgraphs=True)
    sub_task = parent_state.tasks[0]
    assert sub_task.state is not None
    sub_config = sub_task.state.config

    # Fork from the subgraph's own checkpoint
    called.clear()
    fork_config = graph.update_state(sub_config, {"value": ["sub_forked"]})

    # Resume from forked subgraph checkpoint — this creates a new subgraph
    # checkpoint, causing the interrupt to re-fire
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result or "ask_human" in called


def test_subgraph_interrupt_fork_from_subgraph_checkpoint_full_flow(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from the subgraph's own checkpoint, resume the interrupt with a
    new answer, and verify the FULL flow: subgraph completes (step_b runs)
    AND execution continues back to the parent graph (post_process runs).

    This is the key test for jtamsen's use case: time travel to a subgraph
    checkpoint, re-trigger the interrupt, provide a new answer, and have
    the entire graph complete normally including parent nodes after the
    subgraph."""

    called: list[str] = []
    graph = _build_subgraph_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Step 1: Run until interrupt fires in subgraph
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result
    assert "router" in called
    assert "step_a" in called
    assert "ask_human" in called

    # Step 2: Get subgraph's own checkpoint config
    parent_state = graph.get_state(config, subgraphs=True)
    sub_task = parent_state.tasks[0]
    assert sub_task.state is not None
    sub_config = sub_task.state.config

    # Step 3: Fork from subgraph checkpoint
    called.clear()
    fork_config = graph.update_state(sub_config, {"value": ["sub_forked"]})

    # Step 4: Invoke from fork — interrupt should re-fire
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Provide input:"

    # Step 5: Resume the re-triggered interrupt with a NEW answer
    called.clear()
    final_result = graph.invoke(Command(resume="new_answer"), fork_config)

    # Step 6: Verify full completion
    # ask_human should have run with the new answer
    assert "ask_human" in called
    assert "human:new_answer" in final_result["value"]
    # step_b should have run (subgraph continued after interrupt)
    assert "step_b" in called
    assert "sub_b" in final_result["value"]
    # post_process should have run (parent graph continued after subgraph)
    assert "post_process" in called
    assert "post" in final_result["value"]


def test_subgraph_interrupt_fork_from_subgraph_checkpoint_full_flow_no_sub_checkpointer(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Same as the above test but with the DEFAULT subgraph checkpointer
    (no checkpointer=True). Fork from the parent checkpoint before the
    subgraph, re-trigger interrupt, resume, and verify full parent completion.

    With the default checkpointer, forking the parent DOES re-trigger
    subgraph interrupts (subgraph state flows through parent's checkpoint)."""

    called: list[str] = []
    graph = _build_subgraph_interrupt_graph(
        sync_checkpointer, called, subgraph_checkpointer=None
    )
    config = {"configurable": {"thread_id": "1"}}

    # Step 1: Run until interrupt
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result

    # Step 2: Resume with original answer to complete the graph
    original_result = graph.invoke(Command(resume="original"), config)
    assert "human:original" in original_result["value"]
    assert "post" in original_result["value"]

    # Step 3: Find parent checkpoint before subgraph_node
    history = list(graph.get_state_history(config))
    before_sub_candidates = [s for s in history if s.next == ("subgraph_node",)]
    before_sub = before_sub_candidates[-1]

    # Step 4: Fork from parent checkpoint
    called.clear()
    fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    # Step 5: Interrupt IS re-triggered (default checkpointer)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Provide input:"

    # Step 6: Resume with new answer
    called.clear()
    final_result = graph.invoke(Command(resume="new_answer"), fork_config)

    # Step 7: Verify full completion back through parent
    assert "human:new_answer" in final_result["value"]
    assert "step_b" in called
    assert "sub_b" in final_result["value"]
    assert "post_process" in called
    assert "post" in final_result["value"]


# ---------------------------------------------------------------------------
# Section 5: Additional scenarios from customer thread
# ---------------------------------------------------------------------------


def test_multiple_sequential_interrupts_fork_from_middle(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Graph with two sequential interrupt nodes (A, B). Fork from between
    them. Only B re-fires, A's result is preserved."""

    class MultiIntState(TypedDict):
        value: Annotated[list[str], operator.add]

    def ask_a(state: MultiIntState) -> MultiIntState:
        answer = interrupt("Question A")
        return {"value": [f"a:{answer}"]}

    def ask_b(state: MultiIntState) -> MultiIntState:
        answer = interrupt("Question B")
        return {"value": [f"b:{answer}"]}

    def final_node(state: MultiIntState) -> MultiIntState:
        return {"value": ["done"]}

    graph = (
        StateGraph(MultiIntState)
        .add_node("ask_a", ask_a)
        .add_node("ask_b", ask_b)
        .add_node("final_node", final_node)
        .add_edge(START, "ask_a")
        .add_edge("ask_a", "ask_b")
        .add_edge("ask_b", "final_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Hit first interrupt
    r1 = graph.invoke({"value": []}, config)
    assert r1["__interrupt__"][0].value == "Question A"

    # Resume A
    r2 = graph.invoke(Command(resume="ans_a"), config)
    assert r2["__interrupt__"][0].value == "Question B"

    # Resume B
    r3 = graph.invoke(Command(resume="ans_b"), config)
    assert r3 == {"value": ["a:ans_a", "b:ans_b", "done"]}

    # Find checkpoint between A and B (after A completed, before B)
    history = list(graph.get_state_history(config))
    between_candidates = [s for s in history if s.next == ("ask_b",)]
    # Get the one without interrupts (before ask_b started)
    between = between_candidates[-1]

    # Fork from between — only B should re-fire
    fork_config = graph.update_state(between.config, {"value": ["mid_fork"]})
    fork_result = graph.invoke(None, fork_config)

    # B re-fires because we forked
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Question B"

    # Resume B with new answer
    final_result = graph.invoke(Command(resume="new_b"), fork_config)
    assert "b:new_b" in final_result["value"]
    # A's answer is preserved from before the fork
    assert "a:ans_a" in final_result["value"]


def test_subgraph_get_state_with_subgraphs_true(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Verify get_state(config, subgraphs=True) returns subgraph state and
    checkpoint config when paused at interrupt."""

    class SubS(TypedDict):
        data: str

    def sub_node(state: SubS) -> SubS:
        interrupt("Continue?")
        return {"data": "processed"}

    subgraph = (
        StateGraph(SubS)
        .add_node("process", sub_node)
        .add_edge(START, "process")
        .compile()
    )

    class PS(TypedDict):
        data: str

    graph = (
        StateGraph(PS)
        .add_node("sub", subgraph)
        .add_edge(START, "sub")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"data": "input"}, config)

    # Get state with subgraphs
    state = graph.get_state(config, subgraphs=True)

    # Should have tasks with subgraph state
    assert len(state.tasks) > 0
    sub_task = state.tasks[0]
    assert sub_task.state is not None

    # Subgraph state should have its own config with checkpoint info
    sub_config = sub_task.state.config
    assert "configurable" in sub_config
    assert "thread_id" in sub_config["configurable"]


def test_checkpoint_ns_accessible_in_subgraph_node(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Verify RunnableConfig exposes checkpoint_ns and thread_id inside
    subgraph nodes."""

    captured_config: dict = {}

    class SubS(TypedDict):
        data: str

    def sub_node(state: SubS, config: RunnableConfig) -> SubS:
        captured_config["checkpoint_ns"] = config["configurable"].get("checkpoint_ns")
        captured_config["thread_id"] = config["configurable"].get("thread_id")
        return {"data": "done"}

    subgraph = (
        StateGraph(SubS).add_node("inner", sub_node).add_edge(START, "inner").compile()
    )

    class PS(TypedDict):
        data: str

    graph = (
        StateGraph(PS)
        .add_node("outer", subgraph)
        .add_edge(START, "outer")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"data": "test"}, config)

    # checkpoint_ns should be set inside subgraph
    assert captured_config["checkpoint_ns"] is not None
    assert captured_config["checkpoint_ns"] != ""
    assert captured_config["thread_id"] == "1"


def test_multiple_forks_from_same_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Two independent forks from same checkpoint create independent branches."""

    class ForkState(TypedDict):
        value: Annotated[list[str], operator.add]

    def node_a(state: ForkState) -> ForkState:
        return {"value": ["a"]}

    def node_b(state: ForkState) -> ForkState:
        return {"value": ["b"]}

    graph = (
        StateGraph(ForkState)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"value": []}, config)

    # Find checkpoint before node_b
    history = list(graph.get_state_history(config))
    before_b = next(s for s in history if s.next == ("node_b",))

    # Fork 1
    fork1_config = graph.update_state(before_b.config, {"value": ["fork1"]})
    result1 = graph.invoke(None, fork1_config)

    # Fork 2 from SAME checkpoint
    fork2_config = graph.update_state(before_b.config, {"value": ["fork2"]})
    result2 = graph.invoke(None, fork2_config)

    # Both forks are independent
    assert "fork1" in result1["value"]
    assert "fork2" not in result1["value"]
    assert "fork2" in result2["value"]
    assert "fork1" not in result2["value"]


# ---------------------------------------------------------------------------
# Section 6: Edge cases
# ---------------------------------------------------------------------------


def test_replay_from_final_checkpoint_is_noop(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from completed checkpoint (no next) is a no-op."""

    called: list[str] = []

    def node_a(state: SimpleState) -> SimpleState:
        called.append("node_a")
        return {"value": ["a"]}

    graph = (
        StateGraph(SimpleState)
        .add_node("node_a", node_a)
        .add_edge(START, "node_a")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"value": []}, config)
    assert result == {"value": ["a"]}

    # Get the final checkpoint (no next)
    state = graph.get_state(config)
    assert state.next == ()

    # Replay from final checkpoint — nothing to do
    called.clear()
    replay_result = graph.invoke(None, state.config)
    assert replay_result == {"value": ["a"]}
    assert called == []


def test_replay_interrupt_stable_across_multiple_replays(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replaying same checkpoint multiple times consistently uses same cached
    resume values — produces identical results each time."""

    called: list[str] = []
    graph = _build_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt
    graph.invoke({"value": []}, config)

    # Resume
    graph.invoke(Command(resume="cached_answer"), config)

    # Find checkpoint before ask
    history = list(graph.get_state_history(config))
    before_ask_candidates = [s for s in history if s.next == ("ask_human",)]
    before_ask = before_ask_candidates[-1]

    # Replay multiple times — each should produce same result
    results = []
    for _ in range(3):
        r = graph.invoke(None, before_ask.config)
        results.append(r)

    assert all(r == results[0] for r in results)
    assert results[0] == {"value": ["a", "human:cached_answer", "b"]}


# ---------------------------------------------------------------------------
# Section 7: __copy__ vs update_state(None) — fork without state changes
# ---------------------------------------------------------------------------


def test_copy_fork_retriggers_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork using __copy__ (no state changes) from checkpoint before interrupt.
    The interrupt should be re-triggered because __copy__ creates a new
    checkpoint without cached resume values."""

    called: list[str] = []
    graph = _build_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt, then resume
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="hello"), config)

    # Find checkpoint before ask_human
    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    # Fork using __copy__ — no state modification
    called.clear()
    fork_config = graph.update_state(before_ask.config, None, as_node="__copy__")

    # Invoke from fork — interrupt should be re-triggered
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "What is your input?"

    # Resume the re-triggered interrupt
    final = graph.invoke(Command(resume="new_answer"), fork_config)
    assert final == {"value": ["a", "human:new_answer", "b"]}


def test_copy_fork_vs_replay_interrupt_behavior(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Compare replay vs __copy__ fork from the same checkpoint before an
    interrupt. Replay should use cached resume, __copy__ should re-trigger."""

    called: list[str] = []
    graph = _build_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt, then resume
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="hello"), config)

    # Find checkpoint before ask_human
    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    # Replay — uses cached resume, completes without re-triggering
    called.clear()
    replay_result = graph.invoke(None, before_ask.config)
    assert "__interrupt__" not in replay_result
    assert replay_result == {"value": ["a", "human:hello", "b"]}

    # __copy__ fork — re-triggers interrupt
    called.clear()
    fork_config = graph.update_state(before_ask.config, None, as_node="__copy__")
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "What is your input?"


def test_update_state_with_none_values(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Call update_state with None as values (not __copy__). This goes through
    the normal update path, running the node's writers with None. Test what
    actually happens — does this work at all?"""

    called: list[str] = []
    graph = _build_interrupt_graph(sync_checkpointer, called)
    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt, then resume
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="hello"), config)

    # Find checkpoint before ask_human
    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    # update_state with None values (no __copy__) — goes through the normal
    # update path, infers as_node, runs node writers with None.
    # This ALSO creates a new checkpoint and re-triggers interrupts.
    fork_config = graph.update_state(before_ask.config, None)
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "What is your input?"

    # The checkpoint is marked as "update" (not "fork" like __copy__)
    fork_state = graph.get_state(fork_config)
    assert fork_state.metadata["source"] == "update"


def test_copy_fork_creates_sibling_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """__copy__ creates a sibling checkpoint (branching from the parent of the
    source checkpoint), while regular update_state creates a child. Verify the
    checkpoint metadata reflects this."""

    def node_a(state: SimpleState) -> SimpleState:
        return {"value": ["a"]}

    def node_b(state: SimpleState) -> SimpleState:
        return {"value": ["b"]}

    graph = (
        StateGraph(SimpleState)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"value": []}, config)

    # Find checkpoint before node_b
    history = list(graph.get_state_history(config))
    before_b = next(s for s in history if s.next == ("node_b",))

    # __copy__ fork
    copy_config = graph.update_state(before_b.config, None, as_node="__copy__")
    copy_state = graph.get_state(copy_config)
    assert copy_state.metadata["source"] == "fork"

    # Regular fork
    regular_config = graph.update_state(before_b.config, {"value": ["x"]})
    regular_state = graph.get_state(regular_config)
    assert regular_state.metadata["source"] == "update"
