"""Tests for time travel (replay and fork) behavior.

Covers the intersection of replay vs fork across graph structures:
- Replay & fork basics (no interrupt, no subgraph)
- Replay & fork with interrupts (no subgraph)
- Multiple / sequential interrupts
- Subgraph without interrupt
- Subgraph with interrupt
- __copy__ / update_state(None)
- Observability (get_state, config access)

Key concepts:
- Replay (invoke with checkpoint_id): Re-executes nodes after the checkpoint.
  Interrupts re-fire on replay.
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


class State(TypedDict):
    value: Annotated[list[str], operator.add]


def _checkpoint_summary(history: list) -> list[dict]:
    """Summarize checkpoint history into a readable format for assertions.

    Returns a list of dicts (newest-first, matching get_state_history order) with:
      - id: short checkpoint id suffix (last 6 chars)
      - parent_id: short parent checkpoint id suffix or None
      - next: tuple of next node names
      - values: channel values snapshot
    """
    summaries = []
    for s in history:
        cid = s.config["configurable"]["checkpoint_id"]
        pid = (
            s.parent_config["configurable"]["checkpoint_id"]
            if s.parent_config
            else None
        )
        summaries.append(
            {
                "id": cid[-6:],
                "parent_id": pid[-6:] if pid else None,
                "next": s.next,
                "values": s.values,
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# Section 1: Replay & fork basics (no interrupt, no subgraph)
# ---------------------------------------------------------------------------


def test_replay_reruns_nodes_after_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from checkpoint before node_b. node_b re-executes (it's after
    the checkpoint), node_a does not."""

    called: list[str] = []

    def node_a(state: State) -> State:
        called.append("node_a")
        return {"value": ["a"]}

    def node_b(state: State) -> State:
        called.append("node_b")
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
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

    assert replay_result == {"value": ["a", "b"]}
    assert "node_b" in called
    assert "node_a" not in called


def test_replay_from_final_checkpoint_is_noop(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from completed checkpoint (no next nodes) is a no-op."""

    called: list[str] = []

    def node_a(state: State) -> State:
        called.append("node_a")
        return {"value": ["a"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_edge(START, "node_a")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"value": []}, config)
    assert result == {"value": ["a"]}

    state = graph.get_state(config)
    assert state.next == ()

    called.clear()
    replay_result = graph.invoke(None, state.config)
    assert replay_result == {"value": ["a"]}
    assert called == []


def test_fork_reruns_with_modified_state(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from checkpoint before node_b with modified state. node_b
    re-executes with the new state."""

    called: list[str] = []

    def node_a(state: State) -> State:
        called.append("node_a")
        return {"value": ["a"]}

    def node_b(state: State) -> State:
        called.append("node_b")
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"value": []}, config)

    history = list(graph.get_state_history(config))
    before_b = next(s for s in history if s.next == ("node_b",))

    called.clear()
    fork_config = graph.update_state(before_b.config, {"value": ["x"]})
    fork_result = graph.invoke(None, fork_config)

    assert "node_b" in called
    assert fork_result == {"value": ["a", "x", "b"]}


def test_multiple_forks_from_same_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Two independent forks from the same checkpoint create independent
    branches that don't affect each other."""

    def node_a(state: State) -> State:
        return {"value": ["a"]}

    def node_b(state: State) -> State:
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"value": []}, config)

    history = list(graph.get_state_history(config))
    before_b = next(s for s in history if s.next == ("node_b",))

    fork1_config = graph.update_state(before_b.config, {"value": ["fork1"]})
    result1 = graph.invoke(None, fork1_config)

    fork2_config = graph.update_state(before_b.config, {"value": ["fork2"]})
    result2 = graph.invoke(None, fork2_config)

    assert "fork1" in result1["value"]
    assert "fork2" not in result1["value"]
    assert "fork2" in result2["value"]
    assert "fork1" not in result2["value"]


# ---------------------------------------------------------------------------
# Section 2: Replay & fork with interrupts (no subgraph)
# ---------------------------------------------------------------------------


def test_replay_from_before_interrupt_refires(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from checkpoint before interrupt node. The node re-executes
    and interrupt re-fires."""

    call_count: dict[str, int] = {"node_a": 0, "ask_human": 0, "node_b": 0}

    def node_a(state: State) -> State:
        call_count["node_a"] += 1
        return {"value": ["a"]}

    def ask_human(state: State) -> State:
        call_count["ask_human"] += 1
        answer = interrupt("What is your input?")
        return {"value": [f"human:{answer}"]}

    def node_b(state: State) -> State:
        call_count["node_b"] += 1
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("ask_human", ask_human)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "ask_human")
        .add_edge("ask_human", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result
    assert call_count["ask_human"] == 1

    # Resume
    result = graph.invoke(Command(resume="hello"), config)
    assert result == {"value": ["a", "human:hello", "b"]}
    assert call_count["ask_human"] == 2  # re-executes on resume

    # Find checkpoint before ask_human
    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    # Replay — interrupt re-fires, node re-executes
    replay_result = graph.invoke(None, before_ask.config)

    assert "__interrupt__" in replay_result
    assert replay_result["__interrupt__"][0].value == "What is your input?"
    assert replay_result["value"] == ["a"]
    assert call_count["ask_human"] == 3  # re-executed again
    assert call_count["node_a"] == 1  # NOT re-executed (before checkpoint)
    assert call_count["node_b"] == 1  # NOT re-executed (after interrupt)


def test_replay_interrupt_stable_across_replays(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replaying the same checkpoint multiple times consistently produces
    identical results (interrupt re-fires each time)."""

    def node_a(state: State) -> State:
        return {"value": ["a"]}

    def ask_human(state: State) -> State:
        answer = interrupt("What is your input?")
        return {"value": [f"human:{answer}"]}

    def node_b(state: State) -> State:
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("ask_human", ask_human)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "ask_human")
        .add_edge("ask_human", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="cached_answer"), config)

    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    results = []
    for _ in range(3):
        r = graph.invoke(None, before_ask.config)
        results.append(r)

    assert all(r == results[0] for r in results)
    assert "__interrupt__" in results[0]


def test_fork_from_before_interrupt_refires(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from checkpoint before interrupt node. Interrupt IS re-triggered
    because fork has no cached resume values. Resume with new answer."""

    called: list[str] = []

    def node_a(state: State) -> State:
        called.append("node_a")
        return {"value": ["a"]}

    def ask_human(state: State) -> State:
        called.append("ask_human")
        answer = interrupt("What is your input?")
        return {"value": [f"human:{answer}"]}

    def node_b(state: State) -> State:
        called.append("node_b")
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("ask_human", ask_human)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "ask_human")
        .add_edge("ask_human", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="hello"), config)

    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    called.clear()
    fork_config = graph.update_state(before_ask.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "What is your input?"

    # Resume the forked interrupt with a different answer
    final = graph.invoke(Command(resume="world"), fork_config)
    assert final == {"value": ["a", "forked", "human:world", "b"]}


def test_fork_from_interrupt_checkpoint_refires(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from the checkpoint where interrupt fired. Interrupt re-triggered
    because fork clears cached data. Resume with different answer."""

    called: list[str] = []

    def node_a(state: State) -> State:
        called.append("node_a")
        return {"value": ["a"]}

    def ask_human(state: State) -> State:
        called.append("ask_human")
        answer = interrupt("What is your input?")
        return {"value": [f"human:{answer}"]}

    def node_b(state: State) -> State:
        called.append("node_b")
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("ask_human", ask_human)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "ask_human")
        .add_edge("ask_human", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="hello"), config)

    history = list(graph.get_state_history(config))
    interrupt_checkpoint = next(
        s
        for s in history
        if s.next == ("ask_human",) and s.tasks and any(t.interrupts for t in s.tasks)
    )

    fork_config = graph.update_state(interrupt_checkpoint.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    assert "__interrupt__" in fork_result

    final = graph.invoke(Command(resume="different"), fork_config)
    assert "human:different" in final["value"]


# ---------------------------------------------------------------------------
# Section 3: Multiple / sequential interrupts
# ---------------------------------------------------------------------------


def test_sequential_interrupts_fork_from_middle(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Graph with two sequential interrupt nodes. Fork from between them:
    only the second re-fires, the first's result is preserved. Also verify
    replaying from before the first re-fires only the first."""

    called: list[str] = []

    def node_a(state: State) -> State:
        called.append("node_a")
        return {"value": ["a"]}

    def interrupt_1(state: State) -> State:
        called.append("interrupt_1")
        answer = interrupt("First question?")
        return {"value": [f"i1:{answer}"]}

    def interrupt_2(state: State) -> State:
        called.append("interrupt_2")
        answer = interrupt("Second question?")
        return {"value": [f"i2:{answer}"]}

    def node_b(state: State) -> State:
        called.append("node_b")
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("interrupt_1", interrupt_1)
        .add_node("interrupt_2", interrupt_2)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "interrupt_1")
        .add_edge("interrupt_1", "interrupt_2")
        .add_edge("interrupt_2", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Hit first interrupt
    r1 = graph.invoke({"value": []}, config)
    assert r1["__interrupt__"][0].value == "First question?"

    # Resume first → hit second
    r2 = graph.invoke(Command(resume="ans1"), config)
    assert r2["__interrupt__"][0].value == "Second question?"

    # Resume second → complete
    r3 = graph.invoke(Command(resume="ans2"), config)
    assert r3 == {"value": ["a", "i1:ans1", "i2:ans2", "b"]}

    history = list(graph.get_state_history(config))

    # Fork from between the two interrupts — only second re-fires
    between = [s for s in history if s.next == ("interrupt_2",)][-1]
    fork_config = graph.update_state(between.config, {"value": ["mid_fork"]})
    fork_result = graph.invoke(None, fork_config)

    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Second question?"

    # Resume with new answer, first answer preserved
    final_result = graph.invoke(Command(resume="new_b"), fork_config)
    assert "i2:new_b" in final_result["value"]
    assert "i1:ans1" in final_result["value"]

    # Replay from before first interrupt — first re-fires, second does not
    before_i1 = [s for s in history if s.next == ("interrupt_1",)][-1]
    called.clear()
    replay_result = graph.invoke(None, before_i1.config)
    assert "__interrupt__" in replay_result
    assert replay_result["__interrupt__"][0].value == "First question?"
    assert "interrupt_1" in called
    assert "interrupt_2" not in called


def test_multiple_interrupts_in_one_node(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """A single node with two sequential interrupt() calls. Resuming resolves
    them one at a time. Replaying from before the node re-fires the first."""

    def multi_interrupt_node(state: State) -> State:
        answer1 = interrupt("First question?")
        answer2 = interrupt("Second question?")
        return {"value": [f"a1:{answer1}", f"a2:{answer2}"]}

    def after(state: State) -> State:
        return {"value": ["done"]}

    graph = (
        StateGraph(State)
        .add_node("ask", multi_interrupt_node)
        .add_node("after", after)
        .add_edge(START, "ask")
        .add_edge("ask", "after")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Hit first interrupt
    result = graph.invoke({"value": []}, config)
    assert result["__interrupt__"][0].value == "First question?"

    # Resume first → hit second
    interrupt_state = graph.get_state(config)
    result = graph.invoke(Command(resume="ans1"), interrupt_state.config)
    assert result["__interrupt__"][0].value == "Second question?"

    # Resume second → complete
    interrupt_state2 = graph.get_state(config)
    result = graph.invoke(Command(resume="ans2"), interrupt_state2.config)
    assert result == {"value": ["a1:ans1", "a2:ans2", "done"]}

    # Replay from before the node — first interrupt re-fires
    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask",)][-1]
    replay_result = graph.invoke(None, before_ask.config)
    assert replay_result["__interrupt__"][0].value == "First question?"


# ---------------------------------------------------------------------------
# Section 4: Subgraph without interrupt
# ---------------------------------------------------------------------------


def test_subgraph_replay_from_before(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from parent checkpoint before subgraph node. Subgraph and
    post_process re-execute, parent_node does not."""

    called: list[str] = []

    def parent_node(state: State) -> State:
        called.append("parent_node")
        return {"value": ["parent"]}

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["sub_a"]}

    def step_b(state: State) -> State:
        called.append("step_b")
        return {"value": ["sub_b"]}

    subgraph = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "step_b")
        .compile()
    )

    def post_process(state: State) -> State:
        called.append("post_process")
        return {"value": ["post"]}

    graph = (
        StateGraph(State)
        .add_node("parent_node", parent_node)
        .add_node("subgraph", subgraph)
        .add_node("post_process", post_process)
        .add_edge(START, "parent_node")
        .add_edge("parent_node", "subgraph")
        .add_edge("subgraph", "post_process")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"value": []}, config)
    assert "sub_a" in result["value"]
    assert "sub_b" in result["value"]
    assert "post" in result["value"]

    history = list(graph.get_state_history(config))
    before_sub = next(s for s in history if s.next == ("subgraph",))

    called.clear()
    replay_result = graph.invoke(None, before_sub.config)

    assert "sub_a" in replay_result["value"]
    assert "sub_b" in replay_result["value"]
    assert "post" in replay_result["value"]
    assert "parent_node" not in called


def test_subgraph_fork_from_before(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from parent checkpoint before subgraph with modified state.
    Subgraph re-executes with forked state."""

    called: list[str] = []

    def parent_node(state: State) -> State:
        called.append("parent_node")
        return {"value": ["parent"]}

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["sub_a"]}

    def step_b(state: State) -> State:
        called.append("step_b")
        return {"value": ["sub_b"]}

    subgraph = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "step_b")
        .compile()
    )

    def post_process(state: State) -> State:
        called.append("post_process")
        return {"value": ["post"]}

    graph = (
        StateGraph(State)
        .add_node("parent_node", parent_node)
        .add_node("subgraph", subgraph)
        .add_node("post_process", post_process)
        .add_edge(START, "parent_node")
        .add_edge("parent_node", "subgraph")
        .add_edge("subgraph", "post_process")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"value": []}, config)

    history = list(graph.get_state_history(config))
    before_sub = next(s for s in history if s.next == ("subgraph",))

    called.clear()
    fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    assert "step_a" in called
    assert "step_b" in called
    assert "post_process" in called
    assert "forked" in fork_result["value"]
    assert "sub_a" in fork_result["value"]
    assert "sub_b" in fork_result["value"]


# ---------------------------------------------------------------------------
# Section 5: Subgraph with interrupt
# ---------------------------------------------------------------------------


def test_subgraph_interrupt_replay_from_parent(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from parent checkpoint before subgraph. Subgraph re-executes
    and interrupt re-fires."""

    called: list[str] = []

    def router(state: State) -> State:
        called.append("router")
        return {"value": ["routed"]}

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["sub_a"]}

    def ask_human(state: State) -> State:
        called.append("ask_human")
        answer = interrupt("Provide input:")
        return {"value": [f"human:{answer}"]}

    def step_b(state: State) -> State:
        called.append("step_b")
        return {"value": ["sub_b"]}

    subgraph = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_human", ask_human)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_human")
        .add_edge("ask_human", "step_b")
        .compile(checkpointer=True)
    )

    def post_process(state: State) -> State:
        called.append("post_process")
        return {"value": ["post"]}

    graph = (
        StateGraph(State)
        .add_node("router", router)
        .add_node("subgraph_node", subgraph)
        .add_node("post_process", post_process)
        .add_edge(START, "router")
        .add_edge("router", "subgraph_node")
        .add_edge("subgraph_node", "post_process")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt, then resume
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result

    completed_result = graph.invoke(Command(resume="answer"), config)
    assert "human:answer" in completed_result["value"]
    assert "sub_b" in completed_result["value"]
    assert "post" in completed_result["value"]

    # Find parent checkpoint before subgraph_node
    history = list(graph.get_state_history(config))
    before_sub = [s for s in history if s.next == ("subgraph_node",)][-1]

    # Replay from before subgraph — subgraph starts fresh, interrupt re-fires
    called.clear()
    replay_result = graph.invoke(None, before_sub.config)
    assert "__interrupt__" in replay_result
    # Subgraph ran from scratch (step_a and ask_human called)
    assert "step_a" in called
    assert "ask_human" in called
    # step_b should NOT be called (interrupt stops execution)
    assert "step_b" not in called


def test_subgraph_interrupt_replay_from_interrupt_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from the parent checkpoint where subgraph interrupt fired.
    Interrupt re-fires."""

    called: list[str] = []

    def router(state: State) -> State:
        called.append("router")
        return {"value": ["routed"]}

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["sub_a"]}

    def ask_human(state: State) -> State:
        called.append("ask_human")
        answer = interrupt("Provide input:")
        return {"value": [f"human:{answer}"]}

    def step_b(state: State) -> State:
        called.append("step_b")
        return {"value": ["sub_b"]}

    subgraph = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_human", ask_human)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_human")
        .add_edge("ask_human", "step_b")
        .compile(checkpointer=True)
    )

    def post_process(state: State) -> State:
        called.append("post_process")
        return {"value": ["post"]}

    graph = (
        StateGraph(State)
        .add_node("router", router)
        .add_node("subgraph_node", subgraph)
        .add_node("post_process", post_process)
        .add_edge(START, "router")
        .add_edge("router", "subgraph_node")
        .add_edge("subgraph_node", "post_process")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt
    graph.invoke({"value": []}, config)

    # Verify subgraph state is accessible
    parent_state = graph.get_state(config, subgraphs=True)
    assert len(parent_state.tasks) > 0
    assert parent_state.tasks[0].state is not None

    # Resume
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

    called.clear()
    replay_result = graph.invoke(None, interrupt_checkpoint.config)
    assert "__interrupt__" in replay_result
    # Subgraph starts fresh during replay — all nodes re-run from scratch.
    # step_a re-runs, ask_human re-fires interrupt, step_b not reached.
    assert "step_a" in called
    assert "ask_human" in called
    assert "step_b" not in called


def test_subgraph_interrupt_full_flow(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from the subgraph's own checkpoint, resume the interrupt with a
    new answer, and verify the FULL flow: subgraph completes (step_b runs)
    AND execution continues back to the parent graph (post_process runs).

    This is the key test for time-traveling to a subgraph checkpoint,
    re-triggering the interrupt, providing a new answer, and having the
    entire graph complete normally including parent nodes after the subgraph."""

    called: list[str] = []

    def router(state: State) -> State:
        called.append("router")
        return {"value": ["routed"]}

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["sub_a"]}

    def ask_human(state: State) -> State:
        called.append("ask_human")
        answer = interrupt("Provide input:")
        return {"value": [f"human:{answer}"]}

    def step_b(state: State) -> State:
        called.append("step_b")
        return {"value": ["sub_b"]}

    subgraph = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_human", ask_human)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_human")
        .add_edge("ask_human", "step_b")
        .compile(checkpointer=True)
    )

    def post_process(state: State) -> State:
        called.append("post_process")
        return {"value": ["post"]}

    graph = (
        StateGraph(State)
        .add_node("router", router)
        .add_node("subgraph_node", subgraph)
        .add_node("post_process", post_process)
        .add_edge(START, "router")
        .add_edge("router", "subgraph_node")
        .add_edge("subgraph_node", "post_process")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt fires in subgraph
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result
    assert "router" in called
    assert "step_a" in called
    assert "ask_human" in called

    # Get subgraph's own checkpoint config
    parent_state = graph.get_state(config, subgraphs=True)
    sub_task = parent_state.tasks[0]
    assert sub_task.state is not None
    sub_config = sub_task.state.config

    # Fork from subgraph checkpoint
    called.clear()
    fork_config = graph.update_state(sub_config, {"value": ["sub_forked"]})

    # Invoke from fork — interrupt should re-fire
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Provide input:"

    # Resume the re-triggered interrupt with a NEW answer
    called.clear()
    final_result = graph.invoke(Command(resume="new_answer"), fork_config)

    # Verify full completion
    assert "ask_human" in called
    assert "human:new_answer" in final_result["value"]
    assert "step_b" in called
    assert "sub_b" in final_result["value"]
    assert "post_process" in called
    assert "post" in final_result["value"]


def test_subgraph_interrupt_full_flow_no_sub_checkpointer(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Same as test_subgraph_interrupt_full_flow but with no sub-checkpointer
    (checkpointer=None). Fork from the parent checkpoint before the subgraph,
    re-trigger interrupt, resume, and verify full parent completion."""

    called: list[str] = []

    def router(state: State) -> State:
        called.append("router")
        return {"value": ["routed"]}

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["sub_a"]}

    def ask_human(state: State) -> State:
        called.append("ask_human")
        answer = interrupt("Provide input:")
        return {"value": [f"human:{answer}"]}

    def step_b(state: State) -> State:
        called.append("step_b")
        return {"value": ["sub_b"]}

    subgraph = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_human", ask_human)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_human")
        .add_edge("ask_human", "step_b")
        .compile(checkpointer=None)
    )

    def post_process(state: State) -> State:
        called.append("post_process")
        return {"value": ["post"]}

    graph = (
        StateGraph(State)
        .add_node("router", router)
        .add_node("subgraph_node", subgraph)
        .add_node("post_process", post_process)
        .add_edge(START, "router")
        .add_edge("router", "subgraph_node")
        .add_edge("subgraph_node", "post_process")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt, then resume to complete
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result

    original_result = graph.invoke(Command(resume="original"), config)
    assert "human:original" in original_result["value"]
    assert "post" in original_result["value"]

    # Find parent checkpoint before subgraph_node
    history = list(graph.get_state_history(config))
    before_sub = [s for s in history if s.next == ("subgraph_node",)][-1]

    # Fork from parent checkpoint
    called.clear()
    fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    # Interrupt IS re-triggered
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Provide input:"

    # Resume with new answer
    called.clear()
    final_result = graph.invoke(Command(resume="new_answer"), fork_config)

    # Verify full completion back through parent
    assert "human:new_answer" in final_result["value"]
    assert "step_b" in called
    assert "sub_b" in final_result["value"]
    assert "post_process" in called
    assert "post" in final_result["value"]


def test_subgraph_replay_from_subgraph_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay directly from a subgraph's own checkpoint (via get_state with
    subgraphs=True). The subgraph resumes from its checkpoint and the parent
    graph completes normally afterwards."""

    called: list[str] = []

    def router(state: State) -> State:
        called.append("router")
        return {"value": ["routed"]}

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["sub_a"]}

    def ask_human(state: State) -> State:
        called.append("ask_human")
        answer = interrupt("Provide input:")
        return {"value": [f"human:{answer}"]}

    def step_b(state: State) -> State:
        called.append("step_b")
        return {"value": ["sub_b"]}

    subgraph = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_human", ask_human)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_human")
        .add_edge("ask_human", "step_b")
        .compile(checkpointer=True)
    )

    def post_process(state: State) -> State:
        called.append("post_process")
        return {"value": ["post"]}

    graph = (
        StateGraph(State)
        .add_node("router", router)
        .add_node("subgraph_node", subgraph)
        .add_node("post_process", post_process)
        .add_edge(START, "router")
        .add_edge("router", "subgraph_node")
        .add_edge("subgraph_node", "post_process")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until interrupt fires in subgraph
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result

    # Get subgraph checkpoint config (without forking)
    parent_state = graph.get_state(config, subgraphs=True)
    sub_task = parent_state.tasks[0]
    assert sub_task.state is not None
    sub_config = sub_task.state.config

    # Replay directly from subgraph checkpoint (no update_state / no fork)
    called.clear()
    replay_result = graph.invoke(None, sub_config)
    assert "__interrupt__" in replay_result
    assert replay_result["__interrupt__"][0].value == "Provide input:"

    # Resume from the replayed checkpoint
    called.clear()
    final_result = graph.invoke(Command(resume="replayed_answer"), sub_config)

    assert "ask_human" in called
    assert "human:replayed_answer" in final_result["value"]
    assert "step_b" in called
    assert "sub_b" in final_result["value"]
    assert "post_process" in called
    assert "post" in final_result["value"]


def test_subgraph_time_travel_to_first_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Time travel to a subgraph checkpoint at the FIRST interrupt.

    Architecture:
      Parent:    START --> executor (subgraph, checkpointer=True) --> END
      Executor:  START --> step_a --> ask_1 (interrupt) --> ask_2 (interrupt) --> END

    Flow: run through both interrupts, then time travel back to the subgraph
    checkpoint captured at the first interrupt. ask_1 should re-fire,
    step_a should NOT re-run. Then resume through both interrupts with new answers.
    """

    called: list[str] = []

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["step_a_done"]}

    def ask_1(state: State) -> State:
        called.append("ask_1")
        answer = interrupt("Question 1?")
        return {"value": [f"ask_1:{answer}"]}

    def ask_2(state: State) -> State:
        called.append("ask_2")
        answer = interrupt("Question 2?")
        return {"value": [f"ask_2:{answer}"]}

    executor = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_1", ask_1)
        .add_node("ask_2", ask_2)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_1")
        .add_edge("ask_1", "ask_2")
        .add_edge("ask_2", "__end__")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(State)
        .add_node("executor", executor)
        .add_edge(START, "executor")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until first interrupt (ask_1)
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value == "Question 1?"

    # Capture subgraph state at the first interrupt
    parent_state = graph.get_state(config, subgraphs=True)
    sub_config_at_first = parent_state.tasks[0].state.config

    # Resume first interrupt
    result = graph.invoke(Command(resume="answer_1"), config)
    assert result["__interrupt__"][0].value == "Question 2?"

    # Resume second interrupt to complete
    result = graph.invoke(Command(resume="answer_2"), config)
    assert "__interrupt__" not in result

    # --- Scenario 1: Replay from subgraph checkpoint at 1st interrupt ---
    called.clear()
    replay_result = graph.invoke(None, sub_config_at_first)
    assert "__interrupt__" in replay_result
    assert replay_result["__interrupt__"][0].value == "Question 1?"
    # step_a should NOT re-run — it was before this checkpoint
    assert "step_a" not in called
    # ask_1 re-fires because the interrupt replays
    assert "ask_1" in called

    # --- Scenario 2: Fork from subgraph checkpoint at 1st interrupt ---
    called.clear()
    fork_config = graph.update_state(sub_config_at_first, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Question 1?"
    assert "step_a" not in called
    assert "ask_1" in called


def test_subgraph_time_travel_to_second_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Time travel to a subgraph checkpoint at the SECOND interrupt.

    Architecture:
      Parent:    START --> executor (subgraph, checkpointer=True) --> END
      Executor:  START --> step_a --> ask_1 (interrupt) --> ask_2 (interrupt) --> END

    Flow: run through both interrupts resuming each, then time travel back to the
    subgraph checkpoint at the second interrupt. Only ask_2 should re-fire.
    Then resume with a new answer and verify state.
    """

    called: list[str] = []

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["step_a_done"]}

    def ask_1(state: State) -> State:
        called.append("ask_1")
        answer = interrupt("Question 1?")
        return {"value": [f"ask_1:{answer}"]}

    def ask_2(state: State) -> State:
        called.append("ask_2")
        answer = interrupt("Question 2?")
        return {"value": [f"ask_2:{answer}"]}

    executor = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_1", ask_1)
        .add_node("ask_2", ask_2)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_1")
        .add_edge("ask_1", "ask_2")
        .add_edge("ask_2", "__end__")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(State)
        .add_node("executor", executor)
        .add_edge(START, "executor")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until first interrupt (ask_1)
    result = graph.invoke({"value": []}, config)
    assert result["__interrupt__"][0].value == "Question 1?"

    # Resume first interrupt
    result = graph.invoke(Command(resume="answer_1"), config)
    assert result["__interrupt__"][0].value == "Question 2?"

    # Capture subgraph state at the second interrupt
    parent_state = graph.get_state(config, subgraphs=True)
    sub_config = parent_state.tasks[0].state.config

    # Resume second interrupt to complete the graph
    result = graph.invoke(Command(resume="answer_2"), config)
    assert "__interrupt__" not in result

    # --- Scenario 1: Replay from subgraph checkpoint at 2nd interrupt ---
    called.clear()
    replay_result = graph.invoke(None, sub_config)
    assert "__interrupt__" in replay_result
    assert replay_result["__interrupt__"][0].value == "Question 2?"
    # step_a and ask_1 should NOT re-run — they were before this checkpoint
    assert "step_a" not in called
    assert "ask_1" not in called

    # --- Scenario 2: Fork from subgraph checkpoint at 2nd interrupt ---
    called.clear()
    fork_config = graph.update_state(sub_config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Question 2?"
    assert "step_a" not in called
    assert "ask_1" not in called


def test_subgraph_time_travel_after_completion(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Time travel to a subgraph checkpoint AFTER both interrupts are resolved.

    Architecture:
      Parent:    START --> executor (subgraph, checkpointer=True) --> END
      Executor:  START --> step_a --> ask_1 (interrupt) --> ask_2 (interrupt) --> END

    After completing the full flow, capture the subgraph's final state checkpoint
    and replay from it — should be a no-op (no nodes re-run).
    """

    called: list[str] = []

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["step_a_done"]}

    def ask_1(state: State) -> State:
        called.append("ask_1")
        answer = interrupt("Question 1?")
        return {"value": [f"ask_1:{answer}"]}

    def ask_2(state: State) -> State:
        called.append("ask_2")
        answer = interrupt("Question 2?")
        return {"value": [f"ask_2:{answer}"]}

    executor = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_1", ask_1)
        .add_node("ask_2", ask_2)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_1")
        .add_edge("ask_1", "ask_2")
        .add_edge("ask_2", "__end__")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(State)
        .add_node("executor", executor)
        .add_edge(START, "executor")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run through both interrupts
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="answer_1"), config)

    # Before resuming 2nd interrupt, get state history to find the
    # subgraph checkpoint that will exist after ask_2 completes
    graph.invoke(Command(resume="answer_2"), config)

    # Get the final parent state — no pending tasks
    final_state = graph.get_state(config)
    assert len(final_state.tasks) == 0

    # Replay from the final parent checkpoint — should be a no-op
    called.clear()
    replay_result = graph.invoke(None, final_state.config)
    assert "__interrupt__" not in replay_result
    assert "step_a" not in called
    assert "ask_1" not in called
    assert "ask_2" not in called
    # All values should be present
    assert "step_a_done" in replay_result["value"]
    assert "ask_1:answer_1" in replay_result["value"]
    assert "ask_2:answer_2" in replay_result["value"]


def test_3_levels_deep_time_travel_to_first_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Time travel to the innermost subgraph checkpoint at the FIRST interrupt.

    Architecture:
      Parent:  START --> outer (subgraph, checkpointer=True) --> END
      Outer:   START --> inner (subgraph, checkpointer=True) --> END
      Inner:   START --> step_a --> ask_1 (interrupt) --> ask_2 (interrupt) --> END
    """

    called: list[str] = []

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["step_a_done"]}

    def ask_1(state: State) -> State:
        called.append("ask_1")
        answer = interrupt("Question 1?")
        return {"value": [f"ask_1:{answer}"]}

    def ask_2(state: State) -> State:
        called.append("ask_2")
        answer = interrupt("Question 2?")
        return {"value": [f"ask_2:{answer}"]}

    inner = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_1", ask_1)
        .add_node("ask_2", ask_2)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_1")
        .add_edge("ask_1", "ask_2")
        .add_edge("ask_2", "__end__")
        .compile(checkpointer=True)
    )

    middle = (
        StateGraph(State)
        .add_node("inner", inner)
        .add_edge(START, "inner")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(State)
        .add_node("outer", middle)
        .add_edge(START, "outer")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until first interrupt
    result = graph.invoke({"value": []}, config)
    assert result["__interrupt__"][0].value == "Question 1?"

    # Capture innermost subgraph state at the first interrupt
    parent_state = graph.get_state(config, subgraphs=True)
    mid_state = parent_state.tasks[0].state
    inner_config = mid_state.tasks[0].state.config

    # Resume through both interrupts to complete
    graph.invoke(Command(resume="answer_1"), config)
    graph.invoke(Command(resume="answer_2"), config)

    # --- Scenario 1: Replay from innermost checkpoint at 1st interrupt ---
    called.clear()
    replay_result = graph.invoke(None, inner_config)
    assert "__interrupt__" in replay_result
    assert replay_result["__interrupt__"][0].value == "Question 1?"
    assert "step_a" not in called
    assert "ask_1" in called

    # --- Scenario 2: Fork from innermost checkpoint at 1st interrupt ---
    called.clear()
    fork_config = graph.update_state(inner_config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Question 1?"
    assert "step_a" not in called
    assert "ask_1" in called


def test_3_levels_deep_time_travel_to_second_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Time travel to the innermost subgraph checkpoint at the SECOND interrupt.

    Architecture:
      Parent:  START --> outer (subgraph, checkpointer=True) --> END
      Outer:   START --> inner (subgraph, checkpointer=True) --> END
      Inner:   START --> step_a --> ask_1 (interrupt) --> ask_2 (interrupt) --> END
    """

    called: list[str] = []

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["step_a_done"]}

    def ask_1(state: State) -> State:
        called.append("ask_1")
        answer = interrupt("Question 1?")
        return {"value": [f"ask_1:{answer}"]}

    def ask_2(state: State) -> State:
        called.append("ask_2")
        answer = interrupt("Question 2?")
        return {"value": [f"ask_2:{answer}"]}

    inner = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_1", ask_1)
        .add_node("ask_2", ask_2)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_1")
        .add_edge("ask_1", "ask_2")
        .add_edge("ask_2", "__end__")
        .compile(checkpointer=True)
    )

    middle = (
        StateGraph(State)
        .add_node("inner", inner)
        .add_edge(START, "inner")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(State)
        .add_node("outer", middle)
        .add_edge(START, "outer")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until first interrupt
    graph.invoke({"value": []}, config)

    # Resume first interrupt
    result = graph.invoke(Command(resume="answer_1"), config)
    assert result["__interrupt__"][0].value == "Question 2?"

    # Capture innermost subgraph state at the second interrupt
    parent_state = graph.get_state(config, subgraphs=True)
    mid_state = parent_state.tasks[0].state
    inner_config = mid_state.tasks[0].state.config

    # Resume second interrupt to complete
    graph.invoke(Command(resume="answer_2"), config)

    # --- Scenario 1: Replay from innermost checkpoint at 2nd interrupt ---
    called.clear()
    replay_result = graph.invoke(None, inner_config)
    assert "__interrupt__" in replay_result
    assert replay_result["__interrupt__"][0].value == "Question 2?"
    assert "step_a" not in called
    assert "ask_1" not in called

    # --- Scenario 2: Fork from innermost checkpoint at 2nd interrupt ---
    called.clear()
    fork_config = graph.update_state(inner_config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Question 2?"
    assert "step_a" not in called
    assert "ask_1" not in called


def test_3_levels_deep_time_travel_to_middle_subgraph(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Time travel to the MIDDLE-level subgraph checkpoint (not innermost).

    Architecture:
      Parent:  START --> outer (subgraph, checkpointer=True) --> END
      Outer:   START --> inner (subgraph, checkpointer=True) --> END
      Inner:   START --> step_a --> ask_1 (interrupt) --> ask_2 (interrupt) --> END

    After completing the full flow, time travel back to the middle subgraph's
    checkpoint at the second interrupt. The middle subgraph should replay the
    inner subgraph from the correct point.
    """

    called: list[str] = []

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["step_a_done"]}

    def ask_1(state: State) -> State:
        called.append("ask_1")
        answer = interrupt("Question 1?")
        return {"value": [f"ask_1:{answer}"]}

    def ask_2(state: State) -> State:
        called.append("ask_2")
        answer = interrupt("Question 2?")
        return {"value": [f"ask_2:{answer}"]}

    inner = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_1", ask_1)
        .add_node("ask_2", ask_2)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_1")
        .add_edge("ask_1", "ask_2")
        .add_edge("ask_2", "__end__")
        .compile(checkpointer=True)
    )

    middle = (
        StateGraph(State)
        .add_node("inner", inner)
        .add_edge(START, "inner")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(State)
        .add_node("outer", middle)
        .add_edge(START, "outer")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until first interrupt
    graph.invoke({"value": []}, config)

    # Resume first, capture middle config at second interrupt
    graph.invoke(Command(resume="answer_1"), config)
    parent_state = graph.get_state(config, subgraphs=True)
    mid_config = parent_state.tasks[0].state.config

    # Resume second to complete
    graph.invoke(Command(resume="answer_2"), config)

    # --- Scenario 1: Replay from middle-level subgraph checkpoint ---
    # The middle subgraph's checkpoint knows about the inner subgraph's state
    # via checkpoint_map, so the inner replays from the correct point.
    called.clear()
    replay_result = graph.invoke(None, mid_config)
    assert "__interrupt__" in replay_result

    # --- Scenario 2: Fork from middle-level subgraph checkpoint ---
    called.clear()
    fork_config = graph.update_state(mid_config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result


def test_3_levels_deep_middle_has_interrupts(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Time travel when the MIDDLE subgraph itself has interrupts.

    Architecture:
      Parent:  START --> outer (subgraph, checkpointer=True) --> END
      Outer:   START --> pre (interrupt) --> inner (subgraph, checkpointer=True) --> END
      Inner:   START --> step_a --> ask_1 (interrupt) --> END

    Flow: run through both interrupts (pre then ask_1), then time travel back to
    the middle subgraph checkpoint at each interrupt point.
    """

    called: list[str] = []

    def pre(state: State) -> State:
        called.append("pre")
        answer = interrupt("Pre-question?")
        return {"value": [f"pre:{answer}"]}

    def step_a(state: State) -> State:
        called.append("step_a")
        return {"value": ["step_a_done"]}

    def ask_1(state: State) -> State:
        called.append("ask_1")
        answer = interrupt("Question 1?")
        return {"value": [f"ask_1:{answer}"]}

    inner = (
        StateGraph(State)
        .add_node("step_a", step_a)
        .add_node("ask_1", ask_1)
        .add_edge(START, "step_a")
        .add_edge("step_a", "ask_1")
        .add_edge("ask_1", "__end__")
        .compile(checkpointer=True)
    )

    middle = (
        StateGraph(State)
        .add_node("pre", pre)
        .add_node("inner", inner)
        .add_edge(START, "pre")
        .add_edge("pre", "inner")
        .add_edge("inner", "__end__")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(State)
        .add_node("outer", middle)
        .add_edge(START, "outer")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run until first interrupt (pre in middle subgraph)
    result = graph.invoke({"value": []}, config)
    assert result["__interrupt__"][0].value == "Pre-question?"

    # Capture middle subgraph config at the pre interrupt
    parent_state = graph.get_state(config, subgraphs=True)
    mid_config_at_pre = parent_state.tasks[0].state.config

    # Resume pre, hits ask_1 in inner subgraph
    result = graph.invoke(Command(resume="pre_answer"), config)
    assert result["__interrupt__"][0].value == "Question 1?"

    # Capture middle subgraph config at the ask_1 interrupt
    parent_state = graph.get_state(config, subgraphs=True)
    mid_config_at_ask1 = parent_state.tasks[0].state.config

    # Resume ask_1 to complete
    result = graph.invoke(Command(resume="answer_1"), config)
    assert "__interrupt__" not in result

    # --- Time travel to middle checkpoint at pre interrupt ---
    called.clear()
    replay_result = graph.invoke(None, mid_config_at_pre)
    assert "__interrupt__" in replay_result
    assert replay_result["__interrupt__"][0].value == "Pre-question?"
    # pre should re-fire (interrupt replays), but nothing else should run
    assert "pre" in called
    assert "step_a" not in called
    assert "ask_1" not in called

    # Fork from middle checkpoint at pre interrupt
    called.clear()
    fork_config = graph.update_state(mid_config_at_pre, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Pre-question?"
    assert "pre" in called
    assert "step_a" not in called

    # --- Time travel to middle checkpoint at ask_1 interrupt ---
    called.clear()
    replay_result = graph.invoke(None, mid_config_at_ask1)
    assert "__interrupt__" in replay_result
    assert replay_result["__interrupt__"][0].value == "Question 1?"
    # pre should NOT re-run (it completed before this checkpoint)
    assert "pre" not in called
    # ask_1 re-fires
    assert "ask_1" in called

    # Fork from middle checkpoint at ask_1 interrupt
    called.clear()
    fork_config = graph.update_state(mid_config_at_ask1, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Question 1?"
    assert "pre" not in called
    assert "ask_1" in called


# ---------------------------------------------------------------------------
# Section 6: __copy__ / update_state(None)
# ---------------------------------------------------------------------------


def test_copy_fork_retriggers_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork using __copy__ (no state changes) from checkpoint before interrupt.
    The interrupt is re-triggered because __copy__ creates a new checkpoint
    without cached resume values. Resume with new answer to verify."""

    called: list[str] = []

    def node_a(state: State) -> State:
        called.append("node_a")
        return {"value": ["a"]}

    def ask_human(state: State) -> State:
        called.append("ask_human")
        answer = interrupt("What is your input?")
        return {"value": [f"human:{answer}"]}

    def node_b(state: State) -> State:
        called.append("node_b")
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("ask_human", ask_human)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "ask_human")
        .add_edge("ask_human", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="hello"), config)

    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    called.clear()
    fork_config = graph.update_state(before_ask.config, None, as_node="__copy__")

    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "What is your input?"

    final = graph.invoke(Command(resume="new_answer"), fork_config)
    assert final == {"value": ["a", "human:new_answer", "b"]}


def test_copy_fork_creates_sibling_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """__copy__ creates a checkpoint with source="fork", while regular
    update_state creates one with source="update"."""

    def node_a(state: State) -> State:
        return {"value": ["a"]}

    def node_b(state: State) -> State:
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"value": []}, config)

    history = list(graph.get_state_history(config))
    before_b = next(s for s in history if s.next == ("node_b",))

    # __copy__ fork → source="fork"
    copy_config = graph.update_state(before_b.config, None, as_node="__copy__")
    copy_state = graph.get_state(copy_config)
    assert copy_state.metadata["source"] == "fork"

    # Regular update → source="update"
    regular_config = graph.update_state(before_b.config, {"value": ["x"]})
    regular_state = graph.get_state(regular_config)
    assert regular_state.metadata["source"] == "update"


def test_update_state_with_none_values(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """update_state with None values (not __copy__) goes through the normal
    update path, creating a new checkpoint that re-triggers interrupts."""

    def node_a(state: State) -> State:
        return {"value": ["a"]}

    def ask_human(state: State) -> State:
        answer = interrupt("What is your input?")
        return {"value": [f"human:{answer}"]}

    def node_b(state: State) -> State:
        return {"value": ["b"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("ask_human", ask_human)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "ask_human")
        .add_edge("ask_human", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="hello"), config)

    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    fork_config = graph.update_state(before_ask.config, None)
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "What is your input?"

    fork_state = graph.get_state(fork_config)
    assert fork_state.metadata["source"] == "update"


# ---------------------------------------------------------------------------
# Section 7: Observability (get_state, config access)
# ---------------------------------------------------------------------------


def test_get_state_with_subgraphs_returns_subgraph_state(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """get_state(config, subgraphs=True) returns subgraph state and checkpoint
    config when paused at interrupt."""

    class SubState(TypedDict):
        data: str

    def sub_node(state: SubState) -> SubState:
        interrupt("Continue?")
        return {"data": "processed"}

    subgraph = (
        StateGraph(SubState)
        .add_node("process", sub_node)
        .add_edge(START, "process")
        .compile()
    )

    class ParentState(TypedDict):
        data: str

    graph = (
        StateGraph(ParentState)
        .add_node("sub", subgraph)
        .add_edge(START, "sub")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"data": "input"}, config)

    state = graph.get_state(config, subgraphs=True)

    assert len(state.tasks) > 0
    sub_task = state.tasks[0]
    assert sub_task.state is not None

    sub_config = sub_task.state.config
    assert "configurable" in sub_config
    assert "thread_id" in sub_config["configurable"]


def test_checkpoint_ns_accessible_in_subgraph(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """RunnableConfig exposes checkpoint_ns and thread_id inside subgraph
    nodes."""

    captured_config: dict = {}

    class SubState(TypedDict):
        data: str

    def sub_node(state: SubState, config: RunnableConfig) -> SubState:
        captured_config["checkpoint_ns"] = config["configurable"].get("checkpoint_ns")
        captured_config["thread_id"] = config["configurable"].get("thread_id")
        return {"data": "done"}

    subgraph = (
        StateGraph(SubState)
        .add_node("inner", sub_node)
        .add_edge(START, "inner")
        .compile()
    )

    class ParentState(TypedDict):
        data: str

    graph = (
        StateGraph(ParentState)
        .add_node("outer", subgraph)
        .add_edge(START, "outer")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"data": "test"}, config)

    assert captured_config["checkpoint_ns"] is not None
    assert captured_config["checkpoint_ns"] != ""
    assert captured_config["thread_id"] == "1"


# ---------------------------------------------------------------------------
# Section 8: Stateful vs stateless subgraph state retention on replay
# ---------------------------------------------------------------------------


def test_stateful_subgraph_retains_state_on_parent_replay(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Stateful subgraph (checkpointer=True) remembers accumulated state
    from prior invocations when the parent replays."""
    started: list[tuple[str, dict]] = []
    observed: list[tuple[str, dict]] = []

    class SubState(TypedDict):
        value: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        results: Annotated[list[str], operator.add]

    def parent_node(state: ParentState) -> ParentState:
        return {"results": ["p"]}

    def step_a(state: SubState) -> SubState:
        started.append(("step_a", dict(state)))
        answer = interrupt("question_a")
        observed.append(("step_a", dict(state)))
        return {"value": [f"a:{answer}"]}

    def step_b(state: SubState) -> SubState:
        started.append(("step_b", dict(state)))
        answer = interrupt("question_b")
        observed.append(("step_b", dict(state)))
        return {"value": [f"b:{answer}"]}

    sub = (
        StateGraph(SubState)
        .add_node("step_a", step_a)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "step_b")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(ParentState)
        .add_node("parent_node", parent_node)
        .add_node("sub_node", sub)
        .add_edge(START, "parent_node")
        .add_edge("parent_node", "sub_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # === 1st invocation: answer "a1" and "b1" ===
    graph.invoke({"results": []}, config)  # hits step_a interrupt
    graph.invoke(Command(resume="a1"), config)  # hits step_b interrupt
    graph.invoke(Command(resume="b1"), config)  # completes

    # step_a saw empty state (fresh subgraph)
    assert observed[0] == ("step_a", {"value": []})
    # step_b saw step_a's answer
    assert observed[1] == ("step_b", {"value": ["a:a1"]})

    # === 2nd invocation: answer "a2" and "b2" ===
    observed.clear()
    graph.invoke({"results": []}, config)  # hits step_a interrupt
    graph.invoke(Command(resume="a2"), config)  # hits step_b interrupt
    graph.invoke(Command(resume="b2"), config)  # completes

    # Stateful subgraph retained state from 1st invocation
    assert observed[0] == ("step_a", {"value": ["a:a1", "b:b1"]})
    assert observed[1] == ("step_b", {"value": ["a:a1", "b:b1", "a:a2"]})

    # === Replay from checkpoint before sub_node in 2nd invocation ===
    history = list(graph.get_state_history(config))
    # History is newest-first, so first match = 2nd invocation
    before_sub_2nd = [s for s in history if s.next == ("sub_node",)][0]

    started.clear()
    replay = graph.invoke(None, before_sub_2nd.config)

    assert "__interrupt__" in replay
    # Replay sees 1st invocation's final state, NOT 2nd invocation's
    assert started[0] == ("step_a", {"value": ["a:a1", "b:b1"]})


def test_stateful_subgraph_retains_state_on_parent_fork(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Stateful subgraph (checkpointer=True) remembers accumulated state
    from prior invocations when the parent forks."""
    observed: list[tuple[str, dict]] = []

    class SubState(TypedDict):
        value: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        results: Annotated[list[str], operator.add]

    def parent_node(state: ParentState) -> ParentState:
        return {"results": ["p"]}

    def step_a(state: SubState) -> SubState:
        observed.append(("step_a", dict(state)))
        answer = interrupt("question_a")
        return {"value": [f"a:{answer}"]}

    def step_b(state: SubState) -> SubState:
        observed.append(("step_b", dict(state)))
        answer = interrupt("question_b")
        return {"value": [f"b:{answer}"]}

    sub = (
        StateGraph(SubState)
        .add_node("step_a", step_a)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "step_b")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(ParentState)
        .add_node("parent_node", parent_node)
        .add_node("sub_node", sub)
        .add_edge(START, "parent_node")
        .add_edge("parent_node", "sub_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # === 1st invocation: answer "a1" and "b1" ===
    graph.invoke({"results": []}, config)  # hits step_a interrupt
    graph.invoke(Command(resume="a1"), config)  # hits step_b interrupt
    graph.invoke(Command(resume="b1"), config)  # completes

    # === 2nd invocation: answer "a2" and "b2" ===
    graph.invoke({"results": []}, config)  # hits step_a interrupt
    graph.invoke(Command(resume="a2"), config)  # hits step_b interrupt
    graph.invoke(Command(resume="b2"), config)  # completes

    # === Fork from checkpoint before sub_node in 2nd invocation ===
    history = list(graph.get_state_history(config))
    before_sub_2nd = [s for s in history if s.next == ("sub_node",)][0]
    fork_config = graph.update_state(before_sub_2nd.config, {"results": ["forked"]})

    observed.clear()
    fork_result = graph.invoke(None, fork_config)

    assert "__interrupt__" in fork_result
    # Fork sees 1st invocation's final state, NOT 2nd invocation's
    assert observed[0] == ("step_a", {"value": ["a:a1", "b:b1"]})


def test_stateful_subgraph_loads_state_across_ticks(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """When replaying from a checkpoint before a non-subgraph node, a stateful
    subgraph that runs in a later tick should still load its accumulated state
    from the previous execution.

    Sequence: node_a -> node_b -> sub_node -> node_a -> node_b -> sub_node
    Replay from the 2nd node_a: sub_node in the later tick should see state
    from the end of the 1st sub_node execution.
    """
    observed: list[tuple[str, dict]] = []

    class SubState(TypedDict):
        value: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        results: Annotated[list[str], operator.add]

    def node_a(state: ParentState) -> ParentState:
        return {"results": ["a"]}

    def node_b(state: ParentState) -> ParentState:
        return {"results": ["b"]}

    def sub_step(state: SubState) -> SubState:
        observed.append(("sub_step", dict(state)))
        return {"value": ["s"]}

    sub = (
        StateGraph(SubState)
        .add_node("sub_step", sub_step)
        .add_edge(START, "sub_step")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(ParentState)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_node("sub_node", sub)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .add_edge("node_b", "sub_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # 1st invocation
    graph.invoke({"results": []}, config)
    assert observed[0] == ("sub_step", {"value": []})

    # 2nd invocation
    observed.clear()
    graph.invoke({"results": []}, config)
    assert observed[0] == ("sub_step", {"value": ["s"]})

    # Replay from checkpoint before node_a in 2nd invocation
    history = list(graph.get_state_history(config))
    before_a_2nd = [s for s in history if s.next == ("node_a",)][0]

    observed.clear()
    graph.invoke(None, before_a_2nd.config)

    # sub_node runs in a later tick (after node_a and node_b replay),
    # and should see state from end of 1st sub_node execution
    assert observed[0] == ("sub_step", {"value": ["s"]})


# ---------------------------------------------------------------------------
# Section 8: Append-only checkpoint history (branching / forking)
# ---------------------------------------------------------------------------


def test_replay_creates_branch_preserving_old_checkpoints(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replaying from a mid-run checkpoint creates a new branch of checkpoints
    while the original checkpoint sequence is preserved (append-only).

    Original run (newest first):
      C4  next=()          values=[a, b1, c]     parent=C3
      C3  next=(node_c,)   values=[a, b1]        parent=C2
      C2  next=(node_b,)   values=[a]            parent=C1
      C1  next=(node_a,)   values=[]             parent=C0
      C0  next=(__start__,) values={}             parent=None

    After replay from C2 (newest first):
      C6  next=()          values=[a, b2, c]     parent=C5   <- new branch tip
      C5  next=(node_c,)   values=[a, b2]        parent=C2   <- branches from C2
      C4  next=()          values=[a, b1, c]     parent=C3   <- old branch preserved
      C3  next=(node_c,)   values=[a, b1]        parent=C2
      C2  next=(node_b,)   values=[a]            parent=C1
      C1  next=(node_a,)   values=[]             parent=C0
      C0  next=(__start__,) values={}             parent=None
    """

    call_count = 0

    def node_a(state: State) -> State:
        return {"value": ["a"]}

    def node_b(state: State) -> State:
        nonlocal call_count
        call_count += 1
        return {"value": [f"b{call_count}"]}

    def node_c(state: State) -> State:
        return {"value": ["c"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_node("node_c", node_c)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .add_edge("node_b", "node_c")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"value": []}, config)
    assert result == {"value": ["a", "b1", "c"]}

    # -- Original checkpoint history (newest first) --
    original_history = list(graph.get_state_history(config))
    original_summary = _checkpoint_summary(original_history)
    assert len(original_summary) == 5
    # Verify the shape: next tuples newest->oldest
    assert [s["next"] for s in original_summary] == [
        (),
        ("node_c",),
        ("node_b",),
        ("node_a",),
        ("__start__",),
    ]
    # Verify values at each checkpoint
    assert [s["values"] for s in original_summary] == [
        {"value": ["a", "b1", "c"]},
        {"value": ["a", "b1"]},
        {"value": ["a"]},
        {"value": []},
        {"value": []},
    ]
    original_ids = {s.config["configurable"]["checkpoint_id"] for s in original_history}

    # Find checkpoint before node_b and replay from it
    before_b = next(s for s in original_history if s.next == ("node_b",))
    before_b_id = before_b.config["configurable"]["checkpoint_id"]
    replay_result = graph.invoke(None, before_b.config)
    assert replay_result == {"value": ["a", "b2", "c"]}

    # -- Post-replay checkpoint history (newest first) --
    post_replay_history = list(graph.get_state_history(config))
    post_summary = _checkpoint_summary(post_replay_history)
    assert len(post_summary) == 7  # 5 original + 2 new branch checkpoints

    # Verify the full shape after replay
    assert [s["next"] for s in post_summary] == [
        (),  # new branch tip (C6)
        ("node_c",),  # new branch (C5)
        (),  # old branch tip (C4)
        ("node_c",),  # old (C3)
        ("node_b",),  # branch point (C2)
        ("node_a",),  # old (C1)
        ("__start__",),  # old (C0)
    ]
    assert [s["values"] for s in post_summary] == [
        {"value": ["a", "b2", "c"]},  # new branch tip
        {"value": ["a", "b2"]},  # new: node_b re-ran with call_count=2
        {"value": ["a", "b1", "c"]},  # old branch tip preserved
        {"value": ["a", "b1"]},  # old
        {"value": ["a"]},  # branch point
        {"value": []},  # old
        {"value": []},  # old
    ]

    # All original checkpoint IDs still exist (append-only)
    post_ids = {s.config["configurable"]["checkpoint_id"] for s in post_replay_history}
    assert original_ids.issubset(post_ids)

    # New branch's oldest checkpoint parent is the branch point
    new_checkpoints = [
        s
        for s in post_replay_history
        if s.config["configurable"]["checkpoint_id"] not in original_ids
    ]
    oldest_new = sorted(new_checkpoints, key=lambda s: s.created_at)[0]
    assert oldest_new.parent_config is not None
    assert oldest_new.parent_config["configurable"]["checkpoint_id"] == before_b_id

    # get_state returns the new branch tip
    latest = graph.get_state(config)
    assert latest.values == {"value": ["a", "b2", "c"]}
    assert latest.config["configurable"]["checkpoint_id"] not in original_ids


def test_replay_creates_branch_in_subgraph(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replaying a graph with a subgraph from a mid-run checkpoint creates a
    new branch while preserving the original checkpoint sequence.

    The subgraph re-executes on the new branch and the old checkpoints
    (including sub-checkpoints) remain in the history.
    """

    sub_call_count = 0

    class SubState(TypedDict):
        sub_value: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        value: Annotated[list[str], operator.add]
        sub_value: Annotated[list[str], operator.add]

    def parent_start(state: ParentState) -> ParentState:
        return {"value": ["p_start"]}

    def sub_step(state: SubState) -> SubState:
        nonlocal sub_call_count
        sub_call_count += 1
        return {"sub_value": [f"sub{sub_call_count}"]}

    def parent_end(state: ParentState) -> ParentState:
        return {"value": ["p_end"]}

    sub = (
        StateGraph(SubState)
        .add_node("sub_step", sub_step)
        .add_edge(START, "sub_step")
        .compile()
    )

    graph = (
        StateGraph(ParentState)
        .add_node("parent_start", parent_start)
        .add_node("sub_graph", sub)
        .add_node("parent_end", parent_end)
        .add_edge(START, "parent_start")
        .add_edge("parent_start", "sub_graph")
        .add_edge("sub_graph", "parent_end")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"value": [], "sub_value": []}, config)
    assert result == {"value": ["p_start", "p_end"], "sub_value": ["sub1"]}

    # Capture original checkpoint IDs
    original_history = list(graph.get_state_history(config))
    original_ids = {s.config["configurable"]["checkpoint_id"] for s in original_history}

    # Find checkpoint before sub_graph
    before_sub = next(s for s in original_history if s.next == ("sub_graph",))
    before_sub_id = before_sub.config["configurable"]["checkpoint_id"]

    # Replay from before sub_graph
    replay_result = graph.invoke(None, before_sub.config)
    assert replay_result == {"value": ["p_start", "p_end"], "sub_value": ["sub2"]}

    # Get full history after replay
    post_replay_history = list(graph.get_state_history(config))
    post_replay_ids = {
        s.config["configurable"]["checkpoint_id"] for s in post_replay_history
    }

    # All original checkpoint IDs still exist (append-only)
    assert original_ids.issubset(post_replay_ids)

    # New checkpoints were added (the branch)
    new_ids = post_replay_ids - original_ids
    assert len(new_ids) >= 2  # sub_graph + parent_end at minimum

    # The oldest new checkpoint's parent is the checkpoint we replayed from
    new_checkpoints = [
        s
        for s in post_replay_history
        if s.config["configurable"]["checkpoint_id"] in new_ids
    ]
    oldest_new = sorted(new_checkpoints, key=lambda s: s.created_at)[0]
    assert oldest_new.parent_config is not None
    assert oldest_new.parent_config["configurable"]["checkpoint_id"] == before_sub_id

    # get_state returns the new branch tip
    latest = graph.get_state(config)
    assert latest.config["configurable"]["checkpoint_id"] in new_ids
    assert latest.values == {"value": ["p_start", "p_end"], "sub_value": ["sub2"]}


def test_fork_creates_branch_preserving_old_checkpoints(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Forking (update_state + invoke) from a mid-run checkpoint creates a new
    branch of checkpoints while the original sequence is preserved.

    Original run (newest first):
      C4  next=()          values=[a, b1, c]     parent=C3
      C3  next=(node_c,)   values=[a, b1]        parent=C2
      C2  next=(node_b,)   values=[a]            parent=C1
      C1  next=(node_a,)   values=[]             parent=C0
      C0  next=(__start__,) values={}             parent=None

    After fork from C2 with update {"value": ["x"]} (newest first):
      C7  next=()          values=[a, x, b2, c]  parent=C6
      C6  next=(node_c,)   values=[a, x, b2]     parent=C5
      C5  next=(node_b,)   values=[a, x]         parent=C2   <- fork checkpoint
      C4  next=()          values=[a, b1, c]     parent=C3   <- old branch preserved
      C3  next=(node_c,)   values=[a, b1]        parent=C2
      C2  next=(node_b,)   values=[a]            parent=C1   <- fork point
      C1  next=(node_a,)   values=[]             parent=C0
      C0  next=(__start__,) values={}             parent=None
    """

    call_count = 0

    def node_a(state: State) -> State:
        return {"value": ["a"]}

    def node_b(state: State) -> State:
        nonlocal call_count
        call_count += 1
        return {"value": [f"b{call_count}"]}

    def node_c(state: State) -> State:
        return {"value": ["c"]}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_node("node_c", node_c)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .add_edge("node_b", "node_c")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"value": []}, config)
    assert result == {"value": ["a", "b1", "c"]}

    # -- Original checkpoint history (newest first) --
    original_history = list(graph.get_state_history(config))
    original_summary = _checkpoint_summary(original_history)
    assert len(original_summary) == 5
    assert [s["next"] for s in original_summary] == [
        (),
        ("node_c",),
        ("node_b",),
        ("node_a",),
        ("__start__",),
    ]
    assert [s["values"] for s in original_summary] == [
        {"value": ["a", "b1", "c"]},
        {"value": ["a", "b1"]},
        {"value": ["a"]},
        {"value": []},
        {"value": []},
    ]
    original_ids = {s.config["configurable"]["checkpoint_id"] for s in original_history}

    # Fork from before node_b with modified state
    before_b = next(s for s in original_history if s.next == ("node_b",))
    before_b_id = before_b.config["configurable"]["checkpoint_id"]
    fork_config = graph.update_state(before_b.config, {"value": ["x"]})
    fork_result = graph.invoke(None, fork_config)
    assert fork_result == {"value": ["a", "x", "b2", "c"]}

    # -- Post-fork checkpoint history (newest first) --
    post_fork_history = list(graph.get_state_history(config))
    post_summary = _checkpoint_summary(post_fork_history)
    # 5 original + 1 fork checkpoint (update_state) + 2 new nodes (node_b, node_c)
    assert len(post_summary) == 8

    assert [s["next"] for s in post_summary] == [
        (),  # new branch tip (C7)
        ("node_c",),  # new branch (C6)
        ("node_b",),  # fork checkpoint from update_state (C5)
        (),  # old branch tip (C4)
        ("node_c",),  # old (C3)
        ("node_b",),  # fork point (C2)
        ("node_a",),  # old (C1)
        ("__start__",),  # old (C0)
    ]
    assert [s["values"] for s in post_summary] == [
        {"value": ["a", "x", "b2", "c"]},  # new branch tip
        {"value": ["a", "x", "b2"]},  # new: node_b re-ran
        {"value": ["a", "x"]},  # fork: state updated with "x"
        {"value": ["a", "b1", "c"]},  # old branch tip preserved
        {"value": ["a", "b1"]},  # old
        {"value": ["a"]},  # fork point
        {"value": []},  # old
        {"value": []},  # old
    ]

    # All original checkpoint IDs still exist (append-only)
    post_ids = {s.config["configurable"]["checkpoint_id"] for s in post_fork_history}
    assert original_ids.issubset(post_ids)

    # Fork checkpoint's parent is the branch point
    new_checkpoints = [
        s
        for s in post_fork_history
        if s.config["configurable"]["checkpoint_id"] not in original_ids
    ]
    oldest_new = sorted(new_checkpoints, key=lambda s: s.created_at)[0]
    assert oldest_new.parent_config is not None
    assert oldest_new.parent_config["configurable"]["checkpoint_id"] == before_b_id

    # get_state returns the new branch tip
    latest = graph.get_state(config)
    assert latest.values == {"value": ["a", "x", "b2", "c"]}
    assert latest.config["configurable"]["checkpoint_id"] not in original_ids


def test_fork_creates_branch_in_subgraph(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Forking a graph with a subgraph from a mid-run checkpoint creates a new
    branch while preserving the original checkpoint sequence.

    The subgraph re-executes on the new branch and the old checkpoints remain.
    """

    sub_call_count = 0

    class SubState(TypedDict):
        sub_value: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        value: Annotated[list[str], operator.add]
        sub_value: Annotated[list[str], operator.add]

    def parent_start(state: ParentState) -> ParentState:
        return {"value": ["p_start"]}

    def sub_step(state: SubState) -> SubState:
        nonlocal sub_call_count
        sub_call_count += 1
        return {"sub_value": [f"sub{sub_call_count}"]}

    def parent_end(state: ParentState) -> ParentState:
        return {"value": ["p_end"]}

    sub = (
        StateGraph(SubState)
        .add_node("sub_step", sub_step)
        .add_edge(START, "sub_step")
        .compile()
    )

    graph = (
        StateGraph(ParentState)
        .add_node("parent_start", parent_start)
        .add_node("sub_graph", sub)
        .add_node("parent_end", parent_end)
        .add_edge(START, "parent_start")
        .add_edge("parent_start", "sub_graph")
        .add_edge("sub_graph", "parent_end")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"value": [], "sub_value": []}, config)
    assert result == {"value": ["p_start", "p_end"], "sub_value": ["sub1"]}

    # Capture original checkpoint IDs
    original_history = list(graph.get_state_history(config))
    original_ids = {s.config["configurable"]["checkpoint_id"] for s in original_history}

    # Find checkpoint before sub_graph and fork with modified state
    before_sub = next(s for s in original_history if s.next == ("sub_graph",))
    before_sub_id = before_sub.config["configurable"]["checkpoint_id"]
    fork_config = graph.update_state(before_sub.config, {"value": ["extra"]})
    fork_result = graph.invoke(None, fork_config)
    assert fork_result == {
        "value": ["p_start", "extra", "p_end"],
        "sub_value": ["sub2"],
    }

    # Get full history after fork
    post_fork_history = list(graph.get_state_history(config))
    post_fork_ids = {
        s.config["configurable"]["checkpoint_id"] for s in post_fork_history
    }

    # All original checkpoint IDs still exist (append-only)
    assert original_ids.issubset(post_fork_ids)

    # New checkpoints were added (the branch)
    new_ids = post_fork_ids - original_ids
    assert len(new_ids) >= 3  # fork checkpoint + sub_graph + parent_end

    # The oldest new checkpoint's parent is the checkpoint we forked from
    new_checkpoints = [
        s
        for s in post_fork_history
        if s.config["configurable"]["checkpoint_id"] in new_ids
    ]
    oldest_new = sorted(new_checkpoints, key=lambda s: s.created_at)[0]
    assert oldest_new.parent_config is not None
    assert oldest_new.parent_config["configurable"]["checkpoint_id"] == before_sub_id

    # get_state returns the new branch tip
    latest = graph.get_state(config)
    assert latest.config["configurable"]["checkpoint_id"] in new_ids
    assert latest.values == {
        "value": ["p_start", "extra", "p_end"],
        "sub_value": ["sub2"],
    }


def test_stateless_subgraph_starts_fresh_on_parent_replay(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Stateless subgraph (no checkpointer) always starts with empty state,
    even after prior invocations have completed."""
    started: list[tuple[str, dict]] = []
    observed: list[tuple[str, dict]] = []

    class SubState(TypedDict):
        value: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        results: Annotated[list[str], operator.add]

    def parent_node(state: ParentState) -> ParentState:
        return {"results": ["p"]}

    def step_a(state: SubState) -> SubState:
        started.append(("step_a", dict(state)))
        answer = interrupt("question_a")
        observed.append(("step_a", dict(state)))
        return {"value": [f"a:{answer}"]}

    def step_b(state: SubState) -> SubState:
        started.append(("step_b", dict(state)))
        answer = interrupt("question_b")
        observed.append(("step_b", dict(state)))
        return {"value": [f"b:{answer}"]}

    sub = (
        StateGraph(SubState)
        .add_node("step_a", step_a)
        .add_node("step_b", step_b)
        .add_edge(START, "step_a")
        .add_edge("step_a", "step_b")
        .compile()  # no checkpointer — stateless
    )

    graph = (
        StateGraph(ParentState)
        .add_node("parent_node", parent_node)
        .add_node("sub_node", sub)
        .add_edge(START, "parent_node")
        .add_edge("parent_node", "sub_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # === 1st invocation: answer "a1" and "b1" ===
    graph.invoke({"results": []}, config)  # hits step_a interrupt
    graph.invoke(Command(resume="a1"), config)  # hits step_b interrupt
    graph.invoke(Command(resume="b1"), config)  # completes

    # step_a saw empty state, step_b saw only step_a's answer
    assert observed[0] == ("step_a", {"value": []})
    assert observed[1] == ("step_b", {"value": ["a:a1"]})

    # === 2nd invocation: answer "a2" and "b2" ===
    observed.clear()
    graph.invoke({"results": []}, config)  # hits step_a interrupt
    graph.invoke(Command(resume="a2"), config)  # hits step_b interrupt
    graph.invoke(Command(resume="b2"), config)  # completes

    # Stateless subgraph starts fresh — no memory of 1st invocation
    assert observed[0] == ("step_a", {"value": []})
    assert observed[1] == ("step_b", {"value": ["a:a2"]})

    # === Replay from checkpoint before sub_node in 2nd invocation ===
    history = list(graph.get_state_history(config))
    before_sub_2nd = [s for s in history if s.next == ("sub_node",)][0]

    started.clear()
    replay = graph.invoke(None, before_sub_2nd.config)

    assert "__interrupt__" in replay
    # Stateless subgraph starts completely fresh on replay
    assert started[0] == ("step_a", {"value": []})


def test_stateful_subgraph_loads_latest_after_replay(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """After replaying a parent checkpoint, a subsequent (3rd) invocation should
    load the subgraph state created by the replay — not the state from the
    checkpoint we replayed from."""
    observed: list[tuple[str, dict]] = []

    class SubState(TypedDict):
        value: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        results: Annotated[list[str], operator.add]

    def parent_node(state: ParentState) -> ParentState:
        return {"results": ["p"]}

    def sub_step(state: SubState) -> SubState:
        observed.append(("sub_step", dict(state)))
        return {"value": ["s"]}

    sub = (
        StateGraph(SubState)
        .add_node("sub_step", sub_step)
        .add_edge(START, "sub_step")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(ParentState)
        .add_node("parent_node", parent_node)
        .add_node("sub_node", sub)
        .add_edge(START, "parent_node")
        .add_edge("parent_node", "sub_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # 1st invocation — subgraph starts fresh
    graph.invoke({"results": []}, config)
    assert observed[-1] == ("sub_step", {"value": []})

    # 2nd invocation — subgraph sees state from 1st
    graph.invoke({"results": []}, config)
    assert observed[-1] == ("sub_step", {"value": ["s"]})

    # Replay from checkpoint before parent_node in 2nd invocation
    history = list(graph.get_state_history(config))
    before_parent_2nd = [s for s in history if s.next == ("parent_node",)][0]

    observed.clear()
    graph.invoke(None, before_parent_2nd.config)
    # Replay should load subgraph state from end of 1st invocation
    assert observed[0] == ("sub_step", {"value": ["s"]})

    # 3rd invocation — should see state from the replay (2 × "s"), not from
    # the checkpoint we replayed from (1 × "s")
    observed.clear()
    graph.invoke({"results": []}, config)
    assert observed[0] == ("sub_step", {"value": ["s", "s"]})


def test_three_level_nested_subgraph_loads_state_on_replay(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Three levels of nesting: parent -> mid -> inner.
    Replaying from the parent should load correct state at all levels."""
    observed: list[tuple[str, dict]] = []

    class InnerState(TypedDict):
        inner_trail: Annotated[list[str], operator.add]

    class MidState(TypedDict):
        mid_trail: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        results: Annotated[list[str], operator.add]

    def inner_step(state: InnerState) -> InnerState:
        observed.append(("inner_step", dict(state)))
        return {"inner_trail": ["inner"]}

    def mid_step(state: MidState) -> MidState:
        observed.append(("mid_step", dict(state)))
        return {"mid_trail": ["mid"]}

    def parent_step(state: ParentState) -> ParentState:
        return {"results": ["p"]}

    inner = (
        StateGraph(InnerState)
        .add_node("inner_step", inner_step)
        .add_edge(START, "inner_step")
        .compile(checkpointer=True)
    )

    mid = (
        StateGraph(MidState)
        .add_node("mid_step", mid_step)
        .add_node("inner_node", inner)
        .add_edge(START, "mid_step")
        .add_edge("mid_step", "inner_node")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(ParentState)
        .add_node("parent_step", parent_step)
        .add_node("mid_node", mid)
        .add_edge(START, "parent_step")
        .add_edge("parent_step", "mid_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # 1st invocation — everything starts fresh
    graph.invoke({"results": []}, config)
    assert observed == [
        ("mid_step", {"mid_trail": []}),
        ("inner_step", {"inner_trail": []}),
    ]

    # 2nd invocation — both levels see accumulated state
    observed.clear()
    graph.invoke({"results": []}, config)
    assert observed == [
        ("mid_step", {"mid_trail": ["mid"]}),
        ("inner_step", {"inner_trail": ["inner"]}),
    ]

    # Replay from checkpoint before parent_step in 2nd invocation
    history = list(graph.get_state_history(config))
    before_parent_2nd = [s for s in history if s.next == ("parent_step",)][0]

    observed.clear()
    graph.invoke(None, before_parent_2nd.config)

    # Both mid and inner should load state from end of 1st invocation
    assert observed == [
        ("mid_step", {"mid_trail": ["mid"]}),
        ("inner_step", {"inner_trail": ["inner"]}),
    ]

    # 3rd invocation — sees state from replay, not from the replayed checkpoint
    observed.clear()
    graph.invoke({"results": []}, config)
    assert observed == [
        ("mid_step", {"mid_trail": ["mid", "mid"]}),
        ("inner_step", {"inner_trail": ["inner", "inner"]}),
    ]


def test_three_level_nested_subgraph_loads_state_on_fork(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Three levels of nesting with fork instead of replay."""
    observed: list[tuple[str, dict]] = []

    class InnerState(TypedDict):
        inner_trail: Annotated[list[str], operator.add]

    class MidState(TypedDict):
        mid_trail: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        results: Annotated[list[str], operator.add]

    def inner_step(state: InnerState) -> InnerState:
        observed.append(("inner_step", dict(state)))
        return {"inner_trail": ["inner"]}

    def mid_step(state: MidState) -> MidState:
        observed.append(("mid_step", dict(state)))
        return {"mid_trail": ["mid"]}

    def parent_step(state: ParentState) -> ParentState:
        return {"results": ["p"]}

    inner = (
        StateGraph(InnerState)
        .add_node("inner_step", inner_step)
        .add_edge(START, "inner_step")
        .compile(checkpointer=True)
    )

    mid = (
        StateGraph(MidState)
        .add_node("mid_step", mid_step)
        .add_node("inner_node", inner)
        .add_edge(START, "mid_step")
        .add_edge("mid_step", "inner_node")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(ParentState)
        .add_node("parent_step", parent_step)
        .add_node("mid_node", mid)
        .add_edge(START, "parent_step")
        .add_edge("parent_step", "mid_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # 1st invocation
    graph.invoke({"results": []}, config)

    # 2nd invocation
    graph.invoke({"results": []}, config)

    # Fork from checkpoint before parent_step in 2nd invocation
    history = list(graph.get_state_history(config))
    before_parent_2nd = [s for s in history if s.next == ("parent_step",)][0]
    fork_config = graph.update_state(before_parent_2nd.config, {"results": ["forked"]})

    observed.clear()
    graph.invoke(None, fork_config)

    # Both mid and inner should load state from end of 1st invocation
    assert observed == [
        ("mid_step", {"mid_trail": ["mid"]}),
        ("inner_step", {"inner_trail": ["inner"]}),
    ]


def test_replay_from_first_invocation_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replaying from the 1st invocation's checkpoint should load the subgraph
    state from before that invocation (i.e. empty)."""
    observed: list[tuple[str, dict]] = []

    class SubState(TypedDict):
        value: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        results: Annotated[list[str], operator.add]

    def parent_node(state: ParentState) -> ParentState:
        return {"results": ["p"]}

    def sub_step(state: SubState) -> SubState:
        observed.append(("sub_step", dict(state)))
        return {"value": ["s"]}

    sub = (
        StateGraph(SubState)
        .add_node("sub_step", sub_step)
        .add_edge(START, "sub_step")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(ParentState)
        .add_node("parent_node", parent_node)
        .add_node("sub_node", sub)
        .add_edge(START, "parent_node")
        .add_edge("parent_node", "sub_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # Run twice so subgraph accumulates state
    graph.invoke({"results": []}, config)
    graph.invoke({"results": []}, config)

    # Replay from before sub_node in 1st invocation (furthest back)
    history = list(graph.get_state_history(config))
    before_sub_1st = [s for s in history if s.next == ("sub_node",)][-1]

    observed.clear()
    graph.invoke(None, before_sub_1st.config)
    # Should see empty state — no prior subgraph checkpoints exist
    assert observed[0] == ("sub_step", {"value": []})


def test_parallel_subgraph_nodes_load_correct_state_on_replay(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Two sibling subgraph nodes that fan out from a common predecessor.
    Each should independently load its own historical checkpoint on replay."""
    observed: list[tuple[str, dict]] = []

    class SubStateA(TypedDict):
        a_trail: Annotated[list[str], operator.add]

    class SubStateB(TypedDict):
        b_trail: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        results: Annotated[list[str], operator.add]

    def parent_step(state: ParentState) -> ParentState:
        return {"results": ["p"]}

    def sub_a_step(state: SubStateA) -> SubStateA:
        observed.append(("sub_a", dict(state)))
        return {"a_trail": ["a"]}

    def sub_b_step(state: SubStateB) -> SubStateB:
        observed.append(("sub_b", dict(state)))
        return {"b_trail": ["b"]}

    sub_a = (
        StateGraph(SubStateA)
        .add_node("sub_a_step", sub_a_step)
        .add_edge(START, "sub_a_step")
        .compile(checkpointer=True)
    )

    sub_b = (
        StateGraph(SubStateB)
        .add_node("sub_b_step", sub_b_step)
        .add_edge(START, "sub_b_step")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(ParentState)
        .add_node("parent_step", parent_step)
        .add_node("sub_a_node", sub_a)
        .add_node("sub_b_node", sub_b)
        .add_edge(START, "parent_step")
        # Fan out: both subgraphs run in parallel after parent_step
        .add_edge("parent_step", "sub_a_node")
        .add_edge("parent_step", "sub_b_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # 1st invocation — both subgraphs start fresh
    graph.invoke({"results": []}, config)
    a_obs = [o for o in observed if o[0] == "sub_a"]
    b_obs = [o for o in observed if o[0] == "sub_b"]
    assert a_obs[0] == ("sub_a", {"a_trail": []})
    assert b_obs[0] == ("sub_b", {"b_trail": []})

    # 2nd invocation — both see accumulated state
    observed.clear()
    graph.invoke({"results": []}, config)
    a_obs = [o for o in observed if o[0] == "sub_a"]
    b_obs = [o for o in observed if o[0] == "sub_b"]
    assert a_obs[0] == ("sub_a", {"a_trail": ["a"]})
    assert b_obs[0] == ("sub_b", {"b_trail": ["b"]})

    # Replay from checkpoint before parent_step in 2nd invocation
    history = list(graph.get_state_history(config))
    before_parent_2nd = [s for s in history if s.next == ("parent_step",)][0]

    observed.clear()
    graph.invoke(None, before_parent_2nd.config)

    # Each subgraph should independently load state from end of 1st invocation
    a_obs = [o for o in observed if o[0] == "sub_a"]
    b_obs = [o for o in observed if o[0] == "sub_b"]
    assert a_obs[0] == ("sub_a", {"a_trail": ["a"]})
    assert b_obs[0] == ("sub_b", {"b_trail": ["b"]})

    # 3rd invocation — sees state from replay
    observed.clear()
    graph.invoke({"results": []}, config)
    a_obs = [o for o in observed if o[0] == "sub_a"]
    b_obs = [o for o in observed if o[0] == "sub_b"]
    assert a_obs[0] == ("sub_a", {"a_trail": ["a", "a"]})
    assert b_obs[0] == ("sub_b", {"b_trail": ["b", "b"]})


def test_subgraph_called_in_loop_loads_state_on_replay(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Parent calls a subgraph node multiple times per invocation via a
    conditional loop. Replay should restore the subgraph's accumulated state
    from the correct point in the parent's timeline."""
    observed: list[tuple[str, dict]] = []

    class SubState(TypedDict):
        sub_trail: Annotated[list[str], operator.add]

    class ParentState(TypedDict):
        counter: int
        results: Annotated[list[str], operator.add]

    def inc(state: ParentState) -> ParentState:
        return {"counter": state["counter"] + 1, "results": [f"inc:{state['counter']}"]}

    def sub_step(state: SubState) -> SubState:
        observed.append(("sub_step", dict(state)))
        return {"sub_trail": ["s"]}

    def should_loop(state: ParentState) -> str:
        return "inc" if state["counter"] < 2 else "__end__"

    sub = (
        StateGraph(SubState)
        .add_node("sub_step", sub_step)
        .add_edge(START, "sub_step")
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(ParentState)
        .add_node("inc", inc)
        .add_node("sub_node", sub)
        .add_edge(START, "inc")
        .add_edge("inc", "sub_node")
        .add_conditional_edges("sub_node", should_loop, ["inc", "__end__"])
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # 1st invocation: loop runs inc->sub->inc->sub->end
    # counter goes 0->1->2, subgraph called twice
    graph.invoke({"counter": 0, "results": []}, config)
    assert len(observed) == 2
    assert observed[0] == ("sub_step", {"sub_trail": []})
    assert observed[1] == ("sub_step", {"sub_trail": ["s"]})

    # 2nd invocation: loop runs again, subgraph sees accumulated state
    observed.clear()
    graph.invoke({"counter": 0, "results": []}, config)
    assert len(observed) == 2
    assert observed[0] == ("sub_step", {"sub_trail": ["s", "s"]})
    assert observed[1] == ("sub_step", {"sub_trail": ["s", "s", "s"]})

    # Replay from the START of the 2nd invocation's loop (counter=0).
    # History has two inc checkpoints with counter=0: 2nd invocation's (newer)
    # and 1st invocation's (older). Pick the newer one.
    history = list(graph.get_state_history(config))
    start_of_loop_2nd = [
        s for s in history if s.next == ("inc",) and s.values["counter"] == 0
    ][0]

    observed.clear()
    graph.invoke(None, start_of_loop_2nd.config)

    # Full loop re-runs (2 sub calls). Subgraph loads state from end of
    # 1st invocation (2 × "s"), same as the original 2nd invocation.
    assert len(observed) == 2
    assert observed[0] == ("sub_step", {"sub_trail": ["s", "s"]})
    assert observed[1] == ("sub_step", {"sub_trail": ["s", "s", "s"]})

    # Also test replay from MID-loop (counter=1) in the 2nd invocation.
    # Only one loop iteration remains, and the subgraph should load state
    # that includes the first loop iteration of the 2nd invocation (3 × "s").
    mid_loop_2nd = [
        s for s in history if s.next == ("inc",) and s.values["counter"] == 1
    ][0]

    observed.clear()
    graph.invoke(None, mid_loop_2nd.config)

    assert len(observed) == 1
    assert observed[0] == ("sub_step", {"sub_trail": ["s", "s", "s"]})
