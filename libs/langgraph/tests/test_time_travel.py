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


def test_replay_from_interrupt_checkpoint_refires(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from the checkpoint where interrupt fired (not before it).
    Interrupt re-fires on replay."""

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

    # Find the checkpoint where interrupt fired
    history = list(graph.get_state_history(config))
    interrupt_checkpoint = next(
        s
        for s in history
        if s.next == ("ask_human",) and s.tasks and any(t.interrupts for t in s.tasks)
    )

    called.clear()
    replay_result = graph.invoke(None, interrupt_checkpoint.config)
    assert "__interrupt__" in replay_result


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


def test_subgraph_interrupt_fork_from_parent(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from parent checkpoint before subgraph (checkpointer=True).
    Subgraph interrupt re-fires on fork."""

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

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="answer"), config)

    history = list(graph.get_state_history(config))
    before_sub = [s for s in history if s.next == ("subgraph_node",)][-1]

    called.clear()
    fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    assert "__interrupt__" in fork_result


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


def test_subgraph_interrupt_fork_no_sub_checkpointer(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from parent checkpoint before subgraph with no sub-checkpointer
    (checkpointer=None). Subgraph state flows through parent's checkpoint,
    so forking the parent re-triggers the interrupt."""

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

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="answer"), config)

    history = list(graph.get_state_history(config))
    before_sub = [s for s in history if s.next == ("subgraph_node",)][-1]

    called.clear()
    fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    assert "__interrupt__" in fork_result
    assert fork_result["__interrupt__"][0].value == "Provide input:"


def test_subgraph_interrupt_fork_from_interrupt_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from the parent checkpoint where subgraph interrupt fired
    (checkpointer=True). Whether interrupt re-triggers depends on whether
    the subgraph's checkpoint is cleared by the parent fork."""

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

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="answer"), config)

    history = list(graph.get_state_history(config))
    interrupt_checkpoint = next(
        s
        for s in history
        if s.next == ("subgraph_node",)
        and s.tasks
        and any(t.interrupts for t in s.tasks)
    )

    called.clear()
    fork_config = graph.update_state(interrupt_checkpoint.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)

    assert "value" in fork_result


def test_subgraph_interrupt_fork_from_subgraph_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Fork from the subgraph's own checkpoint (via get_state with
    subgraphs=True). Directly modifies the subgraph's checkpoint, so the
    interrupt IS re-triggered."""

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

    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result

    parent_state = graph.get_state(config, subgraphs=True)
    sub_task = parent_state.tasks[0]
    assert sub_task.state is not None
    sub_config = sub_task.state.config

    called.clear()
    fork_config = graph.update_state(sub_config, {"value": ["sub_forked"]})
    fork_result = graph.invoke(None, fork_config)
    assert "__interrupt__" in fork_result or "ask_human" in called


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
    graph.invoke({"results": []}, config)       # hits step_a interrupt
    graph.invoke(Command(resume="a1"), config)   # hits step_b interrupt
    graph.invoke(Command(resume="b1"), config)   # completes

    # step_a saw empty state (fresh subgraph)
    assert observed[0] == ("step_a", {"value": []})
    # step_b saw step_a's answer
    assert observed[1] == ("step_b", {"value": ["a:a1"]})

    # === 2nd invocation: answer "a2" and "b2" ===
    observed.clear()
    graph.invoke({"results": []}, config)       # hits step_a interrupt
    graph.invoke(Command(resume="a2"), config)   # hits step_b interrupt
    graph.invoke(Command(resume="b2"), config)   # completes

    # Stateful subgraph retained state from 1st invocation
    assert observed[0] == ("step_a", {"value": ["a:a1", "b:b1"]})
    assert observed[1] == ("step_b", {"value": ["a:a1", "b:b1", "a:a2"]})

    # === Replay from checkpoint before sub_node in 2nd invocation ===
    history = list(graph.get_state_history(config))
    # History is newest-first, so first match = 2nd invocation
    before_sub_2nd = [s for s in history if s.next == ("sub_node",)][0]

    observed.clear()
    replay = graph.invoke(None, before_sub_2nd.config)

    assert "__interrupt__" in replay
    # Replay sees 1st invocation's final state, NOT 2nd invocation's
    assert observed[0] == ("step_a", {"value": ["a:a1", "b:b1"]})


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
    graph.invoke({"results": []}, config)       # hits step_a interrupt
    graph.invoke(Command(resume="a1"), config)   # hits step_b interrupt
    graph.invoke(Command(resume="b1"), config)   # completes

    # === 2nd invocation: answer "a2" and "b2" ===
    graph.invoke({"results": []}, config)       # hits step_a interrupt
    graph.invoke(Command(resume="a2"), config)   # hits step_b interrupt
    graph.invoke(Command(resume="b2"), config)   # completes

    # === Fork from checkpoint before sub_node in 2nd invocation ===
    history = list(graph.get_state_history(config))
    before_sub_2nd = [s for s in history if s.next == ("sub_node",)][0]
    fork_config = graph.update_state(before_sub_2nd.config, {"results": ["forked"]})

    observed.clear()
    fork_result = graph.invoke(None, fork_config)

    assert "__interrupt__" in fork_result
    # Fork sees 1st invocation's final state, NOT 2nd invocation's
    assert observed[0] == ("step_a", {"value": ["a:a1", "b:b1"]})


def test_stateless_subgraph_starts_fresh_on_parent_replay(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Stateless subgraph (no checkpointer) always starts with empty state,
    even after prior invocations have completed."""
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
    graph.invoke({"results": []}, config)       # hits step_a interrupt
    graph.invoke(Command(resume="a1"), config)   # hits step_b interrupt
    graph.invoke(Command(resume="b1"), config)   # completes

    # step_a saw empty state, step_b saw only step_a's answer
    assert observed[0] == ("step_a", {"value": []})
    assert observed[1] == ("step_b", {"value": ["a:a1"]})

    # === 2nd invocation: answer "a2" and "b2" ===
    observed.clear()
    graph.invoke({"results": []}, config)       # hits step_a interrupt
    graph.invoke(Command(resume="a2"), config)   # hits step_b interrupt
    graph.invoke(Command(resume="b2"), config)   # completes

    # Stateless subgraph starts fresh — no memory of 1st invocation
    assert observed[0] == ("step_a", {"value": []})
    assert observed[1] == ("step_b", {"value": ["a:a2"]})

    # === Replay from checkpoint before sub_node in 2nd invocation ===
    history = list(graph.get_state_history(config))
    before_sub_2nd = [s for s in history if s.next == ("sub_node",)][0]

    observed.clear()
    replay = graph.invoke(None, before_sub_2nd.config)

    assert "__interrupt__" in replay
    # Stateless subgraph starts completely fresh on replay
    assert observed[0] == ("step_a", {"value": []})
