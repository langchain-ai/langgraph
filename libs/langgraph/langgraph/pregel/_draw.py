from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.graph import Graph, Node

from langgraph._internal._constants import CONF, CONFIG_KEY_SEND, INPUT
from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.managed.base import ManagedValueSpec
from langgraph.pregel._algo import (
    PregelTaskWrites,
    apply_writes,
    increment,
    prepare_next_tasks,
)
from langgraph.pregel._checkpoint import channels_from_checkpoint, empty_checkpoint
from langgraph.pregel._io import map_input
from langgraph.pregel._read import PregelNode
from langgraph.pregel._write import ChannelWrite
from langgraph.types import All, Checkpointer


def draw_graph(
    config: RunnableConfig,
    *,
    nodes: dict[str, PregelNode],
    specs: dict[str, BaseChannel | ManagedValueSpec],
    input_channels: str | Sequence[str],
    interrupt_after_nodes: All | Sequence[str],
    interrupt_before_nodes: All | Sequence[str],
    trigger_to_nodes: Mapping[str, Sequence[str]],
    checkpointer: Checkpointer,
    subgraphs: dict[str, Graph],
    limit: int = 250,
) -> Graph:
    """Get the graph for this Pregel instance.

    Args:
        config: The configuration to use for the graph.
        subgraphs: The subgraphs to include in the graph.
        checkpointer: The checkpointer to use for the graph.

    Returns:
        The graph for this Pregel instance.
    """
    # (src, dest, is_conditional, label)
    edges: set[tuple[str, str, bool, str | None]] = set()

    step = -1
    checkpoint = empty_checkpoint()
    get_next_version = (
        checkpointer.get_next_version
        if isinstance(checkpointer, BaseCheckpointSaver)
        else increment
    )
    channels, managed = channels_from_checkpoint(
        specs,
        checkpoint,
    )
    static_seen: set[Any] = set()
    sources: dict[str, set[tuple[str, bool, str | None]]] = {}
    step_sources: dict[str, set[tuple[str, bool, str | None]]] = {}
    # remove node mappers
    nodes = {
        k: v.copy(update={"mapper": None}) if v.mapper is not None else v
        for k, v in nodes.items()
    }
    # apply input writes
    input_writes = list(map_input(input_channels, {}))
    updated_channels = apply_writes(
        checkpoint,
        channels,
        [
            PregelTaskWrites((), INPUT, input_writes, []),
        ],
        get_next_version,
        trigger_to_nodes,
    )
    # prepare first tasks
    tasks = prepare_next_tasks(
        checkpoint,
        [],
        nodes,
        channels,
        managed,
        config,
        step,
        -1,
        for_execution=True,
        store=None,
        checkpointer=None,
        manager=None,
        trigger_to_nodes=trigger_to_nodes,
        updated_channels=updated_channels,
    )
    start_tasks = tasks
    # run the pregel loop
    for step in range(step, limit):
        if not tasks:
            break
        conditionals: dict[tuple[str, str, Any], str | None] = {}
        # run task writers
        for task in tasks.values():
            for w in task.writers:
                # apply regular writes
                if isinstance(w, ChannelWrite):
                    empty_input = (
                        cast(BaseChannel, specs["__root__"]).ValueType()
                        if "__root__" in specs
                        else None
                    )
                    w.invoke(empty_input, task.config)
                # apply conditional writes declared for static analysis, only once
                if w not in static_seen:
                    static_seen.add(w)
                    # apply static writes
                    if writes := ChannelWrite.get_static_writes(w):
                        # END writes are not written, but become edges directly
                        for t in writes:
                            if t[0] == END:
                                edges.add((task.name, t[0], True, t[2]))
                        writes = [t for t in writes if t[0] != END]
                        conditionals.update(
                            {(task.name, t[0], t[1] or None): t[2] for t in writes}
                        )
                        task.config[CONF][CONFIG_KEY_SEND]([t[:2] for t in writes])
        # collect sources
        step_sources = {
            task.name: {
                (
                    w[0],
                    (task.name, w[0], w[1] or None) in conditionals,
                    conditionals.get((task.name, w[0], w[1] or None)),
                )
                for w in task.writes
            }
            for task in tasks.values()
        }
        sources.update(step_sources)
        # invert triggers
        trigger_to_sources: dict[str, set[tuple[str, bool, str | None]]] = defaultdict(
            set
        )
        for src, triggers in sources.items():
            for trigger, cond, label in triggers:
                trigger_to_sources[trigger].add((src, cond, label))
        # apply writes
        updated_channels = apply_writes(
            checkpoint, channels, tasks.values(), get_next_version, trigger_to_nodes
        )
        # prepare next tasks
        tasks = prepare_next_tasks(
            checkpoint,
            [],
            nodes,
            channels,
            managed,
            config,
            step,
            limit,
            for_execution=True,
            store=None,
            checkpointer=None,
            manager=None,
            trigger_to_nodes=trigger_to_nodes,
            updated_channels=updated_channels,
        )
        # collect edges
        for task in tasks.values():
            added = False
            for trigger in task.triggers:
                for src, cond, label in sorted(trigger_to_sources[trigger]):
                    edges.add((src, task.name, cond, label))
                    # if the edge is from this step, skip adding the implicit edges
                    if (trigger, cond, label) in step_sources.get(src, set()):
                        added = True
                    else:
                        sources[src].discard((trigger, cond, label))
            # if no edges from this step, add implicit edges from all previous tasks
            if not added:
                for src in step_sources:
                    edges.add((src, task.name, True, None))
    # assemble the graph
    graph = Graph()
    # add nodes
    for name, node in nodes.items():
        metadata = dict(node.metadata or {})
        if name in interrupt_before_nodes and name in interrupt_after_nodes:
            metadata["__interrupt"] = "before,after"
        elif name in interrupt_before_nodes:
            metadata["__interrupt"] = "before"
        elif name in interrupt_after_nodes:
            metadata["__interrupt"] = "after"
        graph.add_node(node.bound, name, metadata=metadata or None)
    # add start node
    if START not in nodes:
        graph.add_node(None, START)
        for task in start_tasks.values():
            add_edge(graph, START, task.name)
    # add discovered edges
    for src, dest, is_conditional, label in sorted(edges):
        add_edge(
            graph,
            src,
            dest,
            data=label if label != dest else None,
            conditional=is_conditional,
        )
    # add end edges
    termini = {d for _, d, _, _ in edges if d != END}.difference(
        s for s, _, _, _ in edges
    )
    if termini:
        for src in sorted(termini):
            add_edge(graph, src, END)
    elif len(step_sources) == 1:
        for src in sorted(step_sources):
            add_edge(graph, src, END, conditional=True)
    # replace subgraphs
    for name, subgraph in subgraphs.items():
        if (
            len(subgraph.nodes) > 1
            and name in graph.nodes
            and subgraph.first_node()
            and subgraph.last_node()
        ):
            subgraph.trim_first_node()
            subgraph.trim_last_node()
            # replace the node with the subgraph
            graph.nodes.pop(name)
            first, last = graph.extend(subgraph, prefix=name)
            for idx, edge in enumerate(graph.edges):
                if edge.source == name:
                    edge = edge.copy(source=cast(Node, last).id)
                if edge.target == name:
                    edge = edge.copy(target=cast(Node, first).id)
                graph.edges[idx] = edge

    return graph


def add_edge(
    graph: Graph,
    source: str,
    target: str,
    *,
    data: Any | None = None,
    conditional: bool = False,
) -> None:
    """Add an edge to the graph."""
    for edge in graph.edges:
        if edge.source == source and edge.target == target:
            return
    if target not in graph.nodes and target == END:
        graph.add_node(None, END)
    graph.add_edge(graph.nodes[source], graph.nodes[target], data, conditional)
