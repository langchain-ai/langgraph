from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence, Union

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.graph import Graph

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import CONF, CONFIG_KEY_SEND, END, INPUT
from langgraph.managed.base import ManagedValueSpec
from langgraph.pregel.algo import (
    PregelTaskWrites,
    apply_writes,
    increment,
    prepare_next_tasks,
)
from langgraph.pregel.checkpoint import empty_checkpoint
from langgraph.pregel.io import map_input
from langgraph.pregel.manager import ChannelsManager
from langgraph.pregel.read import DEFAULT_BOUND, PregelNode
from langgraph.pregel.write import ChannelWrite, ChannelWriteTupleEntry
from langgraph.types import All, Checkpointer, LoopProtocol


def draw_graph(
    config: RunnableConfig,
    *,
    nodes: dict[str, PregelNode],
    specs: dict[str, Union[BaseChannel, ManagedValueSpec]],
    input_channels: Union[str, Sequence[str]],
    interrupt_after_nodes: Union[All, Sequence[str]],
    interrupt_before_nodes: Union[All, Sequence[str]],
    trigger_to_nodes: Optional[Mapping[str, Sequence[str]]],
    checkpointer: Checkpointer,
    subgraphs: dict[str, Graph],
) -> Graph:
    """Get the graph for this Pregel instance.

    Args:
        config: The configuration to use for the graph.
        subgraphs: The subgraphs to include in the graph.
        checkpointer: The checkpointer to use for the graph.

    Returns:
        The graph for this Pregel instance.
    """
    # (src, dest, is_conditional)
    edges: list[tuple[str, str, bool]] = []

    step = -1
    checkpoint = empty_checkpoint()
    get_next_version = (
        checkpointer.get_next_version
        if isinstance(checkpointer, BaseCheckpointSaver)
        else increment
    )
    with ChannelsManager(
        specs,
        checkpoint,
        LoopProtocol(step=step, stop=-1, config=config),
        skip_context=True,
    ) as (channels, managed):
        declared_seen: set[Any] = set()
        sources: dict[str, set[tuple[str, bool]]] = {}
        step_sources: dict[str, set[tuple[str, bool]]] = {}
        # remove node mappers
        nodes = {
            k: v.copy(update={"mapper": None}) if v.mapper is not None else v
            for k, v in nodes.items()
        }
        # apply input writes
        input_writes = list(map_input(input_channels, {}))
        _, updated_channels = apply_writes(
            checkpoint,
            channels,
            [
                PregelTaskWrites((), INPUT, input_writes, []),
            ],
            get_next_version,
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
            for_execution=True,
            store=None,
            checkpointer=None,
            manager=None,
            trigger_to_nodes=trigger_to_nodes,
            updated_channels=updated_channels,
        )
        # run the pregel loop
        while tasks:
            conditionals = set()
            # run task writers
            for task in tasks.values():
                for w in task.writers:
                    if isinstance(w, ChannelWrite):
                        w.invoke(None, task.config)
                        # apply declared writes (Command)
                        for entry in w.writes:
                            if (
                                isinstance(entry, ChannelWriteTupleEntry)
                                and entry.declared
                                and entry not in conditionals
                            ):
                                # visit only once
                                declared_seen.add(entry)
                                # apply them
                                current_len = len(task.writes)
                                task.config[CONF][CONFIG_KEY_SEND](entry.declared)
                                conditionals.update(list(task.writes)[current_len:])
                    elif w not in declared_seen:
                        # visit only once
                        declared_seen.add(w)
                        # get declared writes
                        if writes := ChannelWrite.get_declared_writes(w):
                            # apply them
                            current_len = len(task.writes)
                            ChannelWrite.do_write(task.config, writes)
                            conditionals.update(list(task.writes)[current_len:])
            # collect sources
            step_sources = {
                task.name: {(w[0], w in conditionals) for w in task.writes}
                for task in tasks.values()
            }
            sources.update(step_sources)
            # invert triggers
            trigger_to_sources: dict[str, set[tuple[str, bool]]] = defaultdict(set)
            for src, triggers in sources.items():
                for trigger, cond in triggers:
                    trigger_to_sources[trigger].add((src, cond))
            # apply writes
            _, updated_channels = apply_writes(
                checkpoint, channels, tasks.values(), get_next_version
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
                for_execution=True,
                store=None,
                checkpointer=None,
                manager=None,
                trigger_to_nodes=trigger_to_nodes,
                updated_channels=updated_channels,
            )
            # collect edges
            for task in tasks.values():
                for trigger in task.triggers:
                    for src, cond in sorted(trigger_to_sources[trigger]):
                        edges.append((src, task.name, cond))
        # assemble the graph
        graph = Graph()
        for name, node in nodes.items():
            metadata = dict(node.metadata or {})
            if name in interrupt_before_nodes and name in interrupt_after_nodes:
                metadata["__interrupt"] = "before,after"
            elif name in interrupt_before_nodes:
                metadata["__interrupt"] = "before"
            elif name in interrupt_after_nodes:
                metadata["__interrupt"] = "after"
            graph.add_node(node.bound, name, metadata=metadata)
        for src, dest, is_conditional in edges:
            # TODO conditional labels
            graph.add_edge(
                graph.nodes[src], graph.nodes[dest], conditional=is_conditional
            )
        # replace subgraphs
        for name, subgraph in subgraphs.items():
            subgraph.trim_first_node()
            subgraph.trim_last_node()
            if (
                len(subgraph.nodes) > 1
                and name in graph.nodes
                and subgraph.first_node()
                and subgraph.last_node()
            ):
                # replace the node with the subgraph
                graph.nodes.pop(name)
                first, last = graph.extend(subgraph, prefix=name)
                for idx, edge in enumerate(graph.edges):
                    if edge.source == name:
                        graph.edges[idx] = edge.copy(source=last)
                    elif edge.target == name:
                        graph.edges[idx] = edge.copy(target=first)
        # add end edges
        if step_sources:
            end = graph.add_node(DEFAULT_BOUND, END)
            for src in step_sources:
                graph.add_edge(graph.nodes[src], end)
            termini = set(d for _, d, _ in edges).difference((s for s, _, _ in edges))
            for src in termini.union(step_sources):
                # TODO conditional labels
                graph.add_edge(graph.nodes[src], end, conditional=src not in termini)

        return graph
