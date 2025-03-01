from __future__ import annotations

import concurrent
import concurrent.futures
import queue
from collections import deque
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)
from uuid import UUID, uuid5

from typing_extensions import Self

from langgraph.channels.base import (
    BaseChannel,
)
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointConfig,
    CheckpointTuple,
    copy_checkpoint,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_NODE_FINISHED,
    CONFIG_KEY_READ,
    CONFIG_KEY_RUNNER_SUBMIT,
    CONFIG_KEY_SEND,
    CONFIG_KEY_STORE,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_STREAM_WRITER,
    CONFIG_KEY_TASK_ID,
    END,
    ERROR,
    INPUT,
    INTERRUPT,
    NS_END,
    NS_SEP,
    NULL_TASK_ID,
    PUSH,
    SCHEDULED,
)
from langgraph.errors import (
    ErrorCode,
    GraphRecursionError,
    InvalidUpdateError,
    create_error_message,
)
from langgraph.pregel.algo import (
    PregelTaskWrites,
    apply_writes,
    local_read,
    local_write,
    prepare_next_tasks,
)
from langgraph.pregel.debug import tasks_w_writes
from langgraph.pregel.io import read_channels
from langgraph.pregel.loop import StreamProtocol, SyncPregelLoop
from langgraph.pregel.manager import ChannelsManager
from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.read import DEFAULT_BOUND, PregelNode
from langgraph.pregel.retry import RetryPolicy
from langgraph.pregel.runner import PregelRunner
from langgraph.pregel.utils import get_new_channel_versions
from langgraph.pregel.validate import validate_graph, validate_keys
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.store.base import BaseStore
from langgraph.types import (
    All,
    Checkpointer,
    StateSnapshot,
    StreamMode,
)
from langgraph.utils.config import (
    AnyConfig,
    RunnableConfig,
    ensure_config,
    get_runtree_for_config,
    merge_configs,
    patch_checkpoint_map,
    patch_config,
    patch_configurable,
    recast_checkpoint_ns,
)
from langgraph.utils.queue import SyncQueue  # type: ignore[attr-defined]
from langgraph.utils.runnable import (
    Runnable,
    RunnableLike,
    RunnableSeq,
    coerce_to_runnable,
)

WriteValue = Union[Callable[[Any], Any], Any]


class NodeBuilder:
    channels: Union[list[str], dict[str, str]]
    triggers: list[str]
    tags: list[str]
    writes: list[ChannelWriteEntry]
    bound: Runnable

    def __init__(
        self,
    ) -> None:
        self.channels = {}
        self.triggers = []
        self.tags = []
        self.writes = []
        self.bound = DEFAULT_BOUND

    def subscribe_to(
        self,
        channels: Union[str, Sequence[str]],
        *,
        key: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Self:
        """Add channels to subscribe to with optional key and tags.

        Args:
            channels: Channel name(s) to subscribe to
            key: Optional key to use for the channel in input
            tags: Optional tags to add to the node

        Returns:
            Self for chaining
        """
        if not isinstance(channels, str) and key is not None:
            raise ValueError(
                "Can't specify a key when subscribing to multiple channels"
            )

        if isinstance(channels, str) and key is not None:
            self.channels = {key: channels}
        elif isinstance(channels, str):
            if isinstance(self.channels, list):
                self.channels.append(channels)
            elif not self.channels:
                self.channels = [channels]
            else:
                self.channels = list(self.channels.values())
                self.channels.append(channels)
        else:
            if not self.channels:
                self.channels = {chan: chan for chan in channels}
            elif isinstance(self.channels, list):
                self.channels = {
                    **{chan: chan for chan in self.channels},
                    **{chan: chan for chan in channels},
                }
            else:
                self.channels.update({chan: chan for chan in channels})

        if isinstance(channels, str):
            self.triggers.append(channels)
        else:
            self.triggers.extend(channels)

        if tags:
            self.tags.extend(tags)

        return self

    def read_from(
        self,
        *channels: str,
    ) -> Self:
        """Adds the specified channels to read from, without subscribing to them."""
        assert self.channels and isinstance(
            self.channels, dict
        ), "Channels must be specified first"
        self.channels.update({c: c for c in channels})
        return self

    def add_node(
        self,
        node: RunnableLike,
    ) -> Self:
        """Adds the specified node."""
        if self.bound is not DEFAULT_BOUND:
            self.bound = RunnableSeq(self.bound, coerce_to_runnable(node))
        else:
            self.bound = coerce_to_runnable(node)
        return self

    def write_to(
        self,
        *channels: str,
        **kwargs: WriteValue,
    ) -> Self:
        """Add channel writes.

        Args:
            *channels: Channel names to write to
            **kwargs: Channel name and value mappings

        Returns:
            Self for chaining
        """
        self.writes.extend([ChannelWriteEntry(c) for c in channels])
        self.writes.extend(
            [
                (
                    ChannelWriteEntry(k, mapper=v)
                    if callable(v)
                    else ChannelWriteEntry(k, value=v)
                )
                for k, v in kwargs.items()
            ]
        )

        return self

    def build(self) -> PregelNode:
        """Builds the node."""
        assert self.triggers, "No channels specified"
        return PregelNode(
            channels=self.channels,
            triggers=self.triggers,
            tags=self.tags,
            writers=[ChannelWrite(self.writes)],
            bound=self.bound,
        )


class Pregel(PregelProtocol):
    """Pregel manages the runtime behavior for LangGraph applications.

    ## Overview

    Pregel combines [**actors**](https://en.wikipedia.org/wiki/Actor_model)
    and **channels** into a single application.
    **Actors** read data from channels and write data to channels.
    Pregel organizes the execution of the application into multiple steps,
    following the **Pregel Algorithm**/**Bulk Synchronous Parallel** model.

    Each step consists of three phases:

    - **Plan**: Determine which **actors** to execute in this step. For example,
        in the first step, select the **actors** that subscribe to the special
        **input** channels; in subsequent steps,
        select the **actors** that subscribe to channels updated in the previous step.
    - **Execution**: Execute all selected **actors** in parallel,
        until all complete, or one fails, or a timeout is reached. During this
        phase, channel updates are invisible to actors until the next step.
    - **Update**: Update the channels with the values written by the **actors**
        in this step.

    Repeat until no **actors** are selected for execution, or a maximum number of
    steps is reached.

    ## Actors

    An **actor** is a [PregelNode][langgraph.pregel.read.PregelNode].
    It subscribes to channels, reads data from them, and writes data to them.
    It can be thought of as an **actor** in the Pregel algorithm.
    [PregelNodes][langgraph.pregel.read.PregelNode] implement LangChain's
    Runnable interface.

    ## Channels

    Channels are used to communicate between actors (PregelNodes).
    Each channel has a value type, an update type, and an update function – which
    takes a sequence of updates and
    modifies the stored value. Channels can be used to send data from one chain to
    another, or to send data from a chain to itself in a future step. LangGraph
    provides a number of built-in channels:

    ### Basic channels: LastValue and Topic

    - `LastValue`: The default channel, stores the last value sent to the channel,
       useful for input and output values, or for sending data from one step to the next
    - `Topic`: A configurable PubSub Topic, useful for sending multiple values
       between *actors*, or for accumulating output. Can be configured to deduplicate
       values, and/or to accumulate values over the course of multiple steps.

    ### Advanced channels: Context and BinaryOperatorAggregate

    - `Context`: exposes the value of a context manager, managing its lifecycle.
      Useful for accessing external resources that require setup and/or teardown. eg.
      `client = Context(httpx.Client)`
    - `BinaryOperatorAggregate`: stores a persistent value, updated by applying
       a binary operator to the current value and each update
       sent to the channel, useful for computing aggregates over multiple steps. eg.
      `total = BinaryOperatorAggregate(int, operator.add)`

    ## Examples

    Most users will interact with Pregel via a
    [StateGraph (Graph API)][langgraph.graph.StateGraph] or via an
    [entrypoint (Functional API)][langgraph.func.entrypoint].

    However, for **advanced** use cases, Pregel can be used directly. If you're
    not sure whether you need to use Pregel directly, then the answer is probably no
    – you should use the Graph API or Functional API instead. These are higher-level
    interfaces that will compile down to Pregel under the hood.

    Here are some examples to give you a sense of how it works:

    Example: Single node application
        ```python
        from langgraph.channels import EphemeralValue
        from langgraph.pregel import Pregel, Channel, ChannelWriteEntry

        node1 = (
            Channel.subscribe_to("a")
            | (lambda x: x + x)
            | Channel.write_to("b")
        )

        app = Pregel(
            nodes={"node1": node1},
            channels={
                "a": EphemeralValue(str),
                "b": EphemeralValue(str),
            },
            input_channels=["a"],
            output_channels=["b"],
        )

        app.invoke({"a": "foo"})
        ```

        ```con
        {'b': 'foofoo'}
        ```

    Example: Using multiple nodes and multiple output channels
        ```python
        from langgraph.channels import LastValue, EphemeralValue
        from langgraph.pregel import Pregel, Channel, ChannelWriteEntry

        node1 = (
            Channel.subscribe_to("a")
            | (lambda x: x + x)
            | Channel.write_to("b")
        )

        node2 = (
            Channel.subscribe_to("b")
            | (lambda x: x + x)
            | Channel.write_to("c")
        )


        app = Pregel(
            nodes={"node1": node1, "node2": node2},
            channels={
                "a": EphemeralValue(str),
                "b": LastValue(str),
                "c": EphemeralValue(str),
            },
            input_channels=["a"],
            output_channels=["b", "c"],
        )

        app.invoke({"a": "foo"})
        ```

        ```con
        {'b': 'foofoo', 'c': 'foofoofoofoo'}
        ```

    Example: Using a Topic channel
        ```python
        from langgraph.channels import LastValue, EphemeralValue, Topic
        from langgraph.pregel import Pregel, Channel, ChannelWriteEntry

        node1 = (
            Channel.subscribe_to("a")
            | (lambda x: x + x)
            | {
                "b": Channel.write_to("b"),
                "c": Channel.write_to("c")
            }
        )

        node2 = (
            Channel.subscribe_to("b")
            | (lambda x: x + x)
            | {
                "c": Channel.write_to("c"),
            }
        )


        app = Pregel(
            nodes={"node1": node1, "node2": node2},
            channels={
                "a": EphemeralValue(str),
                "b": EphemeralValue(str),
                "c": Topic(str, accumulate=True),
            },
            input_channels=["a"],
            output_channels=["c"],
        )

        app.invoke({"a": "foo"})
        ```

        ```pycon
        {'c': ['foofoo', 'foofoofoofoo']}
        ```

    Example: Using a BinaryOperatorAggregate channel
        ```python
        from langgraph.channels import EphemeralValue, BinaryOperatorAggregate
        from langgraph.pregel import Pregel, Channel


        node1 = (
            Channel.subscribe_to("a")
            | (lambda x: x + x)
            | {
                "b": Channel.write_to("b"),
                "c": Channel.write_to("c")
            }
        )

        node2 = (
            Channel.subscribe_to("b")
            | (lambda x: x + x)
            | {
                "c": Channel.write_to("c"),
            }
        )


        def reducer(current, update):
            if current:
                return current + " | " + "update"
            else:
                return update

        app = Pregel(
            nodes={"node1": node1, "node2": node2},
            channels={
                "a": EphemeralValue(str),
                "b": EphemeralValue(str),
                "c": BinaryOperatorAggregate(str, operator=reducer),
            },
            input_channels=["a"],
            output_channels=["c"]
        )

        app.invoke({"a": "foo"})
        ```

        ```con
        {'c': 'foofoo | foofoofoofoo'}
        ```

    Example: Introducing a cycle
        This example demonstrates how to introduce a cycle in the graph, by having
        a chain write to a channel it subscribes to. Execution will continue
        until a None value is written to the channel.

        ```python
        from langgraph.channels import EphemeralValue
        from langgraph.pregel import Pregel, Channel, ChannelWrite, ChannelWriteEntry

        example_node = (
            Channel.subscribe_to("value")
            | (lambda x: x + x if len(x) < 10 else None)
            | ChannelWrite(writes=[ChannelWriteEntry(channel="value", skip_none=True)])
        )

        app = Pregel(
            nodes={"example_node": example_node},
            channels={
                "value": EphemeralValue(str),
            },
            input_channels=["value"],
            output_channels=["value"]
        )

        app.invoke({"value": "a"})
        ```

        ```con
        {'value': 'aaaaaaaaaaaaaaaa'}
        ```
    """

    nodes: dict[str, PregelNode]

    channels: dict[str, BaseChannel]

    stream_mode: StreamMode = "values"
    """Mode to stream output, defaults to 'values'."""

    stream_eager: bool = False
    """Whether to force emitting stream events eagerly, automatically turned on
    for stream_mode "messages" and "custom"."""

    output_channels: Union[str, Sequence[str]]

    stream_channels: Optional[Union[str, Sequence[str]]] = None
    """Channels to stream, defaults to all channels not in reserved channels"""

    interrupt_after_nodes: Union[All, Sequence[str]]

    interrupt_before_nodes: Union[All, Sequence[str]]

    input_channels: Union[str, Sequence[str]]

    step_timeout: Optional[float] = None
    """Maximum time to wait for a step to complete, in seconds. Defaults to None."""

    checkpointer: Checkpointer = None
    """Checkpointer used to save and load graph state. Defaults to None."""

    store: Optional[BaseStore] = None
    """Memory store to use for SharedValues. Defaults to None."""

    retry_policy: Optional[RetryPolicy] = None
    """Retry policy to use when running tasks. Set to None to disable."""

    config_type: Optional[Type[Any]] = None

    config: Optional[RunnableConfig] = None

    name: str = "LangGraph"

    def __init__(
        self,
        *,
        nodes: dict[str, PregelNode],
        channels: Optional[dict[str, BaseChannel]],
        auto_validate: bool = True,
        stream_mode: StreamMode = "values",
        stream_eager: bool = False,
        output_channels: Union[str, Sequence[str]],
        stream_channels: Optional[Union[str, Sequence[str]]] = None,
        interrupt_after_nodes: Union[All, Sequence[str]] = (),
        interrupt_before_nodes: Union[All, Sequence[str]] = (),
        input_channels: Union[str, Sequence[str]],
        step_timeout: Optional[float] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        store: Optional[BaseStore] = None,
        retry_policy: Optional[RetryPolicy] = None,
        config_type: Optional[Type[Any]] = None,
        config: Optional[RunnableConfig] = None,
        name: str = "LangGraph",
    ) -> None:
        self.nodes = nodes
        self.channels = channels or {}
        self.stream_mode = stream_mode
        self.stream_eager = stream_eager
        self.output_channels = output_channels
        self.stream_channels = stream_channels
        self.interrupt_after_nodes = interrupt_after_nodes
        self.interrupt_before_nodes = interrupt_before_nodes
        self.input_channels = input_channels
        self.step_timeout = step_timeout
        self.checkpointer = checkpointer
        self.store = store
        self.retry_policy = retry_policy
        self.config_type = config_type
        self.config = config
        self.name = name
        if auto_validate:
            self.validate()

    def copy(self, update: dict[str, Any] | None = None) -> Self:
        attrs = {**self.__dict__, **(update or {})}
        return self.__class__(**attrs)

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self:
        return self.copy(
            {"config": merge_configs(self.config, config, cast(RunnableConfig, kwargs))}
        )

    def validate(self) -> Self:
        validate_graph(
            self.nodes,
            {k: v for k, v in self.channels.items() if isinstance(v, BaseChannel)},
            self.input_channels,
            self.output_channels,
            self.stream_channels,
            self.interrupt_after_nodes,
            self.interrupt_before_nodes,
        )
        return self

    @property
    def stream_channels_list(self) -> Sequence[str]:
        stream_channels = self.stream_channels_asis
        return (
            [stream_channels] if isinstance(stream_channels, str) else stream_channels
        )

    @property
    def stream_channels_asis(self) -> Union[str, Sequence[str]]:
        return self.stream_channels or [
            k for k in self.channels if isinstance(self.channels[k], BaseChannel)
        ]

    def get_subgraphs(
        self, *, namespace: Optional[str] = None, recurse: bool = False
    ) -> Iterator[tuple[str, PregelProtocol]]:
        for name, node in self.nodes.items():
            # filter by prefix
            if namespace is not None:
                if not namespace.startswith(name):
                    continue

            # find the subgraph, if any
            graph = node.subgraphs[0] if node.subgraphs else None

            # if found, yield recursively
            if graph:
                if name == namespace:
                    yield name, graph
                    return  # we found it, stop searching
                if namespace is None:
                    yield name, graph
                if recurse and isinstance(graph, Pregel):
                    if namespace is not None:
                        namespace = namespace[len(name) + 1 :]
                    yield from (
                        (f"{name}{NS_SEP}{n}", s)
                        for n, s in graph.get_subgraphs(
                            namespace=namespace, recurse=recurse
                        )
                    )

    def _prepare_state_snapshot(
        self,
        config: CheckpointConfig,
        saved: Optional[CheckpointTuple],
        recurse: Optional[BaseCheckpointSaver] = None,
        apply_pending_writes: bool = False,
    ) -> StateSnapshot:
        if not saved:
            return StateSnapshot(
                values={},
                next=(),
                config=config,
                metadata=None,
                created_at=None,
                parent_config=None,
                tasks=(),
            )

        with ChannelsManager(
            self.channels,
            saved.checkpoint,
        ) as channels:
            # tasks for this checkpoint
            next_tasks = prepare_next_tasks(
                saved.checkpoint,
                saved.pending_writes or [],
                self.nodes,
                channels,
                saved.config,
                saved.metadata.get("step", -1) + 1,
                for_execution=True,
                store=self.store,
                checkpointer=self.checkpointer
                if isinstance(self.checkpointer, BaseCheckpointSaver)
                else None,
            )
            # get the subgraphs
            subgraphs = dict(self.get_subgraphs())
            parent_ns = saved.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
            task_states: dict[str, Union[RunnableConfig, StateSnapshot]] = {}
            for task in next_tasks.values():
                if task.name not in subgraphs:
                    continue
                # assemble checkpoint_ns for this task
                task_ns = f"{task.name}{NS_END}{task.id}"
                if parent_ns:
                    task_ns = f"{parent_ns}{NS_SEP}{task_ns}"
                if not recurse:
                    # set config as signal that subgraph checkpoints exist
                    config = {
                        CONF: {
                            "thread_id": saved.config[CONF]["thread_id"],
                            CONFIG_KEY_CHECKPOINT_NS: task_ns,
                        }
                    }
                    task_states[task.id] = config
                else:
                    # get the state of the subgraph
                    config = {
                        CONF: {
                            CONFIG_KEY_CHECKPOINTER: recurse,
                            "thread_id": saved.config[CONF]["thread_id"],
                            CONFIG_KEY_CHECKPOINT_NS: task_ns,
                        }
                    }
                    task_states[task.id] = subgraphs[task.name].get_state(
                        config, subgraphs=True
                    )
            # apply pending writes
            if null_writes := [
                w[1:] for w in saved.pending_writes or [] if w[0] == NULL_TASK_ID
            ]:
                apply_writes(
                    saved.checkpoint,
                    channels,
                    [PregelTaskWrites((), INPUT, null_writes, [])],
                    None,
                )
            if apply_pending_writes and saved.pending_writes:
                for tid, k, v in saved.pending_writes:
                    if k in (ERROR, INTERRUPT, SCHEDULED):
                        continue
                    if tid not in next_tasks:
                        continue
                    next_tasks[tid].writes.append((k, v))
                if tasks := [t for t in next_tasks.values() if t.writes]:
                    apply_writes(saved.checkpoint, channels, tasks, None)
            # assemble the state snapshot
            return StateSnapshot(
                read_channels(channels, self.stream_channels_asis),
                tuple(t.name for t in next_tasks.values() if not t.writes),
                patch_checkpoint_map(saved.config, saved.metadata),
                saved.metadata,
                saved.checkpoint["ts"],
                patch_checkpoint_map(saved.parent_config, saved.metadata),
                tasks_w_writes(
                    next_tasks.values(),
                    saved.pending_writes,
                    task_states,
                    self.stream_channels_asis,
                ),
            )

    def get_state(self, config: AnyConfig, *, subgraphs: bool = False) -> StateSnapshot:
        """Get the current state of the graph."""
        checkpointer: Optional[BaseCheckpointSaver] = ensure_config(config)[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast = recast_checkpoint_ns(checkpoint_ns)
            # find the subgraph with the matching name
            for _, pregel in self.get_subgraphs(namespace=recast, recurse=True):
                return pregel.get_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    subgraphs=subgraphs,
                )
            else:
                raise ValueError(f"Subgraph {recast} not found")

        config = merge_configs(self.config, config) if self.config else config
        if self.checkpointer is True:
            ns = cast(str, config[CONF][CONFIG_KEY_CHECKPOINT_NS])
            config = merge_configs(
                config, {CONF: {CONFIG_KEY_CHECKPOINT_NS: recast_checkpoint_ns(ns)}}
            )

        saved = checkpointer.get_tuple(config)
        return self._prepare_state_snapshot(
            config,
            saved,
            recurse=checkpointer if subgraphs else None,
            apply_pending_writes=CONFIG_KEY_CHECKPOINT_ID not in config[CONF],
        )

    def get_state_history(
        self,
        config: AnyConfig,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[StateSnapshot]:
        config = ensure_config(config)
        """Get the history of the state of the graph."""
        checkpointer: Optional[BaseCheckpointSaver] = ensure_config(config)[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast = recast_checkpoint_ns(checkpoint_ns)
            # find the subgraph with the matching name
            for _, pregel in self.get_subgraphs(namespace=recast, recurse=True):
                yield from pregel.get_state_history(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    filter=filter,
                    before=before,
                    limit=limit,
                )
                return
            else:
                raise ValueError(f"Subgraph {recast} not found")

        config = merge_configs(
            self.config,
            config,
            {CONF: {CONFIG_KEY_CHECKPOINT_NS: checkpoint_ns}},
        )
        # eagerly consume list() to avoid holding up the db cursor
        for checkpoint_tuple in list(
            checkpointer.list(config, before=before, limit=limit, filter=filter)
        ):
            yield self._prepare_state_snapshot(
                checkpoint_tuple.config, checkpoint_tuple
            )

    def update_state(
        self,
        config: AnyConfig,
        values: Optional[Union[dict[str, Any], Any]],
        as_node: Optional[str] = None,
    ) -> AnyConfig:
        """Update the state of the graph with the given values, as if they came from
        node `as_node`. If `as_node` is not provided, it will be set to the last node
        that updated the state, if not ambiguous.
        """
        checkpointer: Optional[BaseCheckpointSaver] = ensure_config(config)[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        # delegate to subgraph
        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast = recast_checkpoint_ns(checkpoint_ns)
            # find the subgraph with the matching name
            for _, pregel in self.get_subgraphs(namespace=recast, recurse=True):
                return pregel.update_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    values,
                    as_node,
                )
            else:
                raise ValueError(f"Subgraph {recast} not found")

        # get last checkpoint
        config = ensure_config(self.config, config)
        saved = checkpointer.get_tuple(config)
        checkpoint = copy_checkpoint(saved.checkpoint) if saved else empty_checkpoint()
        checkpoint_previous_versions = (
            saved.checkpoint["channel_versions"].copy() if saved else {}
        )
        step = saved.metadata.get("step", -1) if saved else -1
        # merge configurable fields with previous checkpoint config
        checkpoint_config = patch_configurable(
            config,
            {CONFIG_KEY_CHECKPOINT_NS: config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")},
        )
        checkpoint_metadata = config["metadata"]
        if saved:
            checkpoint_config = patch_configurable(config, saved.config[CONF])
            checkpoint_metadata = {**saved.metadata, **checkpoint_metadata}
        with ChannelsManager(
            self.channels,
            checkpoint,
        ) as channels:
            # no values as END, just clear all tasks
            if values is None and as_node == END:
                if saved is not None:
                    # tasks for this checkpoint
                    next_tasks = prepare_next_tasks(
                        checkpoint,
                        saved.pending_writes or [],
                        self.nodes,
                        channels,
                        saved.config,
                        saved.metadata.get("step", -1) + 1,
                        for_execution=True,
                        store=self.store,
                        checkpointer=self.checkpointer
                        if isinstance(self.checkpointer, BaseCheckpointSaver)
                        else None,
                    )
                    # apply null writes
                    if null_writes := [
                        w[1:]
                        for w in saved.pending_writes or []
                        if w[0] == NULL_TASK_ID
                    ]:
                        apply_writes(
                            saved.checkpoint,
                            channels,
                            [PregelTaskWrites((), INPUT, null_writes, [])],
                            None,
                        )
                    # apply writes from tasks that already ran
                    for tid, k, v in saved.pending_writes or []:
                        if k in (ERROR, INTERRUPT, SCHEDULED):
                            continue
                        if tid not in next_tasks:
                            continue
                        next_tasks[tid].writes.append((k, v))
                    # clear all current tasks
                    apply_writes(checkpoint, channels, next_tasks.values(), None)
                # save checkpoint
                next_config = checkpointer.put(
                    checkpoint_config,
                    create_checkpoint(checkpoint, None, step),
                    {
                        **checkpoint_metadata,
                        "source": "update",
                        "step": step + 1,
                        "writes": {},
                        "parents": saved.metadata.get("parents", {}) if saved else {},
                    },
                    {},
                )
                return patch_checkpoint_map(
                    next_config, saved.metadata if saved else None
                )
            # no values, empty checkpoint
            if values is None and as_node is None:
                next_checkpoint = create_checkpoint(checkpoint, None, step)
                # copy checkpoint
                next_config = checkpointer.put(
                    checkpoint_config,
                    next_checkpoint,
                    {
                        **checkpoint_metadata,
                        "source": "update",
                        "step": step + 1,
                        "writes": {},
                        "parents": saved.metadata.get("parents", {}) if saved else {},
                    },
                    {},
                )
                return patch_checkpoint_map(
                    next_config, saved.metadata if saved else None
                )
            # no values, copy checkpoint
            if values is None and as_node == "__copy__":
                next_checkpoint = create_checkpoint(checkpoint, None, step)
                # copy checkpoint
                next_config = checkpointer.put(
                    saved.parent_config or saved.config if saved else checkpoint_config,
                    next_checkpoint,
                    {
                        **checkpoint_metadata,
                        "source": "fork",
                        "step": step + 1,
                        "parents": saved.metadata.get("parents", {}) if saved else {},
                    },
                    {},
                )
                return patch_checkpoint_map(
                    next_config, saved.metadata if saved else None
                )
            # apply pending writes, if not on specific checkpoint
            if (
                CONFIG_KEY_CHECKPOINT_ID not in config[CONF]
                and saved is not None
                and saved.pending_writes
            ):
                # tasks for this checkpoint
                next_tasks = prepare_next_tasks(
                    checkpoint,
                    saved.pending_writes,
                    self.nodes,
                    channels,
                    saved.config,
                    saved.metadata.get("step", -1) + 1,
                    for_execution=True,
                    store=self.store,
                    checkpointer=self.checkpointer
                    if isinstance(self.checkpointer, BaseCheckpointSaver)
                    else None,
                )
                # apply null writes
                if null_writes := [
                    w[1:] for w in saved.pending_writes or [] if w[0] == NULL_TASK_ID
                ]:
                    apply_writes(
                        saved.checkpoint,
                        channels,
                        [PregelTaskWrites((), INPUT, null_writes, [])],
                        None,
                    )
                # apply writes
                for tid, k, v in saved.pending_writes:
                    if k in (ERROR, INTERRUPT, SCHEDULED):
                        continue
                    if tid not in next_tasks:
                        continue
                    next_tasks[tid].writes.append((k, v))
                if tasks := [t for t in next_tasks.values() if t.writes]:
                    apply_writes(checkpoint, channels, tasks, None)
            # find last node that updated the state, if not provided
            if as_node is None and not any(
                v for vv in checkpoint["versions_seen"].values() for v in vv.values()
            ):
                if (
                    isinstance(self.input_channels, str)
                    and self.input_channels in self.nodes
                ):
                    as_node = self.input_channels
            elif as_node is None:
                last_seen_by_node = sorted(
                    (v, n)
                    for n, seen in checkpoint["versions_seen"].items()
                    if n in self.nodes
                    for v in seen.values()
                )
                # if two nodes updated the state at the same time, it's ambiguous
                if last_seen_by_node:
                    if len(last_seen_by_node) == 1:
                        as_node = last_seen_by_node[0][1]
                    elif last_seen_by_node[-1][0] != last_seen_by_node[-2][0]:
                        as_node = last_seen_by_node[-1][1]
            if as_node is None:
                raise InvalidUpdateError("Ambiguous update, specify as_node")
            if as_node not in self.nodes:
                raise InvalidUpdateError(f"Node {as_node} does not exist")
            # create task to run all writers of the chosen node
            writers = self.nodes[as_node].flat_writers
            if not writers:
                raise InvalidUpdateError(f"Node {as_node} has no writers")
            writes: deque[tuple[str, Any]] = deque()
            task = PregelTaskWrites((), as_node, writes, [INTERRUPT])
            task_id = str(uuid5(UUID(checkpoint["id"]), INTERRUPT))
            run = RunnableSeq(*writers) if len(writers) > 1 else writers[0]
            # execute task
            run.invoke(
                values,
                patch_config(
                    config,
                    run_name=self.name + "UpdateState",
                    configurable={
                        # deque.extend is thread-safe
                        CONFIG_KEY_SEND: partial(
                            local_write,
                            writes.extend,
                            self.nodes.keys(),
                        ),
                        CONFIG_KEY_READ: partial(
                            local_read,
                            checkpoint,
                            channels,
                            task,
                        ),
                    },
                ),
            )
            # save task writes
            # channel writes are saved to current checkpoint
            # push writes are saved to next checkpoint
            channel_writes, push_writes = (
                [w for w in task.writes if w[0] != PUSH],
                [w for w in task.writes if w[0] == PUSH],
            )
            if saved and channel_writes:
                checkpointer.put_writes(checkpoint_config, channel_writes, task_id)
            # apply to checkpoint and save
            apply_writes(checkpoint, channels, [task], checkpointer.get_next_version)
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            next_config = checkpointer.put(
                checkpoint_config,
                checkpoint,
                {
                    **checkpoint_metadata,
                    "source": "update",
                    "step": step + 1,
                    "writes": {as_node: values},
                    "parents": saved.metadata.get("parents", {}) if saved else {},
                },
                get_new_channel_versions(
                    checkpoint_previous_versions, checkpoint["channel_versions"]
                ),
            )
            if push_writes:
                checkpointer.put_writes(next_config, push_writes, task_id)
            return patch_checkpoint_map(next_config, saved.metadata if saved else None)

    def _defaults(
        self,
        config: RunnableConfig,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]],
        log_mode: Optional[Union[StreamMode, list[StreamMode]]],
        output_keys: Optional[Union[str, Sequence[str]]],
        interrupt_before: Optional[Union[All, Sequence[str]]],
        interrupt_after: Optional[Union[All, Sequence[str]]],
    ) -> tuple[
        set[StreamMode],
        set[StreamMode],
        Union[str, Sequence[str]],
        Union[All, Sequence[str]],
        Union[All, Sequence[str]],
        Optional[BaseCheckpointSaver],
        Optional[BaseStore],
    ]:
        if config["recursion_limit"] < 1:
            raise ValueError("recursion_limit must be at least 1")
        if output_keys is None:
            output_keys = self.stream_channels_asis
        else:
            validate_keys(output_keys, self.channels)
        interrupt_before = interrupt_before or self.interrupt_before_nodes
        interrupt_after = interrupt_after or self.interrupt_after_nodes
        stream_mode = stream_mode if stream_mode is not None else self.stream_mode
        if not isinstance(stream_mode, list):
            stream_mode = [stream_mode]
        if CONFIG_KEY_TASK_ID in config.get(CONF, {}):
            # if being called as a node in another graph, always use values mode
            stream_mode = ["values"]
        log_modes: set[StreamMode] = set()
        if isinstance(log_mode, str):
            log_modes.add(log_mode)
        elif log_mode:
            log_modes.update(log_mode)
        if self.checkpointer is False:
            checkpointer: Optional[BaseCheckpointSaver] = None
        elif CONFIG_KEY_CHECKPOINTER in config.get(CONF, {}):
            checkpointer = config[CONF][CONFIG_KEY_CHECKPOINTER]
        elif self.checkpointer is True:
            raise RuntimeError("checkpointer=True cannot be used for root graphs.")
        else:
            checkpointer = self.checkpointer
        if checkpointer and not config.get(CONF):
            raise ValueError(
                "Checkpointer requires one or more of the following 'configurable' fields: thread_id, checkpoint_id, checkpoint_ns"
            )
        if CONFIG_KEY_STORE in config.get(CONF, {}):
            store: Optional[BaseStore] = config[CONF][CONFIG_KEY_STORE]
        else:
            store = self.store
        return (
            set(stream_mode),
            log_modes,
            output_keys,
            interrupt_before,
            interrupt_after,
            checkpointer,
            store,
        )

    def stream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[AnyConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        log_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        subgraphs: bool = False,
    ) -> Iterator[Union[dict[str, Any], Any]]:
        """Stream graph steps for a single input.

        Args:
            input: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to self.stream_mode.
                Options are:

                - `"values"`: Emit all values in the state after each step.
                    When used with functional API, values are emitted once at the end of the workflow.
                - `"updates"`: Emit only the node or task names and updates returned by the nodes or tasks after each step.
                    If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are emitted separately.
                - `"custom"`: Emit custom data from inside nodes or tasks using `StreamWriter`.
                - `"debug"`: Emit debug events with as much information as possible for each step.
            log_modes: The stream modes to log, defaults to none, useful for debugging.
            output_keys: The keys to stream, defaults to all non-context channels.
            interrupt_before: Nodes to interrupt before, defaults to all nodes in the graph.
            interrupt_after: Nodes to interrupt after, defaults to all nodes in the graph.
            subgraphs: Whether to stream subgraphs, defaults to False.

        Yields:
            The output of each step in the graph. The output shape depends on the stream_mode.

        Examples:
            Using different stream modes with a graph:
            ```pycon
            >>> import operator
            >>> from typing_extensions import Annotated, TypedDict
            >>> from langgraph.graph import StateGraph, START
            ...
            >>> class State(TypedDict):
            ...     alist: Annotated[list, operator.add]
            ...     another_list: Annotated[list, operator.add]
            ...
            >>> builder = StateGraph(State)
            >>> builder.add_node("a", lambda _state: {"another_list": ["hi"]})
            >>> builder.add_node("b", lambda _state: {"alist": ["there"]})
            >>> builder.add_edge("a", "b")
            >>> builder.add_edge(START, "a")
            >>> graph = builder.compile()
            ```
            With stream_mode="values":

            ```pycon
            >>> for event in graph.stream({"alist": ['Ex for stream_mode="values"']}, stream_mode="values"):
            ...     print(event)
            {'alist': ['Ex for stream_mode="values"'], 'another_list': []}
            {'alist': ['Ex for stream_mode="values"'], 'another_list': ['hi']}
            {'alist': ['Ex for stream_mode="values"', 'there'], 'another_list': ['hi']}
            ```
            With stream_mode="updates":

            ```pycon
            >>> for event in graph.stream({"alist": ['Ex for stream_mode="updates"']}, stream_mode="updates"):
            ...     print(event)
            {'a': {'another_list': ['hi']}}
            {'b': {'alist': ['there']}}
            ```
            With stream_mode="debug":

            ```pycon
            >>> for event in graph.stream({"alist": ['Ex for stream_mode="debug"']}, stream_mode="debug"):
            ...     print(event)
            {'type': 'task', 'timestamp': '2024-06-23T...+00:00', 'step': 1, 'payload': {'id': '...', 'name': 'a', 'input': {'alist': ['Ex for stream_mode="debug"'], 'another_list': []}, 'triggers': ['start:a']}}
            {'type': 'task_result', 'timestamp': '2024-06-23T...+00:00', 'step': 1, 'payload': {'id': '...', 'name': 'a', 'result': [('another_list', ['hi'])]}}
            {'type': 'task', 'timestamp': '2024-06-23T...+00:00', 'step': 2, 'payload': {'id': '...', 'name': 'b', 'input': {'alist': ['Ex for stream_mode="debug"'], 'another_list': ['hi']}, 'triggers': ['a']}}
            {'type': 'task_result', 'timestamp': '2024-06-23T...+00:00', 'step': 2, 'payload': {'id': '...', 'name': 'b', 'result': [('alist', ['there'])]}}
            ```

            With stream_mode="custom":

            ```pycon
            >>> from langgraph.types import StreamWriter
            ...
            >>> def node_a(state: State, writer: StreamWriter):
            ...     writer({"custom_data": "foo"})
            ...     return {"alist": ["hi"]}
            ...
            >>> builder = StateGraph(State)
            >>> builder.add_node("a", node_a)
            >>> builder.add_edge(START, "a")
            >>> graph = builder.compile()
            ...
            >>> for event in graph.stream({"alist": ['Ex for stream_mode="custom"']}, stream_mode="custom"):
            ...     print(event)
            {'custom_data': 'foo'}
            ```
        """

        stream = SyncQueue()

        def output() -> Iterator:
            while True:
                try:
                    ns, mode, payload = stream.get(block=False)
                except queue.Empty:
                    break
                if mode in log_modes:
                    print((ns, mode, payload))
                if mode not in stream_modes:
                    continue
                if subgraphs and isinstance(stream_mode, list):
                    yield (ns, mode, payload)
                elif isinstance(stream_mode, list):
                    yield (mode, payload)
                elif subgraphs:
                    yield (ns, payload)
                else:
                    yield payload

        config = ensure_config(self.config, config)
        runtree = get_runtree_for_config(
            config, input, name=config.get("run_name", self.get_name())
        )
        config["run_tree"] = runtree
        # assign defaults
        (
            stream_modes,
            log_modes,
            output_keys,
            interrupt_before_,
            interrupt_after_,
            checkpointer,
            store,
        ) = self._defaults(
            config,
            stream_mode=stream_mode,
            log_mode=log_mode,
            output_keys=output_keys,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )
        all_modes = stream_modes.union(log_modes)
        # set up subgraph checkpointing
        if self.checkpointer is True:
            ns = cast(str, config[CONF][CONFIG_KEY_CHECKPOINT_NS])
            config[CONF][CONFIG_KEY_CHECKPOINT_NS] = recast_checkpoint_ns(ns)
        # set up custom stream mode
        if "custom" in all_modes:
            config[CONF][CONFIG_KEY_STREAM_WRITER] = lambda c: stream.put(
                ((), "custom", c)
            )
        with SyncPregelLoop(
            input,
            stream=StreamProtocol(stream.put, all_modes),
            config=config,
            store=store,
            checkpointer=checkpointer,
            nodes=self.nodes,
            specs=self.channels,
            output_keys=output_keys,
            stream_keys=self.stream_channels_asis,
            interrupt_before=interrupt_before_,
            interrupt_after=interrupt_after_,
        ) as loop:
            # create runner
            runner = PregelRunner(
                submit=config[CONF].get(CONFIG_KEY_RUNNER_SUBMIT, loop.submit),
                put_writes=loop.put_writes,
                node_finished=config[CONF].get(CONFIG_KEY_NODE_FINISHED),
            )
            # enable subgraph streaming
            if subgraphs:
                loop.config[CONF][CONFIG_KEY_STREAM] = loop.stream
            # enable concurrent streaming
            if self.stream_eager or subgraphs or "custom" in all_modes:
                # we are careful to have a single waiter live at any one time
                # because on exit we increment semaphore count by exactly 1
                waiter: Optional[concurrent.futures.Future] = None
                # because sync futures cannot be cancelled, we instead
                # release the stream semaphore on exit, which will cause
                # a pending waiter to return immediately
                loop.stack.callback(stream._count.release)

                def get_waiter() -> concurrent.futures.Future[None]:
                    nonlocal waiter
                    if waiter is None or waiter.done():
                        waiter = loop.submit(stream.wait)
                        return waiter
                    else:
                        return waiter

            else:
                get_waiter = None  # type: ignore[assignment]
            # Similarly to Bulk Synchronous Parallel / Pregel model
            # computation proceeds in steps, while there are channel updates.
            # Channel updates from step N are only visible in step N+1
            # channels are guaranteed to be immutable for the duration of the step,
            # with channel updates applied only at the transition between steps.
            while loop.tick(input_keys=self.input_channels):
                for _ in runner.tick(
                    loop.tasks.values(),
                    timeout=self.step_timeout,
                    retry_policy=self.retry_policy,
                    get_waiter=get_waiter,
                ):
                    # emit output
                    yield from output()
        # emit output
        yield from output()
        # handle exit
        if loop.status == "out_of_steps":
            msg = create_error_message(
                message=(
                    f"Recursion limit of {config['recursion_limit']} reached "
                    "without hitting a stop condition. You can increase the "
                    "limit by setting the `recursion_limit` config key."
                ),
                error_code=ErrorCode.GRAPH_RECURSION_LIMIT,
            )
            raise GraphRecursionError(msg)

    def invoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[AnyConfig] = None,
        *,
        stream_mode: StreamMode = "values",
        log_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        """Run the graph with a single input and config.

        Args:
            input: The input data for the graph. It can be a dictionary or any other type.
            config: Optional. The configuration for the graph run.
            stream_mode: Optional[str]. The stream mode for the graph run. Default is "values".
            output_keys: Optional. The output keys to retrieve from the graph run.
            interrupt_before: Optional. The nodes to interrupt the graph run before.
            interrupt_after: Optional. The nodes to interrupt the graph run after.
            **kwargs: Additional keyword arguments to pass to the graph run.

        Returns:
            The output of the graph run. If stream_mode is "values", it returns the latest output.
            If stream_mode is not "values", it returns a list of output chunks.
        """
        output_keys = output_keys if output_keys is not None else self.output_channels
        if stream_mode == "values":
            latest: Union[dict[str, Any], Any] = None
        else:
            chunks = []
        for chunk in self.stream(
            input,
            config,
            log_mode=log_mode,
            stream_mode=stream_mode,
            output_keys=output_keys,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            **kwargs,
        ):
            if stream_mode == "values":
                latest = chunk
            else:
                chunks.append(chunk)
        if stream_mode == "values":
            return latest
        else:
            return chunks
