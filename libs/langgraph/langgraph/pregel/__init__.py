from __future__ import annotations

import asyncio
import concurrent
import concurrent.futures
import queue
import weakref
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from functools import partial
from typing import Any, Callable, Generic, Union, cast, get_type_hints
from uuid import UUID, uuid5

from langchain_core.globals import get_debug
from langchain_core.runnables import (
    RunnableSequence,
)
from langchain_core.runnables.base import Input, Output
from langchain_core.runnables.config import (
    RunnableConfig,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
)
from langchain_core.runnables.graph import Graph
from pydantic import BaseModel
from typing_extensions import Self

from langgraph.cache.base import BaseCache
from langgraph.channels.base import BaseChannel
from langgraph.channels.topic import Topic
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointTuple,
)
from langgraph.config import get_config
from langgraph.constants import (
    CACHE_NS_WRITES,
    CONF,
    CONFIG_KEY_CACHE,
    CONFIG_KEY_CHECKPOINT_DURING,
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
    CONFIG_KEY_THREAD_ID,
    END,
    ERROR,
    INPUT,
    INTERRUPT,
    NS_END,
    NS_SEP,
    NULL_TASK_ID,
    PUSH,
    TASKS,
)
from langgraph.errors import (
    ErrorCode,
    GraphRecursionError,
    InvalidUpdateError,
    create_error_message,
)
from langgraph.managed.base import ManagedValueSpec
from langgraph.pregel.algo import (
    PregelTaskWrites,
    _scratchpad,
    apply_writes,
    local_read,
    prepare_next_tasks,
)
from langgraph.pregel.call import identifier
from langgraph.pregel.checkpoint import (
    channels_from_checkpoint,
    copy_checkpoint,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.pregel.debug import get_bolded_text, get_colored_text, tasks_w_writes
from langgraph.pregel.draw import draw_graph
from langgraph.pregel.io import map_input, read_channels
from langgraph.pregel.loop import AsyncPregelLoop, StreamProtocol, SyncPregelLoop
from langgraph.pregel.messages import StreamMessagesHandler
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
    CachePolicy,
    Checkpointer,
    Interrupt,
    Send,
    StateSnapshot,
    StateUpdate,
    StreamChunk,
    StreamMode,
)
from langgraph.typing import InputT, OutputT, StateT
from langgraph.utils.config import (
    ensure_config,
    merge_configs,
    patch_checkpoint_map,
    patch_config,
    patch_configurable,
    recast_checkpoint_ns,
)
from langgraph.utils.pydantic import create_model
from langgraph.utils.queue import AsyncQueue, SyncQueue  # type: ignore[attr-defined]
from langgraph.utils.runnable import (
    Runnable,
    RunnableLike,
    RunnableSeq,
    coerce_to_runnable,
)

try:
    from langchain_core.tracers._streaming import _StreamingCallbackHandler
except ImportError:
    _StreamingCallbackHandler = None  # type: ignore

WriteValue = Union[Callable[[Input], Output], Any]


class NodeBuilder:
    __slots__ = (
        "_channels",
        "_triggers",
        "_tags",
        "_metadata",
        "_writes",
        "_bound",
        "_retry_policy",
        "_cache_policy",
    )

    _channels: str | list[str]
    _triggers: list[str]
    _tags: list[str]
    _metadata: dict[str, Any]
    _writes: list[ChannelWriteEntry]
    _bound: Runnable
    _retry_policy: list[RetryPolicy]
    _cache_policy: CachePolicy | None

    def __init__(
        self,
    ) -> None:
        self._channels = []
        self._triggers = []
        self._tags = []
        self._metadata = {}
        self._writes = []
        self._bound = DEFAULT_BOUND
        self._retry_policy = []
        self._cache_policy = None

    def subscribe_only(
        self,
        channel: str,
    ) -> Self:
        """Subscribe to a single channel."""
        if not self._channels:
            self._channels = channel
        else:
            raise ValueError(
                "Cannot subscribe to single channels when other channels are already subscribed to"
            )

        self._triggers.append(channel)

        return self

    def subscribe_to(
        self,
        *channels: str,
        read: bool = True,
    ) -> Self:
        """Add channels to subscribe to. Node will be invoked when any of these
        channels are updated, with a dict of the channel values as input.

        Args:
            channels: Channel name(s) to subscribe to
            read: If True, the channels will be included in the input to the node.
                Otherwise, they will trigger the node without being sent in input.

        Returns:
            Self for chaining
        """
        if isinstance(self._channels, str):
            raise ValueError(
                "Cannot subscribe to channels when subscribed to a single channel"
            )
        if read:
            if not self._channels:
                self._channels = list(channels)
            else:
                self._channels.extend(channels)

        if isinstance(channels, str):
            self._triggers.append(channels)
        else:
            self._triggers.extend(channels)

        return self

    def read_from(
        self,
        *channels: str,
    ) -> Self:
        """Adds the specified channels to read from, without subscribing to them."""
        assert isinstance(self._channels, list), (
            "Cannot read additional channels when subscribed to single channels"
        )
        self._channels.extend(channels)
        return self

    def do(
        self,
        node: RunnableLike,
    ) -> Self:
        """Adds the specified node."""
        if self._bound is not DEFAULT_BOUND:
            self._bound = RunnableSeq(
                self._bound, coerce_to_runnable(node, name=None, trace=True)
            )
        else:
            self._bound = coerce_to_runnable(node, name=None, trace=True)
        return self

    def write_to(
        self,
        *channels: str | ChannelWriteEntry,
        **kwargs: WriteValue,
    ) -> Self:
        """Add channel writes.

        Args:
            *channels: Channel names to write to
            **kwargs: Channel name and value mappings

        Returns:
            Self for chaining
        """
        self._writes.extend(
            ChannelWriteEntry(c) if isinstance(c, str) else c for c in channels
        )
        self._writes.extend(
            ChannelWriteEntry(k, mapper=v)
            if callable(v)
            else ChannelWriteEntry(k, value=v)
            for k, v in kwargs.items()
        )

        return self

    def meta(self, *tags: str, **metadata: Any) -> Self:
        """Add tags or metadata to the node."""
        self._tags.extend(tags)
        self._metadata.update(metadata)
        return self

    def add_retry_policies(self, *policies: RetryPolicy) -> Self:
        """Adds retry policies to the node."""
        self._retry_policy.extend(policies)
        return self

    def add_cache_policy(self, policy: CachePolicy) -> Self:
        """Adds cache policies to the node."""
        self._cache_policy = policy
        return self

    def build(self) -> PregelNode:
        """Builds the node."""
        return PregelNode(
            channels=self._channels,
            triggers=self._triggers,
            tags=self._tags,
            metadata=self._metadata,
            writers=[ChannelWrite(self._writes)],
            bound=self._bound,
            retry_policy=self._retry_policy,
            cache_policy=self._cache_policy,
        )


class Pregel(PregelProtocol[StateT, InputT, OutputT], Generic[StateT, InputT, OutputT]):
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

    An **actor** is a `PregelNode`.
    It subscribes to channels, reads data from them, and writes data to them.
    It can be thought of as an **actor** in the Pregel algorithm.
    `PregelNodes` implement LangChain's
    Runnable interface.

    ## Channels

    Channels are used to communicate between actors (`PregelNodes`).
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
        from langgraph.pregel import Pregel, NodeBuilder

        node1 = (
            NodeBuilder().subscribe_only("a")
            .do(lambda x: x + x)
            .write_to("b")
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
        from langgraph.pregel import Pregel, NodeBuilder

        node1 = (
            NodeBuilder().subscribe_only("a")
            .do(lambda x: x + x)
            .write_to("b")
        )

        node2 = (
            NodeBuilder().subscribe_to("b")
            .do(lambda x: x["b"] + x["b"])
            .write_to("c")
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
        from langgraph.pregel import Pregel, NodeBuilder

        node1 = (
            NodeBuilder().subscribe_only("a")
            .do(lambda x: x + x)
            .write_to("b", "c")
        )

        node2 = (
            NodeBuilder().subscribe_only("b")
            .do(lambda x: x + x)
            .write_to("c")
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
        from langgraph.pregel import Pregel, NodeBuilder


        node1 = (
            NodeBuilder().subscribe_only("a")
            .do(lambda x: x + x)
            .write_to("b", "c")
        )

        node2 = (
            NodeBuilder().subscribe_only("b")
            .do(lambda x: x + x)
            .write_to("c")
        )


        def reducer(current, update):
            if current:
                return current + " | " + update
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
        from langgraph.pregel import Pregel, NodeBuilder, ChannelWriteEntry

        example_node = (
            NodeBuilder().subscribe_only("value")
            .do(lambda x: x + x if len(x) < 10 else None)
            .write_to(ChannelWriteEntry(channel="value", skip_none=True))
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

    channels: dict[str, BaseChannel | ManagedValueSpec]

    stream_mode: StreamMode = "values"
    """Mode to stream output, defaults to 'values'."""

    stream_eager: bool = False
    """Whether to force emitting stream events eagerly, automatically turned on
    for stream_mode "messages" and "custom"."""

    output_channels: str | Sequence[str]

    stream_channels: str | Sequence[str] | None = None
    """Channels to stream, defaults to all channels not in reserved channels"""

    interrupt_after_nodes: All | Sequence[str]

    interrupt_before_nodes: All | Sequence[str]

    input_channels: str | Sequence[str]

    step_timeout: float | None = None
    """Maximum time to wait for a step to complete, in seconds. Defaults to None."""

    debug: bool
    """Whether to print debug information during execution. Defaults to False."""

    checkpointer: Checkpointer = None
    """Checkpointer used to save and load graph state. Defaults to None."""

    store: BaseStore | None = None
    """Memory store to use for SharedValues. Defaults to None."""

    cache: BaseCache | None = None
    """Cache to use for storing node results. Defaults to None."""

    retry_policy: Sequence[RetryPolicy] = ()
    """Retry policies to use when running tasks. Empty set disables retries."""

    cache_policy: CachePolicy | None = None
    """Cache policy to use for all nodes. Can be overridden by individual nodes.
    Defaults to None."""

    config_type: type[Any] | None = None

    config: RunnableConfig | None = None

    name: str = "LangGraph"

    trigger_to_nodes: Mapping[str, Sequence[str]]

    def __init__(
        self,
        *,
        nodes: dict[str, PregelNode | NodeBuilder],
        channels: dict[str, BaseChannel | ManagedValueSpec] | None,
        auto_validate: bool = True,
        stream_mode: StreamMode = "values",
        stream_eager: bool = False,
        output_channels: str | Sequence[str],
        stream_channels: str | Sequence[str] | None = None,
        interrupt_after_nodes: All | Sequence[str] = (),
        interrupt_before_nodes: All | Sequence[str] = (),
        input_channels: str | Sequence[str],
        step_timeout: float | None = None,
        debug: bool | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        cache: BaseCache | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        config_type: type[Any] | None = None,
        config: RunnableConfig | None = None,
        trigger_to_nodes: Mapping[str, Sequence[str]] | None = None,
        name: str = "LangGraph",
    ) -> None:
        self.nodes = {
            k: v.build() if isinstance(v, NodeBuilder) else v for k, v in nodes.items()
        }
        self.channels = channels or {}
        if TASKS in self.channels and not isinstance(self.channels[TASKS], Topic):
            raise ValueError(
                f"Channel '{TASKS}' is reserved and cannot be used in the graph."
            )
        else:
            self.channels[TASKS] = Topic(Send, accumulate=False)
        self.stream_mode = stream_mode
        self.stream_eager = stream_eager
        self.output_channels = output_channels
        self.stream_channels = stream_channels
        self.interrupt_after_nodes = interrupt_after_nodes
        self.interrupt_before_nodes = interrupt_before_nodes
        self.input_channels = input_channels
        self.step_timeout = step_timeout
        self.debug = debug if debug is not None else get_debug()
        self.checkpointer = checkpointer
        self.store = store
        self.cache = cache
        self.retry_policy = (
            (retry_policy,) if isinstance(retry_policy, RetryPolicy) else retry_policy
        )
        self.cache_policy = cache_policy
        self.config_type = config_type
        self.config = config
        self.trigger_to_nodes = trigger_to_nodes or {}
        self.name = name
        if auto_validate:
            self.validate()

    def get_graph(
        self, config: RunnableConfig | None = None, *, xray: int | bool = False
    ) -> Graph:
        """Return a drawable representation of the computation graph."""
        # gather subgraphs
        if xray:
            subgraphs = {
                k: v.get_graph(
                    config,
                    xray=xray if isinstance(xray, bool) or xray <= 0 else xray - 1,
                )
                for k, v in self.get_subgraphs()
            }
        else:
            subgraphs = {}

        return draw_graph(
            merge_configs(self.config, config),
            nodes=self.nodes,
            specs=self.channels,
            input_channels=self.input_channels,
            interrupt_after_nodes=self.interrupt_after_nodes,
            interrupt_before_nodes=self.interrupt_before_nodes,
            trigger_to_nodes=self.trigger_to_nodes,
            checkpointer=self.checkpointer,
            subgraphs=subgraphs,
        )

    async def aget_graph(
        self, config: RunnableConfig | None = None, *, xray: int | bool = False
    ) -> Graph:
        """Return a drawable representation of the computation graph."""

        # gather subgraphs
        if xray:
            subpregels: dict[str, PregelProtocol] = {
                k: v async for k, v in self.aget_subgraphs()
            }
            subgraphs = {
                k: v
                for k, v in zip(
                    subpregels,
                    await asyncio.gather(
                        *(
                            p.aget_graph(
                                config,
                                xray=xray
                                if isinstance(xray, bool) or xray <= 0
                                else xray - 1,
                            )
                            for p in subpregels.values()
                        )
                    ),
                )
            }
        else:
            subgraphs = {}

        return draw_graph(
            merge_configs(self.config, config),
            nodes=self.nodes,
            specs=self.channels,
            input_channels=self.input_channels,
            interrupt_after_nodes=self.interrupt_after_nodes,
            interrupt_before_nodes=self.interrupt_before_nodes,
            trigger_to_nodes=self.trigger_to_nodes,
            checkpointer=self.checkpointer,
            subgraphs=subgraphs,
        )

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any]:
        """Mime bundle used by Jupyter to display the graph"""
        return {
            "text/plain": repr(self),
            "image/png": self.get_graph().draw_mermaid_png(),
        }

    def copy(self, update: dict[str, Any] | None = None) -> Self:
        attrs = {k: v for k, v in self.__dict__.items() if k != "__orig_class__"}
        attrs.update(update or {})
        return self.__class__(**attrs)

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self:
        """Create a copy of the Pregel object with an updated config."""
        return self.copy(
            {"config": merge_configs(self.config, config, cast(RunnableConfig, kwargs))}
        )

    def validate(self) -> Self:
        validate_graph(
            self.nodes,
            {k: v for k, v in self.channels.items() if isinstance(v, BaseChannel)},
            {k: v for k, v in self.channels.items() if not isinstance(v, BaseChannel)},
            self.input_channels,
            self.output_channels,
            self.stream_channels,
            self.interrupt_after_nodes,
            self.interrupt_before_nodes,
        )
        self.trigger_to_nodes = _trigger_to_nodes(self.nodes)
        return self

    def config_schema(self, *, include: Sequence[str] | None = None) -> type[BaseModel]:
        include = include or []
        fields = {
            **({"configurable": (self.config_type, None)} if self.config_type else {}),
            **{
                field_name: (field_type, None)
                for field_name, field_type in get_type_hints(RunnableConfig).items()
                if field_name in [i for i in include if i != "configurable"]
            },
        }
        return create_model(self.get_name("Config"), field_definitions=fields)

    def get_config_jsonschema(
        self, *, include: Sequence[str] | None = None
    ) -> dict[str, Any]:
        schema = self.config_schema(include=include)
        return schema.model_json_schema()

    @property
    def InputType(self) -> Any:
        if isinstance(self.input_channels, str):
            channel = self.channels[self.input_channels]
            if isinstance(channel, BaseChannel):
                return channel.UpdateType

    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        config = merge_configs(self.config, config)
        if isinstance(self.input_channels, str):
            return super().get_input_schema(config)
        else:
            return create_model(
                self.get_name("Input"),
                field_definitions={
                    k: (c.UpdateType, None)
                    for k in self.input_channels or self.channels.keys()
                    if (c := self.channels[k]) and isinstance(c, BaseChannel)
                },
            )

    def get_input_jsonschema(
        self, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        schema = self.get_input_schema(config)
        return schema.model_json_schema()

    @property
    def OutputType(self) -> Any:
        if isinstance(self.output_channels, str):
            channel = self.channels[self.output_channels]
            if isinstance(channel, BaseChannel):
                return channel.ValueType

    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        config = merge_configs(self.config, config)
        if isinstance(self.output_channels, str):
            return super().get_output_schema(config)
        else:
            return create_model(
                self.get_name("Output"),
                field_definitions={
                    k: (c.ValueType, None)
                    for k in self.output_channels
                    if (c := self.channels[k]) and isinstance(c, BaseChannel)
                },
            )

    def get_output_jsonschema(
        self, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        schema = self.get_output_schema(config)
        return schema.model_json_schema()

    @property
    def stream_channels_list(self) -> Sequence[str]:
        stream_channels = self.stream_channels_asis
        return (
            [stream_channels] if isinstance(stream_channels, str) else stream_channels
        )

    @property
    def stream_channels_asis(self) -> str | Sequence[str]:
        return self.stream_channels or [
            k for k in self.channels if isinstance(self.channels[k], BaseChannel)
        ]

    def get_subgraphs(
        self, *, namespace: str | None = None, recurse: bool = False
    ) -> Iterator[tuple[str, PregelProtocol]]:
        """Get the subgraphs of the graph.

        Args:
            namespace: The namespace to filter the subgraphs by.
            recurse: Whether to recurse into the subgraphs.
                If False, only the immediate subgraphs will be returned.

        Returns:
            Iterator[tuple[str, PregelProtocol]]: An iterator of the (namespace, subgraph) pairs.
        """
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

    async def aget_subgraphs(
        self, *, namespace: str | None = None, recurse: bool = False
    ) -> AsyncIterator[tuple[str, PregelProtocol]]:
        """Get the subgraphs of the graph.

        Args:
            namespace: The namespace to filter the subgraphs by.
            recurse: Whether to recurse into the subgraphs.
                If False, only the immediate subgraphs will be returned.

        Returns:
            AsyncIterator[tuple[str, PregelProtocol]]: An iterator of the (namespace, subgraph) pairs.
        """
        for name, node in self.get_subgraphs(namespace=namespace, recurse=recurse):
            yield name, node

    def _migrate_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Migrate a saved checkpoint to new channel layout."""
        if checkpoint["v"] < 4 and checkpoint.get("pending_sends"):
            pending_sends: list[Send] = checkpoint.pop("pending_sends")
            checkpoint["channel_values"][TASKS] = pending_sends
            checkpoint["channel_versions"][TASKS] = max(
                checkpoint["channel_versions"].values()
            )

    def _prepare_state_snapshot(
        self,
        config: RunnableConfig,
        saved: CheckpointTuple | None,
        recurse: BaseCheckpointSaver | None = None,
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
                interrupts=(),
            )

        # migrate checkpoint if needed
        self._migrate_checkpoint(saved.checkpoint)

        step = saved.metadata.get("step", -1) + 1
        stop = step + 2
        channels, managed = channels_from_checkpoint(
            self.channels,
            saved.checkpoint,
        )
        # tasks for this checkpoint
        next_tasks = prepare_next_tasks(
            saved.checkpoint,
            saved.pending_writes or [],
            self.nodes,
            channels,
            managed,
            saved.config,
            step,
            stop,
            for_execution=True,
            store=self.store,
            checkpointer=(
                self.checkpointer
                if isinstance(self.checkpointer, BaseCheckpointSaver)
                else None
            ),
            manager=None,
        )
        # get the subgraphs
        subgraphs = dict(self.get_subgraphs())
        parent_ns = saved.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        task_states: dict[str, RunnableConfig | StateSnapshot] = {}
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
                self.trigger_to_nodes,
            )
        if apply_pending_writes and saved.pending_writes:
            for tid, k, v in saved.pending_writes:
                if k in (ERROR, INTERRUPT):
                    continue
                if tid not in next_tasks:
                    continue
                next_tasks[tid].writes.append((k, v))
            if tasks := [t for t in next_tasks.values() if t.writes]:
                apply_writes(
                    saved.checkpoint, channels, tasks, None, self.trigger_to_nodes
                )
        tasks_with_writes = tasks_w_writes(
            next_tasks.values(),
            saved.pending_writes,
            task_states,
            self.stream_channels_asis,
        )
        # assemble the state snapshot
        return StateSnapshot(
            read_channels(channels, self.stream_channels_asis),
            tuple(t.name for t in next_tasks.values() if not t.writes),
            patch_checkpoint_map(saved.config, saved.metadata),
            saved.metadata,
            saved.checkpoint["ts"],
            patch_checkpoint_map(saved.parent_config, saved.metadata),
            tasks_with_writes,
            tuple([i for task in tasks_with_writes for i in task.interrupts]),
        )

    async def _aprepare_state_snapshot(
        self,
        config: RunnableConfig,
        saved: CheckpointTuple | None,
        recurse: BaseCheckpointSaver | None = None,
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
                interrupts=(),
            )

        # migrate checkpoint if needed
        self._migrate_checkpoint(saved.checkpoint)

        step = saved.metadata.get("step", -1) + 1
        stop = step + 2
        channels, managed = channels_from_checkpoint(
            self.channels,
            saved.checkpoint,
        )
        # tasks for this checkpoint
        next_tasks = prepare_next_tasks(
            saved.checkpoint,
            saved.pending_writes or [],
            self.nodes,
            channels,
            managed,
            saved.config,
            step,
            stop,
            for_execution=True,
            store=self.store,
            checkpointer=(
                self.checkpointer
                if isinstance(self.checkpointer, BaseCheckpointSaver)
                else None
            ),
            manager=None,
        )
        # get the subgraphs
        subgraphs = {n: g async for n, g in self.aget_subgraphs()}
        parent_ns = saved.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        task_states: dict[str, RunnableConfig | StateSnapshot] = {}
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
                task_states[task.id] = await subgraphs[task.name].aget_state(
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
                self.trigger_to_nodes,
            )
        if apply_pending_writes and saved.pending_writes:
            for tid, k, v in saved.pending_writes:
                if k in (ERROR, INTERRUPT):
                    continue
                if tid not in next_tasks:
                    continue
                next_tasks[tid].writes.append((k, v))
            if tasks := [t for t in next_tasks.values() if t.writes]:
                apply_writes(
                    saved.checkpoint, channels, tasks, None, self.trigger_to_nodes
                )

        tasks_with_writes = tasks_w_writes(
            next_tasks.values(),
            saved.pending_writes,
            task_states,
            self.stream_channels_asis,
        )
        # assemble the state snapshot
        return StateSnapshot(
            read_channels(channels, self.stream_channels_asis),
            tuple(t.name for t in next_tasks.values() if not t.writes),
            patch_checkpoint_map(saved.config, saved.metadata),
            saved.metadata,
            saved.checkpoint["ts"],
            patch_checkpoint_map(saved.parent_config, saved.metadata),
            tasks_with_writes,
            tuple([i for task in tasks_with_writes for i in task.interrupts]),
        )

    def get_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        """Get the current state of the graph."""
        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
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
        thread_id = config[CONF][CONFIG_KEY_THREAD_ID]
        if not isinstance(thread_id, str):
            config[CONF][CONFIG_KEY_THREAD_ID] = str(thread_id)

        saved = checkpointer.get_tuple(config)
        return self._prepare_state_snapshot(
            config,
            saved,
            recurse=checkpointer if subgraphs else None,
            apply_pending_writes=CONFIG_KEY_CHECKPOINT_ID not in config[CONF],
        )

    async def aget_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        """Get the current state of the graph."""
        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
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
            async for _, pregel in self.aget_subgraphs(namespace=recast, recurse=True):
                return await pregel.aget_state(
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
        thread_id = config[CONF][CONFIG_KEY_THREAD_ID]
        if not isinstance(thread_id, str):
            config[CONF][CONFIG_KEY_THREAD_ID] = str(thread_id)

        saved = await checkpointer.aget_tuple(config)
        return await self._aprepare_state_snapshot(
            config,
            saved,
            recurse=checkpointer if subgraphs else None,
            apply_pending_writes=CONFIG_KEY_CHECKPOINT_ID not in config[CONF],
        )

    def get_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[StateSnapshot]:
        """Get the history of the state of the graph."""
        config = ensure_config(config)
        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
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
            {
                CONF: {
                    CONFIG_KEY_CHECKPOINT_NS: checkpoint_ns,
                    CONFIG_KEY_THREAD_ID: str(config[CONF][CONFIG_KEY_THREAD_ID]),
                }
            },
        )
        # eagerly consume list() to avoid holding up the db cursor
        for checkpoint_tuple in list(
            checkpointer.list(config, before=before, limit=limit, filter=filter)
        ):
            yield self._prepare_state_snapshot(
                checkpoint_tuple.config, checkpoint_tuple
            )

    async def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[StateSnapshot]:
        """Asynchronously get the history of the state of the graph."""
        config = ensure_config(config)
        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
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
            async for _, pregel in self.aget_subgraphs(namespace=recast, recurse=True):
                async for state in pregel.aget_state_history(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    filter=filter,
                    before=before,
                    limit=limit,
                ):
                    yield state
                return
            else:
                raise ValueError(f"Subgraph {recast} not found")

        config = merge_configs(
            self.config,
            config,
            {
                CONF: {
                    CONFIG_KEY_CHECKPOINT_NS: checkpoint_ns,
                    CONFIG_KEY_THREAD_ID: str(config[CONF][CONFIG_KEY_THREAD_ID]),
                }
            },
        )
        # eagerly consume list() to avoid holding up the db cursor
        for checkpoint_tuple in [
            c
            async for c in checkpointer.alist(
                config, before=before, limit=limit, filter=filter
            )
        ]:
            yield await self._aprepare_state_snapshot(
                checkpoint_tuple.config, checkpoint_tuple
            )

    def bulk_update_state(
        self,
        config: RunnableConfig,
        supersteps: Sequence[Sequence[StateUpdate]],
    ) -> RunnableConfig:
        """Apply updates to the graph state in bulk. Requires a checkpointer to be set.

        Args:
            config: The config to apply the updates to.
            supersteps: A list of supersteps, each including a list of updates to apply sequentially to a graph state.
                        Each update is a tuple of the form `(values, as_node, task_id)` where task_id is optional.

        Raises:
            ValueError: If no checkpointer is set or no updates are provided.
            InvalidUpdateError: If an invalid update is provided.

        Returns:
            RunnableConfig: The updated config.
        """

        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if len(supersteps) == 0:
            raise ValueError("No supersteps provided")

        if any(len(u) == 0 for u in supersteps):
            raise ValueError("No updates provided")

        # delegate to subgraph
        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast = recast_checkpoint_ns(checkpoint_ns)
            # find the subgraph with the matching name
            for _, pregel in self.get_subgraphs(namespace=recast, recurse=True):
                return pregel.bulk_update_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    supersteps,
                )
            else:
                raise ValueError(f"Subgraph {recast} not found")

        def perform_superstep(
            input_config: RunnableConfig, updates: Sequence[StateUpdate]
        ) -> RunnableConfig:
            # get last checkpoint
            config = ensure_config(self.config, input_config)
            saved = checkpointer.get_tuple(config)
            if saved is not None:
                self._migrate_checkpoint(saved.checkpoint)
            checkpoint = (
                copy_checkpoint(saved.checkpoint) if saved else empty_checkpoint()
            )
            checkpoint_previous_versions = (
                saved.checkpoint["channel_versions"].copy() if saved else {}
            )
            step = saved.metadata.get("step", -1) if saved else -1
            # merge configurable fields with previous checkpoint config
            checkpoint_config = patch_configurable(
                config,
                {
                    CONFIG_KEY_CHECKPOINT_NS: config[CONF].get(
                        CONFIG_KEY_CHECKPOINT_NS, ""
                    )
                },
            )
            if saved:
                checkpoint_config = patch_configurable(config, saved.config[CONF])
            channels, managed = channels_from_checkpoint(
                self.channels,
                checkpoint,
            )
            values, as_node = updates[0][:2]

            # no values as END, just clear all tasks
            if values is None and as_node == END:
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot apply multiple updates when clearing state"
                    )

                if saved is not None:
                    # tasks for this checkpoint
                    next_tasks = prepare_next_tasks(
                        checkpoint,
                        saved.pending_writes or [],
                        self.nodes,
                        channels,
                        managed,
                        saved.config,
                        step + 1,
                        step + 3,
                        for_execution=True,
                        store=self.store,
                        checkpointer=checkpointer,
                        manager=None,
                    )
                    # apply null writes
                    if null_writes := [
                        w[1:]
                        for w in saved.pending_writes or []
                        if w[0] == NULL_TASK_ID
                    ]:
                        apply_writes(
                            checkpoint,
                            channels,
                            [PregelTaskWrites((), INPUT, null_writes, [])],
                            checkpointer.get_next_version,
                            self.trigger_to_nodes,
                        )
                    # apply writes from tasks that already ran
                    for tid, k, v in saved.pending_writes or []:
                        if k in (ERROR, INTERRUPT):
                            continue
                        if tid not in next_tasks:
                            continue
                        next_tasks[tid].writes.append((k, v))
                    # clear all current tasks
                    apply_writes(
                        checkpoint,
                        channels,
                        next_tasks.values(),
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
                # save checkpoint
                next_config = checkpointer.put(
                    checkpoint_config,
                    create_checkpoint(checkpoint, channels, step),
                    {
                        "source": "update",
                        "step": step + 1,
                        "parents": saved.metadata.get("parents", {}) if saved else {},
                    },
                    get_new_channel_versions(
                        checkpoint_previous_versions,
                        checkpoint["channel_versions"],
                    ),
                )
                return patch_checkpoint_map(
                    next_config, saved.metadata if saved else None
                )

            # act as an input
            if as_node == INPUT:
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot apply multiple updates when updating as input"
                    )

                if input_writes := deque(map_input(self.input_channels, values)):
                    apply_writes(
                        checkpoint,
                        channels,
                        [PregelTaskWrites((), INPUT, input_writes, [])],
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )

                    # apply input write to channels
                    next_step = (
                        step + 1
                        if saved and saved.metadata.get("step") is not None
                        else -1
                    )
                    next_config = checkpointer.put(
                        checkpoint_config,
                        create_checkpoint(checkpoint, channels, next_step),
                        {
                            "source": "input",
                            "step": next_step,
                            "parents": saved.metadata.get("parents", {})
                            if saved
                            else {},
                        },
                        get_new_channel_versions(
                            checkpoint_previous_versions,
                            checkpoint["channel_versions"],
                        ),
                    )

                    # store the writes
                    checkpointer.put_writes(
                        next_config,
                        input_writes,
                        str(uuid5(UUID(checkpoint["id"]), INPUT)),
                    )

                    return patch_checkpoint_map(
                        next_config, saved.metadata if saved else None
                    )
                else:
                    raise InvalidUpdateError(
                        f"Received no input writes for {self.input_channels}"
                    )

            # copy checkpoint
            if as_node == "__copy__":
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot copy checkpoint with multiple updates"
                    )

                if saved is None:
                    raise InvalidUpdateError("Cannot copy a non-existent checkpoint")

                next_checkpoint = create_checkpoint(checkpoint, None, step)

                # copy checkpoint
                next_config = checkpointer.put(
                    saved.parent_config
                    or patch_configurable(
                        saved.config, {CONFIG_KEY_CHECKPOINT_ID: None}
                    ),
                    next_checkpoint,
                    {
                        "source": "fork",
                        "step": step + 1,
                        "parents": saved.metadata.get("parents", {}),
                    },
                    {},
                )

                # we want to both clone a checkpoint and update state in one go.
                # reuse the same task ID if possible.
                if isinstance(values, list) and len(values) > 0:
                    # figure out the task IDs for the next update checkpoint
                    next_tasks = prepare_next_tasks(
                        next_checkpoint,
                        saved.pending_writes or [],
                        self.nodes,
                        channels,
                        managed,
                        next_config,
                        step + 2,
                        step + 4,
                        for_execution=True,
                        store=self.store,
                        checkpointer=checkpointer,
                        manager=None,
                    )

                    tasks_group_by = defaultdict(list)
                    user_group_by: dict[str, list[StateUpdate]] = defaultdict(list)

                    for task in next_tasks.values():
                        tasks_group_by[task.name].append(task.id)

                    for item in values:
                        if not isinstance(item, Sequence):
                            raise InvalidUpdateError(
                                f"Invalid update item: {item} when copying checkpoint"
                            )

                        values, as_node = item[:2]

                        user_group = user_group_by[as_node]
                        tasks_group = tasks_group_by[as_node]

                        target_idx = len(user_group)
                        task_id = (
                            tasks_group[target_idx]
                            if target_idx < len(tasks_group)
                            else None
                        )

                        user_group_by[as_node].append(
                            StateUpdate(values=values, as_node=as_node, task_id=task_id)
                        )

                    return perform_superstep(
                        patch_checkpoint_map(next_config, saved.metadata),
                        [item for lst in user_group_by.values() for item in lst],
                    )

                return patch_checkpoint_map(next_config, saved.metadata)

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
                    managed,
                    saved.config,
                    step + 1,
                    step + 3,
                    for_execution=True,
                    store=self.store,
                    checkpointer=checkpointer,
                    manager=None,
                )
                # apply null writes
                if null_writes := [
                    w[1:] for w in saved.pending_writes or [] if w[0] == NULL_TASK_ID
                ]:
                    apply_writes(
                        checkpoint,
                        channels,
                        [PregelTaskWrites((), INPUT, null_writes, [])],
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
                # apply writes
                for tid, k, v in saved.pending_writes:
                    if k in (ERROR, INTERRUPT):
                        continue
                    if tid not in next_tasks:
                        continue
                    next_tasks[tid].writes.append((k, v))
                if tasks := [t for t in next_tasks.values() if t.writes]:
                    apply_writes(
                        checkpoint,
                        channels,
                        tasks,
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
            valid_updates: list[tuple[str, dict[str, Any] | None, str | None]] = []
            if len(updates) == 1:
                values, as_node, task_id = updates[0]
                # find last node that updated the state, if not provided
                if as_node is None and len(self.nodes) == 1:
                    as_node = tuple(self.nodes)[0]
                elif as_node is None and not any(
                    v
                    for vv in checkpoint["versions_seen"].values()
                    for v in vv.values()
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
                valid_updates.append((as_node, values, task_id))
            else:
                for values, as_node, task_id in updates:
                    if as_node is None:
                        raise InvalidUpdateError(
                            "as_node is required when applying multiple updates"
                        )
                    if as_node not in self.nodes:
                        raise InvalidUpdateError(f"Node {as_node} does not exist")

                    valid_updates.append((as_node, values, task_id))

            run_tasks: list[PregelTaskWrites] = []
            run_task_ids: list[str] = []

            for as_node, values, provided_task_id in valid_updates:
                # create task to run all writers of the chosen node
                writers = self.nodes[as_node].flat_writers
                if not writers:
                    raise InvalidUpdateError(f"Node {as_node} has no writers")
                writes: deque[tuple[str, Any]] = deque()
                task = PregelTaskWrites((), as_node, writes, [INTERRUPT])
                task_id = provided_task_id or str(
                    uuid5(UUID(checkpoint["id"]), INTERRUPT)
                )
                run_tasks.append(task)
                run_task_ids.append(task_id)
                run = RunnableSequence(*writers) if len(writers) > 1 else writers[0]
                # execute task
                run.invoke(
                    values,
                    patch_config(
                        config,
                        run_name=self.name + "UpdateState",
                        configurable={
                            # deque.extend is thread-safe
                            CONFIG_KEY_SEND: writes.extend,
                            CONFIG_KEY_TASK_ID: task_id,
                            CONFIG_KEY_READ: partial(
                                local_read,
                                _scratchpad(
                                    None,
                                    [],
                                    task_id,
                                    "",
                                    None,
                                    step,
                                    step + 2,
                                ),
                                channels,
                                managed,
                                task,
                            ),
                        },
                    ),
                )
            # save task writes
            for task_id, task in zip(run_task_ids, run_tasks):
                # channel writes are saved to current checkpoint
                channel_writes = [w for w in task.writes if w[0] != PUSH]
                if saved and channel_writes:
                    checkpointer.put_writes(checkpoint_config, channel_writes, task_id)
            # apply to checkpoint and save
            apply_writes(
                checkpoint,
                channels,
                run_tasks,
                checkpointer.get_next_version,
                self.trigger_to_nodes,
            )
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            next_config = checkpointer.put(
                checkpoint_config,
                checkpoint,
                {
                    "source": "update",
                    "step": step + 1,
                    "parents": saved.metadata.get("parents", {}) if saved else {},
                },
                get_new_channel_versions(
                    checkpoint_previous_versions, checkpoint["channel_versions"]
                ),
            )
            for task_id, task in zip(run_task_ids, run_tasks):
                # save push writes
                if push_writes := [w for w in task.writes if w[0] == PUSH]:
                    checkpointer.put_writes(next_config, push_writes, task_id)

            return patch_checkpoint_map(next_config, saved.metadata if saved else None)

        current_config = patch_configurable(
            config, {CONFIG_KEY_THREAD_ID: str(config[CONF][CONFIG_KEY_THREAD_ID])}
        )
        for superstep in supersteps:
            current_config = perform_superstep(current_config, superstep)
        return current_config

    async def abulk_update_state(
        self,
        config: RunnableConfig,
        supersteps: Sequence[Sequence[StateUpdate]],
    ) -> RunnableConfig:
        """Asynchronously apply updates to the graph state in bulk. Requires a checkpointer to be set.

        Args:
            config: The config to apply the updates to.
            supersteps: A list of supersteps, each including a list of updates to apply sequentially to a graph state.
                        Each update is a tuple of the form `(values, as_node, task_id)` where task_id is optional.

        Raises:
            ValueError: If no checkpointer is set or no updates are provided.
            InvalidUpdateError: If an invalid update is provided.

        Returns:
            RunnableConfig: The updated config.
        """

        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if len(supersteps) == 0:
            raise ValueError("No supersteps provided")

        if any(len(u) == 0 for u in supersteps):
            raise ValueError("No updates provided")

        # delegate to subgraph
        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast = recast_checkpoint_ns(checkpoint_ns)
            # find the subgraph with the matching name
            async for _, pregel in self.aget_subgraphs(namespace=recast, recurse=True):
                return await pregel.abulk_update_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    supersteps,
                )
            else:
                raise ValueError(f"Subgraph {recast} not found")

        async def aperform_superstep(
            input_config: RunnableConfig, updates: Sequence[StateUpdate]
        ) -> RunnableConfig:
            # get last checkpoint
            config = ensure_config(self.config, input_config)
            saved = await checkpointer.aget_tuple(config)
            if saved is not None:
                self._migrate_checkpoint(saved.checkpoint)
            checkpoint = (
                copy_checkpoint(saved.checkpoint) if saved else empty_checkpoint()
            )
            checkpoint_previous_versions = (
                saved.checkpoint["channel_versions"].copy() if saved else {}
            )
            step = saved.metadata.get("step", -1) if saved else -1
            # merge configurable fields with previous checkpoint config
            checkpoint_config = patch_configurable(
                config,
                {
                    CONFIG_KEY_CHECKPOINT_NS: config[CONF].get(
                        CONFIG_KEY_CHECKPOINT_NS, ""
                    )
                },
            )
            if saved:
                checkpoint_config = patch_configurable(config, saved.config[CONF])
            channels, managed = channels_from_checkpoint(
                self.channels,
                checkpoint,
            )
            values, as_node = updates[0][:2]
            # no values, just clear all tasks
            if values is None and as_node == END:
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot apply multiple updates when clearing state"
                    )
                if saved is not None:
                    # tasks for this checkpoint
                    next_tasks = prepare_next_tasks(
                        checkpoint,
                        saved.pending_writes or [],
                        self.nodes,
                        channels,
                        managed,
                        saved.config,
                        step + 1,
                        step + 3,
                        for_execution=True,
                        store=self.store,
                        checkpointer=checkpointer,
                        manager=None,
                    )
                    # apply null writes
                    if null_writes := [
                        w[1:]
                        for w in saved.pending_writes or []
                        if w[0] == NULL_TASK_ID
                    ]:
                        apply_writes(
                            checkpoint,
                            channels,
                            [PregelTaskWrites((), INPUT, null_writes, [])],
                            checkpointer.get_next_version,
                            self.trigger_to_nodes,
                        )
                    # apply writes from tasks that already ran
                    for tid, k, v in saved.pending_writes or []:
                        if k in (ERROR, INTERRUPT):
                            continue
                        if tid not in next_tasks:
                            continue
                        next_tasks[tid].writes.append((k, v))
                    # clear all current tasks
                    apply_writes(
                        checkpoint,
                        channels,
                        next_tasks.values(),
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
                # save checkpoint
                next_config = await checkpointer.aput(
                    checkpoint_config,
                    create_checkpoint(checkpoint, channels, step),
                    {
                        "source": "update",
                        "step": step + 1,
                        "parents": saved.metadata.get("parents", {}) if saved else {},
                    },
                    get_new_channel_versions(
                        checkpoint_previous_versions, checkpoint["channel_versions"]
                    ),
                )
                return patch_checkpoint_map(
                    next_config, saved.metadata if saved else None
                )

            # act as an input
            if as_node == INPUT:
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot apply multiple updates when updating as input"
                    )

                if input_writes := deque(map_input(self.input_channels, values)):
                    apply_writes(
                        checkpoint,
                        channels,
                        [PregelTaskWrites((), INPUT, input_writes, [])],
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )

                    # apply input write to channels
                    next_step = (
                        step + 1
                        if saved and saved.metadata.get("step") is not None
                        else -1
                    )
                    next_config = await checkpointer.aput(
                        checkpoint_config,
                        create_checkpoint(checkpoint, channels, next_step),
                        {
                            "source": "input",
                            "step": next_step,
                            "parents": saved.metadata.get("parents", {})
                            if saved
                            else {},
                        },
                        get_new_channel_versions(
                            checkpoint_previous_versions,
                            checkpoint["channel_versions"],
                        ),
                    )

                    # store the writes
                    await checkpointer.aput_writes(
                        next_config,
                        input_writes,
                        str(uuid5(UUID(checkpoint["id"]), INPUT)),
                    )

                    return patch_checkpoint_map(
                        next_config, saved.metadata if saved else None
                    )
                else:
                    raise InvalidUpdateError(
                        f"Received no input writes for {self.input_channels}"
                    )

            # no values, copy checkpoint
            if as_node == "__copy__":
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot copy checkpoint with multiple updates"
                    )

                if saved is None:
                    raise InvalidUpdateError("Cannot copy a non-existent checkpoint")

                next_checkpoint = create_checkpoint(checkpoint, None, step)

                # copy checkpoint
                next_config = await checkpointer.aput(
                    saved.parent_config
                    or patch_configurable(
                        saved.config, {CONFIG_KEY_CHECKPOINT_ID: None}
                    ),
                    next_checkpoint,
                    {
                        "source": "fork",
                        "step": step + 1,
                        "parents": saved.metadata.get("parents", {}),
                    },
                    {},
                )

                # we want to both clone a checkpoint and update state in one go.
                # reuse the same task ID if possible.
                if isinstance(values, list) and len(values) > 0:
                    # figure out the task IDs for the next update checkpoint
                    next_tasks = prepare_next_tasks(
                        next_checkpoint,
                        saved.pending_writes or [],
                        self.nodes,
                        channels,
                        managed,
                        next_config,
                        step + 2,
                        step + 4,
                        for_execution=True,
                        store=self.store,
                        checkpointer=checkpointer,
                        manager=None,
                    )

                    tasks_group_by = defaultdict(list)
                    user_group_by: dict[str, list[StateUpdate]] = defaultdict(list)

                    for task in next_tasks.values():
                        tasks_group_by[task.name].append(task.id)

                    for item in values:
                        if not isinstance(item, Sequence):
                            raise InvalidUpdateError(
                                f"Invalid update item: {item} when copying checkpoint"
                            )

                        values, as_node = item[:2]
                        user_group = user_group_by[as_node]
                        tasks_group = tasks_group_by[as_node]

                        target_idx = len(user_group)
                        task_id = (
                            tasks_group[target_idx]
                            if target_idx < len(tasks_group)
                            else None
                        )

                        user_group_by[as_node].append(
                            StateUpdate(values=values, as_node=as_node, task_id=task_id)
                        )

                    return await aperform_superstep(
                        patch_checkpoint_map(next_config, saved.metadata),
                        [item for lst in user_group_by.values() for item in lst],
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
                    managed,
                    saved.config,
                    step + 1,
                    step + 3,
                    for_execution=True,
                    store=self.store,
                    checkpointer=checkpointer,
                    manager=None,
                )
                # apply null writes
                if null_writes := [
                    w[1:] for w in saved.pending_writes or [] if w[0] == NULL_TASK_ID
                ]:
                    apply_writes(
                        checkpoint,
                        channels,
                        [PregelTaskWrites((), INPUT, null_writes, [])],
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
                for tid, k, v in saved.pending_writes:
                    if k in (ERROR, INTERRUPT):
                        continue
                    if tid not in next_tasks:
                        continue
                    next_tasks[tid].writes.append((k, v))
                if tasks := [t for t in next_tasks.values() if t.writes]:
                    apply_writes(
                        checkpoint,
                        channels,
                        tasks,
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
            valid_updates: list[tuple[str, dict[str, Any] | None, str | None]] = []
            if len(updates) == 1:
                values, as_node, task_id = updates[0]
                # find last node that updated the state, if not provided
                if as_node is None and len(self.nodes) == 1:
                    as_node = tuple(self.nodes)[0]
                elif as_node is None and not saved:
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
                valid_updates.append((as_node, values, task_id))
            else:
                for values, as_node, task_id in updates:
                    if as_node is None:
                        raise InvalidUpdateError(
                            "as_node is required when applying multiple updates"
                        )
                    if as_node not in self.nodes:
                        raise InvalidUpdateError(f"Node {as_node} does not exist")

                    valid_updates.append((as_node, values, task_id))

            run_tasks: list[PregelTaskWrites] = []
            run_task_ids: list[str] = []

            for as_node, values, provided_task_id in valid_updates:
                # create task to run all writers of the chosen node
                writers = self.nodes[as_node].flat_writers
                if not writers:
                    raise InvalidUpdateError(f"Node {as_node} has no writers")
                writes: deque[tuple[str, Any]] = deque()
                task = PregelTaskWrites((), as_node, writes, [INTERRUPT])
                task_id = provided_task_id or str(
                    uuid5(UUID(checkpoint["id"]), INTERRUPT)
                )
                run_tasks.append(task)
                run_task_ids.append(task_id)
                run = RunnableSequence(*writers) if len(writers) > 1 else writers[0]
                # execute task
                await run.ainvoke(
                    values,
                    patch_config(
                        config,
                        run_name=self.name + "UpdateState",
                        configurable={
                            # deque.extend is thread-safe
                            CONFIG_KEY_SEND: writes.extend,
                            CONFIG_KEY_TASK_ID: task_id,
                            CONFIG_KEY_READ: partial(
                                local_read,
                                _scratchpad(
                                    None,
                                    [],
                                    task_id,
                                    "",
                                    None,
                                    step,
                                    step + 2,
                                ),
                                channels,
                                managed,
                                task,
                            ),
                        },
                    ),
                )
            # save task writes
            for task_id, task in zip(run_task_ids, run_tasks):
                # channel writes are saved to current checkpoint
                channel_writes = [w for w in task.writes if w[0] != PUSH]
                if saved and channel_writes:
                    await checkpointer.aput_writes(
                        checkpoint_config, channel_writes, task_id
                    )
            # apply to checkpoint and save
            apply_writes(
                checkpoint,
                channels,
                run_tasks,
                checkpointer.get_next_version,
                self.trigger_to_nodes,
            )
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            # save checkpoint, after applying writes
            next_config = await checkpointer.aput(
                checkpoint_config,
                checkpoint,
                {
                    "source": "update",
                    "step": step + 1,
                    "parents": saved.metadata.get("parents", {}) if saved else {},
                },
                get_new_channel_versions(
                    checkpoint_previous_versions, checkpoint["channel_versions"]
                ),
            )
            for task_id, task in zip(run_task_ids, run_tasks):
                # save push writes
                if push_writes := [w for w in task.writes if w[0] == PUSH]:
                    await checkpointer.aput_writes(next_config, push_writes, task_id)
            return patch_checkpoint_map(next_config, saved.metadata if saved else None)

        current_config = patch_configurable(
            config, {CONFIG_KEY_THREAD_ID: str(config[CONF][CONFIG_KEY_THREAD_ID])}
        )
        for superstep in supersteps:
            current_config = await aperform_superstep(current_config, superstep)
        return current_config

    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> RunnableConfig:
        """Update the state of the graph with the given values, as if they came from
        node `as_node`. If `as_node` is not provided, it will be set to the last node
        that updated the state, if not ambiguous.
        """
        return self.bulk_update_state(config, [[StateUpdate(values, as_node, task_id)]])

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any,
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> RunnableConfig:
        """Asynchronously update the state of the graph with the given values, as if they came from
        node `as_node`. If `as_node` is not provided, it will be set to the last node
        that updated the state, if not ambiguous.
        """
        return await self.abulk_update_state(
            config, [[StateUpdate(values, as_node, task_id)]]
        )

    def _defaults(
        self,
        config: RunnableConfig,
        *,
        stream_mode: StreamMode | Sequence[StreamMode],
        print_mode: StreamMode | Sequence[StreamMode],
        output_keys: str | Sequence[str] | None,
        interrupt_before: All | Sequence[str] | None,
        interrupt_after: All | Sequence[str] | None,
    ) -> tuple[
        set[StreamMode],
        str | Sequence[str],
        All | Sequence[str],
        All | Sequence[str],
        BaseCheckpointSaver | None,
        BaseStore | None,
        BaseCache | None,
    ]:
        if config["recursion_limit"] < 1:
            raise ValueError("recursion_limit must be at least 1")
        if output_keys is None:
            output_keys = self.stream_channels_asis
        else:
            validate_keys(output_keys, self.channels)
        interrupt_before = interrupt_before or self.interrupt_before_nodes
        interrupt_after = interrupt_after or self.interrupt_after_nodes
        if not isinstance(stream_mode, list):
            stream_modes = {stream_mode}
        else:
            stream_modes = set(stream_mode)
        if isinstance(print_mode, str):
            stream_modes.add(print_mode)
        else:
            stream_modes.update(print_mode)
        if self.checkpointer is False:
            checkpointer: BaseCheckpointSaver | None = None
        elif CONFIG_KEY_CHECKPOINTER in config.get(CONF, {}):
            checkpointer = config[CONF][CONFIG_KEY_CHECKPOINTER]
        elif self.checkpointer is True:
            raise RuntimeError("checkpointer=True cannot be used for root graphs.")
        else:
            checkpointer = self.checkpointer
        if checkpointer and not config.get(CONF):
            raise ValueError(
                "Checkpointer requires one or more of the following 'configurable' "
                "keys: thread_id, checkpoint_ns, checkpoint_id"
            )
        if CONFIG_KEY_STORE in config.get(CONF, {}):
            store: BaseStore | None = config[CONF][CONFIG_KEY_STORE]
        else:
            store = self.store
        if CONFIG_KEY_CACHE in config.get(CONF, {}):
            cache: BaseCache | None = config[CONF][CONFIG_KEY_CACHE]
        else:
            cache = self.cache
        return (
            stream_modes,
            output_keys,
            interrupt_before,
            interrupt_after,
            checkpointer,
            store,
            cache,
        )

    def stream(
        self,
        input: InputT,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        checkpoint_during: bool | None = None,
        debug: bool | None = None,
        subgraphs: bool = False,
    ) -> Iterator[dict[str, Any] | Any]:
        """Stream graph steps for a single input.

        Args:
            input: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to `self.stream_mode`.
                Options are:

                - `"values"`: Emit all values in the state after each step, including interrupts.
                    When used with functional API, values are emitted once at the end of the workflow.
                - `"updates"`: Emit only the node or task names and updates returned by the nodes or tasks after each step.
                    If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are emitted separately.
                - `"custom"`: Emit custom data from inside nodes or tasks using `StreamWriter`.
                - `"messages"`: Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes or tasks.
                    Will be emitted as 2-tuples `(LLM token, metadata)`.
                - `"checkpoints"`: Emit an event when a checkpoint is created, in the same format as returned by get_state().
                - `"tasks"`: Emit events when tasks start and finish, including their results and errors.

                You can pass a list as the `stream_mode` parameter to stream multiple modes at once.
                The streamed outputs will be tuples of `(mode, data)`.

                See [LangGraph streaming guide](https://langchain-ai.github.io/langgraph/how-tos/streaming/) for more details.
            print_mode: Accepts the same values as `stream_mode`, but only prints the output to the console, for debugging purposes. Does not affect the output of the graph in any way.
            output_keys: The keys to stream, defaults to all non-context channels.
            interrupt_before: Nodes to interrupt before, defaults to all nodes in the graph.
            interrupt_after: Nodes to interrupt after, defaults to all nodes in the graph.
            checkpoint_during: Whether to checkpoint intermediate steps, defaults to False. If False, only the final checkpoint is saved.
            subgraphs: Whether to stream events from inside subgraphs, defaults to False.
                If True, the events will be emitted as tuples `(namespace, data)`,
                or `(namespace, mode, data)` if `stream_mode` is a list,
                where `namespace` is a tuple with the path to the node where a subgraph is invoked,
                e.g. `("parent_node:<task_id>", "child_node:<task_id>")`.

                See [LangGraph streaming guide](https://langchain-ai.github.io/langgraph/how-tos/streaming/) for more details.

        Yields:
            The output of each step in the graph. The output shape depends on the stream_mode.
        """

        if stream_mode is None:
            # if being called as a node in another graph, default to values mode
            # but don't overwrite stream_mode arg if provided
            stream_mode = (
                "values"
                if config is not None and CONFIG_KEY_TASK_ID in config.get(CONF, {})
                else self.stream_mode
            )
        if debug or self.debug:
            print_mode = ["updates", "values"]

        stream = SyncQueue()

        config = ensure_config(self.config, config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name", self.get_name()),
            run_id=config.get("run_id"),
        )
        try:
            # assign defaults
            (
                stream_modes,
                output_keys,
                interrupt_before_,
                interrupt_after_,
                checkpointer,
                store,
                cache,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                print_mode=print_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
            )
            # set up subgraph checkpointing
            if self.checkpointer is True:
                ns = cast(str, config[CONF][CONFIG_KEY_CHECKPOINT_NS])
                config[CONF][CONFIG_KEY_CHECKPOINT_NS] = recast_checkpoint_ns(ns)
            # set up messages stream mode
            if "messages" in stream_modes:
                run_manager.inheritable_handlers.append(
                    StreamMessagesHandler(stream.put, subgraphs)
                )
            # set up custom stream mode
            if "custom" in stream_modes:
                config[CONF][CONFIG_KEY_STREAM_WRITER] = lambda c: stream.put(
                    (
                        tuple(
                            get_config()[CONF][CONFIG_KEY_CHECKPOINT_NS].split(NS_SEP)[
                                :-1
                            ]
                        ),
                        "custom",
                        c,
                    )
                )
            elif (
                CONFIG_KEY_STREAM not in config[CONF]
                and CONFIG_KEY_STREAM_WRITER in config[CONF]
            ):
                # remove parent graph stream writer if subgraph streaming not requested
                del config[CONF][CONFIG_KEY_STREAM_WRITER]
            # set checkpointing mode for subgraphs
            if checkpoint_during is not None:
                config[CONF][CONFIG_KEY_CHECKPOINT_DURING] = checkpoint_during
            with SyncPregelLoop(
                input,
                stream=StreamProtocol(stream.put, stream_modes),
                config=config,
                store=store,
                cache=cache,
                checkpointer=checkpointer,
                nodes=self.nodes,
                specs=self.channels,
                output_keys=output_keys,
                input_keys=self.input_channels,
                stream_keys=self.stream_channels_asis,
                interrupt_before=interrupt_before_,
                interrupt_after=interrupt_after_,
                manager=run_manager,
                checkpoint_during=checkpoint_during
                if checkpoint_during is not None
                else config[CONF].get(CONFIG_KEY_CHECKPOINT_DURING, True),
                trigger_to_nodes=self.trigger_to_nodes,
                migrate_checkpoint=self._migrate_checkpoint,
                retry_policy=self.retry_policy,
                cache_policy=self.cache_policy,
            ) as loop:
                # create runner
                runner = PregelRunner(
                    submit=config[CONF].get(
                        CONFIG_KEY_RUNNER_SUBMIT, weakref.WeakMethod(loop.submit)
                    ),
                    put_writes=weakref.WeakMethod(loop.put_writes),
                    node_finished=config[CONF].get(CONFIG_KEY_NODE_FINISHED),
                )
                # enable subgraph streaming
                if subgraphs:
                    loop.config[CONF][CONFIG_KEY_STREAM] = loop.stream
                # enable concurrent streaming
                if (
                    self.stream_eager
                    or subgraphs
                    or "messages" in stream_modes
                    or "custom" in stream_modes
                ):
                    # we are careful to have a single waiter live at any one time
                    # because on exit we increment semaphore count by exactly 1
                    waiter: concurrent.futures.Future | None = None
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
                while loop.tick():
                    for task in loop.match_cached_writes():
                        loop.output_writes(task.id, task.writes, cached=True)
                    for _ in runner.tick(
                        [t for t in loop.tasks.values() if not t.writes],
                        timeout=self.step_timeout,
                        get_waiter=get_waiter,
                        schedule_task=loop.accept_push,
                    ):
                        # emit output
                        yield from _output(
                            stream_mode, print_mode, subgraphs, stream.get, queue.Empty
                        )
                    loop.after_tick()
            # emit output
            yield from _output(
                stream_mode, print_mode, subgraphs, stream.get, queue.Empty
            )
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
            # set final channel values as run output
            run_manager.on_chain_end(loop.output)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise

    async def astream(
        self,
        input: InputT,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        checkpoint_during: bool | None = None,
        debug: bool | None = None,
        subgraphs: bool = False,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """Asynchronously stream graph steps for a single input.

        Args:
            input: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to `self.stream_mode`.
                Options are:

                - `"values"`: Emit all values in the state after each step, including interrupts.
                    When used with functional API, values are emitted once at the end of the workflow.
                - `"updates"`: Emit only the node or task names and updates returned by the nodes or tasks after each step.
                    If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are emitted separately.
                - `"custom"`: Emit custom data from inside nodes or tasks using `StreamWriter`.
                - `"messages"`: Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes or tasks.
                    Will be emitted as 2-tuples `(LLM token, metadata)`.
                - `"debug"`: Emit debug events with as much information as possible for each step.

                You can pass a list as the `stream_mode` parameter to stream multiple modes at once.
                The streamed outputs will be tuples of `(mode, data)`.

                See [LangGraph streaming guide](https://langchain-ai.github.io/langgraph/how-tos/streaming/) for more details.
            print_mode: Accepts the same values as `stream_mode`, but only prints the output to the console, for debugging purposes. Does not affect the output of the graph in any way.
            output_keys: The keys to stream, defaults to all non-context channels.
            interrupt_before: Nodes to interrupt before, defaults to all nodes in the graph.
            interrupt_after: Nodes to interrupt after, defaults to all nodes in the graph.
            checkpoint_during: Whether to checkpoint intermediate steps, defaults to False. If False, only the final checkpoint is saved.
            subgraphs: Whether to stream events from inside subgraphs, defaults to False.
                If True, the events will be emitted as tuples `(namespace, data)`,
                or `(namespace, mode, data)` if `stream_mode` is a list,
                where `namespace` is a tuple with the path to the node where a subgraph is invoked,
                e.g. `("parent_node:<task_id>", "child_node:<task_id>")`.

                See [LangGraph streaming guide](https://langchain-ai.github.io/langgraph/how-tos/streaming/) for more details.

        Yields:
            The output of each step in the graph. The output shape depends on the stream_mode.
        """

        if stream_mode is None:
            # if being called as a node in another graph, default to values mode
            # but don't overwrite stream_mode arg if provided
            stream_mode = (
                "values"
                if config is not None and CONFIG_KEY_TASK_ID in config.get(CONF, {})
                else self.stream_mode
            )
        if debug or self.debug:
            print_mode = ["updates", "values"]

        stream = AsyncQueue()
        aioloop = asyncio.get_running_loop()
        stream_put = cast(
            Callable[[StreamChunk], None],
            partial(aioloop.call_soon_threadsafe, stream.put_nowait),
        )

        config = ensure_config(self.config, config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name", self.get_name()),
            run_id=config.get("run_id"),
        )
        # if running from astream_log() run each proc with streaming
        do_stream = (
            next(
                (
                    True
                    for h in run_manager.handlers
                    if isinstance(h, _StreamingCallbackHandler)
                    and not isinstance(h, StreamMessagesHandler)
                ),
                False,
            )
            if _StreamingCallbackHandler is not None
            else False
        )
        try:
            # assign defaults
            (
                stream_modes,
                output_keys,
                interrupt_before_,
                interrupt_after_,
                checkpointer,
                store,
                cache,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                print_mode=print_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
            )
            # set up subgraph checkpointing
            if self.checkpointer is True:
                ns = cast(str, config[CONF][CONFIG_KEY_CHECKPOINT_NS])
                config[CONF][CONFIG_KEY_CHECKPOINT_NS] = recast_checkpoint_ns(ns)
            # set up messages stream mode
            if "messages" in stream_modes:
                run_manager.inheritable_handlers.append(
                    StreamMessagesHandler(stream_put, subgraphs)
                )
            # set up custom stream mode
            if "custom" in stream_modes:
                config[CONF][CONFIG_KEY_STREAM_WRITER] = (
                    lambda c: aioloop.call_soon_threadsafe(
                        stream.put_nowait,
                        (
                            tuple(
                                get_config()[CONF][CONFIG_KEY_CHECKPOINT_NS].split(
                                    NS_SEP
                                )[:-1]
                            ),
                            "custom",
                            c,
                        ),
                    )
                )
            elif (
                CONFIG_KEY_STREAM not in config[CONF]
                and CONFIG_KEY_STREAM_WRITER in config[CONF]
            ):
                # remove parent graph stream writer if subgraph streaming not requested
                del config[CONF][CONFIG_KEY_STREAM_WRITER]
            # set checkpointing mode for subgraphs
            if checkpoint_during is not None:
                config[CONF][CONFIG_KEY_CHECKPOINT_DURING] = checkpoint_during
            async with AsyncPregelLoop(
                input,
                stream=StreamProtocol(stream.put_nowait, stream_modes),
                config=config,
                store=store,
                cache=cache,
                checkpointer=checkpointer,
                nodes=self.nodes,
                specs=self.channels,
                output_keys=output_keys,
                input_keys=self.input_channels,
                stream_keys=self.stream_channels_asis,
                interrupt_before=interrupt_before_,
                interrupt_after=interrupt_after_,
                manager=run_manager,
                checkpoint_during=checkpoint_during
                if checkpoint_during is not None
                else config[CONF].get(CONFIG_KEY_CHECKPOINT_DURING, True),
                trigger_to_nodes=self.trigger_to_nodes,
                migrate_checkpoint=self._migrate_checkpoint,
                retry_policy=self.retry_policy,
                cache_policy=self.cache_policy,
            ) as loop:
                # create runner
                runner = PregelRunner(
                    submit=config[CONF].get(
                        CONFIG_KEY_RUNNER_SUBMIT, weakref.WeakMethod(loop.submit)
                    ),
                    put_writes=weakref.WeakMethod(loop.put_writes),
                    use_astream=do_stream,
                    node_finished=config[CONF].get(CONFIG_KEY_NODE_FINISHED),
                )
                # enable subgraph streaming
                if subgraphs:
                    loop.config[CONF][CONFIG_KEY_STREAM] = StreamProtocol(
                        stream_put, stream_modes
                    )
                # enable concurrent streaming
                if (
                    self.stream_eager
                    or subgraphs
                    or "messages" in stream_modes
                    or "custom" in stream_modes
                ):

                    def get_waiter() -> asyncio.Task[None]:
                        return aioloop.create_task(stream.wait())

                else:
                    get_waiter = None  # type: ignore[assignment]
                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1
                # channels are guaranteed to be immutable for the duration of the step,
                # with channel updates applied only at the transition between steps
                while loop.tick():
                    for task in await loop.amatch_cached_writes():
                        loop.output_writes(task.id, task.writes, cached=True)
                    async for _ in runner.atick(
                        [t for t in loop.tasks.values() if not t.writes],
                        timeout=self.step_timeout,
                        get_waiter=get_waiter,
                        schedule_task=loop.aaccept_push,
                    ):
                        # emit output
                        for o in _output(
                            stream_mode,
                            print_mode,
                            subgraphs,
                            stream.get_nowait,
                            asyncio.QueueEmpty,
                        ):
                            yield o
                    loop.after_tick()
            # emit output
            for o in _output(
                stream_mode,
                print_mode,
                subgraphs,
                stream.get_nowait,
                asyncio.QueueEmpty,
            ):
                yield o
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
            # set final channel values as run output
            await run_manager.on_chain_end(loop.output)
        except BaseException as e:
            await asyncio.shield(run_manager.on_chain_error(e))
            raise

    def invoke(
        self,
        input: InputT,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode = "values",
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        """Run the graph with a single input and config.

        Args:
            input: The input data for the graph. It can be a dictionary or any other type.
            config: Optional. The configuration for the graph run.
            stream_mode: Optional[str]. The stream mode for the graph run. Default is "values".
            print_mode: Accepts the same values as `stream_mode`, but only prints the output to the console, for debugging purposes. Does not affect the output of the graph in any way.
            output_keys: Optional. The output keys to retrieve from the graph run.
            interrupt_before: Optional. The nodes to interrupt the graph run before.
            interrupt_after: Optional. The nodes to interrupt the graph run after.
            **kwargs: Additional keyword arguments to pass to the graph run.

        Returns:
            The output of the graph run. If stream_mode is "values", it returns the latest output.
            If stream_mode is not "values", it returns a list of output chunks.
        """
        output_keys = output_keys if output_keys is not None else self.output_channels

        latest: dict[str, Any] | Any = None
        chunks: list[dict[str, Any] | Any] = []
        interrupts: list[Interrupt] = []

        for chunk in self.stream(
            input,
            config,
            stream_mode=["updates", "values"]
            if stream_mode == "values"
            else stream_mode,
            print_mode=print_mode,
            output_keys=output_keys,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            **kwargs,
        ):
            if stream_mode == "values":
                if len(chunk) == 2:
                    mode, payload = cast(tuple[StreamMode, Any], chunk)
                else:
                    _, mode, payload = cast(
                        tuple[tuple[str, ...], StreamMode, Any], chunk
                    )
                if (
                    mode == "updates"
                    and isinstance(payload, dict)
                    and (ints := payload.get(INTERRUPT)) is not None
                ):
                    interrupts.extend(ints)
                elif mode == "values":
                    latest = payload
            else:
                chunks.append(chunk)

        if stream_mode == "values":
            if interrupts:
                return (
                    {**latest, INTERRUPT: interrupts}
                    if isinstance(latest, dict)
                    else {INTERRUPT: interrupts}
                )
            return latest
        else:
            return chunks

    async def ainvoke(
        self,
        input: InputT,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode = "values",
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        """Asynchronously invoke the graph on a single input.

        Args:
            input: The input data for the computation. It can be a dictionary or any other type.
            config: Optional. The configuration for the computation.
            stream_mode: Optional. The stream mode for the computation. Default is "values".
            print_mode: Accepts the same values as `stream_mode`, but only prints the output to the console, for debugging purposes. Does not affect the output of the graph in any way.
            output_keys: Optional. The output keys to include in the result. Default is None.
            interrupt_before: Optional. The nodes to interrupt before. Default is None.
            interrupt_after: Optional. The nodes to interrupt after. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the computation. If stream_mode is "values", it returns the latest value.
            If stream_mode is "chunks", it returns a list of chunks.
        """

        output_keys = output_keys if output_keys is not None else self.output_channels

        latest: dict[str, Any] | Any = None
        chunks: list[dict[str, Any] | Any] = []
        interrupts: list[Interrupt] = []

        async for chunk in self.astream(
            input,
            config,
            stream_mode=["updates", "values"]
            if stream_mode == "values"
            else stream_mode,
            print_mode=print_mode,
            output_keys=output_keys,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            **kwargs,
        ):
            if stream_mode == "values":
                if len(chunk) == 2:
                    mode, payload = cast(tuple[StreamMode, Any], chunk)
                else:
                    _, mode, payload = cast(
                        tuple[tuple[str, ...], StreamMode, Any], chunk
                    )
                if (
                    mode == "updates"
                    and isinstance(payload, dict)
                    and (ints := payload.get(INTERRUPT)) is not None
                ):
                    interrupts.extend(ints)
                elif mode == "values":
                    latest = payload
            else:
                chunks.append(chunk)

        if stream_mode == "values":
            if interrupts:
                return (
                    {**latest, INTERRUPT: interrupts}
                    if isinstance(latest, dict)
                    else {INTERRUPT: interrupts}
                )
            return latest
        else:
            return chunks

    def clear_cache(self, nodes: Sequence[str] | None = None) -> None:
        """Clear the cache for the given nodes."""
        if not self.cache:
            raise ValueError("No cache is set for this graph. Cannot clear cache.")
        nodes = nodes or self.nodes.keys()
        # collect namespaces to clear
        namespaces: list[tuple[str, ...]] = []
        for node in nodes:
            if node in self.nodes:
                namespaces.append(
                    (
                        CACHE_NS_WRITES,
                        (identifier(self.nodes[node]) or "__dynamic__"),
                        node,
                    ),
                )
        # clear cache
        self.cache.clear(namespaces)

    async def aclear_cache(self, nodes: Sequence[str] | None = None) -> None:
        """Asynchronously clear the cache for the given nodes."""
        if not self.cache:
            raise ValueError("No cache is set for this graph. Cannot clear cache.")
        nodes = nodes or self.nodes.keys()
        # collect namespaces to clear
        namespaces: list[tuple[str, ...]] = []
        for node in nodes:
            if node in self.nodes:
                namespaces.append(
                    (
                        CACHE_NS_WRITES,
                        (identifier(self.nodes[node]) or "__dynamic__"),
                        node,
                    ),
                )
        # clear cache
        await self.cache.aclear(namespaces)


def _trigger_to_nodes(nodes: dict[str, PregelNode]) -> Mapping[str, Sequence[str]]:
    """Index from a trigger to nodes that depend on it."""
    trigger_to_nodes: defaultdict[str, list[str]] = defaultdict(list)
    for name, node in nodes.items():
        for trigger in node.triggers:
            trigger_to_nodes[trigger].append(name)
    return dict(trigger_to_nodes)


def _output(
    stream_mode: StreamMode | Sequence[StreamMode],
    print_mode: StreamMode | Sequence[StreamMode],
    stream_subgraphs: bool,
    getter: Callable[[], tuple[tuple[str, ...], str, Any]],
    empty_exc: type[Exception],
) -> Iterator:
    while True:
        try:
            ns, mode, payload = getter()
        except empty_exc:
            break
        if mode in print_mode:
            if stream_subgraphs and ns:
                print(
                    " ".join(
                        (
                            get_bolded_text(f"[{mode}]"),
                            get_colored_text(f"[graph={ns}]", color="yellow"),
                            repr(payload),
                        )
                    )
                )
            else:
                print(
                    " ".join(
                        (
                            get_bolded_text(f"[{mode}]"),
                            repr(payload),
                        )
                    )
                )
        if mode in stream_mode:
            if stream_subgraphs and isinstance(stream_mode, list):
                yield (ns, mode, payload)
            elif isinstance(stream_mode, list):
                yield (mode, payload)
            elif stream_subgraphs:
                yield (ns, payload)
            else:
                yield payload
