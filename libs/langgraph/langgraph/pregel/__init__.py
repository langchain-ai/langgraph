from __future__ import annotations

import asyncio
import concurrent
import concurrent.futures
import queue
import weakref
from collections import defaultdict, deque
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    get_type_hints,
    overload,
)
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
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from pydantic import BaseModel
from typing_extensions import Self

from langgraph.channels.base import (
    BaseChannel,
)
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
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
    CONFIG_KEY_RESUMING,
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
from langgraph.managed.base import ManagedValueSpec
from langgraph.pregel.algo import (
    PregelTaskWrites,
    apply_writes,
    local_read,
    local_write,
    prepare_next_tasks,
)
from langgraph.pregel.debug import tasks_w_writes
from langgraph.pregel.io import map_input, read_channels
from langgraph.pregel.loop import AsyncPregelLoop, StreamProtocol, SyncPregelLoop
from langgraph.pregel.manager import AsyncChannelsManager, ChannelsManager
from langgraph.pregel.messages import StreamMessagesHandler
from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.read import PregelNode
from langgraph.pregel.retry import RetryPolicy
from langgraph.pregel.runner import PregelRunner
from langgraph.pregel.utils import get_new_channel_versions
from langgraph.pregel.validate import validate_graph, validate_keys
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.store.base import BaseStore
from langgraph.types import (
    All,
    Checkpointer,
    LoopProtocol,
    StateSnapshot,
    StateUpdate,
    StreamChunk,
    StreamMode,
)
from langgraph.utils.config import (
    ensure_config,
    merge_configs,
    patch_checkpoint_map,
    patch_config,
    patch_configurable,
    recast_checkpoint_ns,
)
from langgraph.utils.fields import get_enhanced_type_hints
from langgraph.utils.pydantic import create_model, is_supported_by_pydantic
from langgraph.utils.queue import AsyncQueue, SyncQueue  # type: ignore[attr-defined]

WriteValue = Union[Callable[[Input], Output], Any]


class Channel:
    @overload
    @classmethod
    def subscribe_to(
        cls,
        channels: str,
        *,
        key: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> PregelNode: ...

    @overload
    @classmethod
    def subscribe_to(
        cls,
        channels: Sequence[str],
        *,
        key: None = None,
        tags: Optional[list[str]] = None,
    ) -> PregelNode: ...

    @classmethod
    def subscribe_to(
        cls,
        channels: Union[str, Sequence[str]],
        *,
        key: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> PregelNode:
        """Runs process.invoke() each time channels are updated,
        with a dict of the channel values as input."""
        if not isinstance(channels, str) and key is not None:
            raise ValueError(
                "Can't specify a key when subscribing to multiple channels"
            )
        return PregelNode(
            channels=cast(
                Union[list[str], Mapping[str, str]],
                (
                    {key: channels}
                    if isinstance(channels, str) and key is not None
                    else (
                        [channels]
                        if isinstance(channels, str)
                        else {chan: chan for chan in channels}
                    )
                ),
            ),
            triggers=[channels] if isinstance(channels, str) else channels,
            tags=tags,
        )

    @classmethod
    def write_to(
        cls,
        *channels: str,
        **kwargs: WriteValue,
    ) -> ChannelWrite:
        """Writes to channels the result of the lambda, or None to skip writing."""
        return ChannelWrite(
            [ChannelWriteEntry(c) for c in channels]
            + [
                (
                    ChannelWriteEntry(k, mapper=v)
                    if callable(v)
                    else ChannelWriteEntry(k, value=v)
                )
                for k, v in kwargs.items()
            ]
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

    channels: dict[str, Union[BaseChannel, ManagedValueSpec]]

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

    debug: bool
    """Whether to print debug information during execution. Defaults to False."""

    checkpointer: Checkpointer = None
    """Checkpointer used to save and load graph state. Defaults to None."""

    store: Optional[BaseStore] = None
    """Memory store to use for SharedValues. Defaults to None."""

    retry_policy: Optional[RetryPolicy] = None
    """Retry policy to use when running tasks. Set to None to disable."""

    config_type: Optional[Type[Any]] = None

    input_model: Optional[Type[BaseModel]] = None

    config: Optional[RunnableConfig] = None

    name: str = "LangGraph"

    trigger_to_nodes: Optional[Mapping[str, Sequence[str]]] = None

    def __init__(
        self,
        *,
        nodes: dict[str, PregelNode],
        channels: Optional[dict[str, Union[BaseChannel, ManagedValueSpec]]],
        auto_validate: bool = True,
        stream_mode: StreamMode = "values",
        stream_eager: bool = False,
        output_channels: Union[str, Sequence[str]],
        stream_channels: Optional[Union[str, Sequence[str]]] = None,
        interrupt_after_nodes: Union[All, Sequence[str]] = (),
        interrupt_before_nodes: Union[All, Sequence[str]] = (),
        input_channels: Union[str, Sequence[str]],
        step_timeout: Optional[float] = None,
        debug: Optional[bool] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        store: Optional[BaseStore] = None,
        retry_policy: Optional[RetryPolicy] = None,
        config_type: Optional[Type[Any]] = None,
        input_model: Optional[Type[BaseModel]] = None,
        config: Optional[RunnableConfig] = None,
        trigger_to_nodes: Optional[Mapping[str, Sequence[str]]] = None,
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
        self.debug = debug if debug is not None else get_debug()
        self.checkpointer = checkpointer
        self.store = store
        self.retry_policy = retry_policy
        self.config_type = config_type
        self.input_model = input_model
        self.config = config
        self.trigger_to_nodes = trigger_to_nodes
        self.name = name
        if auto_validate:
            self.validate()

    def get_graph(
        self, config: Optional[RunnableConfig] = None, *, xray: Union[int, bool] = False
    ) -> Graph:
        raise NotImplementedError

    async def aget_graph(
        self, config: Optional[RunnableConfig] = None, *, xray: Union[int, bool] = False
    ) -> Graph:
        raise NotImplementedError

    def copy(self, update: Optional[dict[str, Any]] = None) -> Self:
        attrs = {**self.__dict__, **(update or {})}
        return self.__class__(**attrs)

    def with_config(
        self, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Self:
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
        self.trigger_to_nodes = _trigger_to_nodes(self.nodes)
        return self

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [
            spec
            for spec in get_unique_config_specs(
                [spec for node in self.nodes.values() for spec in node.config_specs]
                + (
                    self.checkpointer.config_specs
                    if isinstance(self.checkpointer, BaseCheckpointSaver)
                    else []
                )
                + (
                    [
                        ConfigurableFieldSpec(
                            id=name,
                            annotation=typ,
                            default=default,
                            description=description,
                        )
                        for name, typ, default, description in get_enhanced_type_hints(
                            self.config_type
                        )
                    ]
                    if self.config_type is not None
                    else []
                )
            )
            # these are provided by the Pregel class
            if spec.id
            not in [
                CONFIG_KEY_READ,
                CONFIG_KEY_SEND,
                CONFIG_KEY_CHECKPOINTER,
                CONFIG_KEY_RESUMING,
            ]
        ]

    def config_schema(
        self, *, include: Optional[Sequence[str]] = None
    ) -> Type[BaseModel]:
        # If the config type is not set explicitly, we will try to infer it.
        # If the config type is provided, but isn't directly supported by pydantic
        # (e.g., vanilla python class), we will also delegate to the parent class,
        # which handles cases where Pydantic doesn't support the type.
        if self.config_type is None or not is_supported_by_pydantic(self.config_type):
            return super().config_schema(include=include)

        include = include or []
        fields = {
            "configurable": (self.config_type, None),
            **{
                field_name: (field_type, None)
                for field_name, field_type in get_type_hints(RunnableConfig).items()
                if field_name in [i for i in include if i != "configurable"]
            },
        }
        return create_model(self.get_name("Config"), field_definitions=fields)

    def get_config_jsonschema(
        self, *, include: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        schema = self.config_schema(include=include)
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        else:
            return schema.schema()

    @property
    def InputType(self) -> Any:
        if isinstance(self.input_channels, str):
            channel = self.channels[self.input_channels]
            if isinstance(channel, BaseChannel):
                return channel.UpdateType

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        if self.input_model is not None:
            return self.input_model
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
        self, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        schema = self.get_input_schema(config)
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        else:
            return schema.schema()

    @property
    def OutputType(self) -> Any:
        if isinstance(self.output_channels, str):
            channel = self.channels[self.output_channels]
            if isinstance(channel, BaseChannel):
                return channel.ValueType

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
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
        self, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        schema = self.get_output_schema(config)
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        else:
            return schema.schema()

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

    async def aget_subgraphs(
        self, *, namespace: Optional[str] = None, recurse: bool = False
    ) -> AsyncIterator[tuple[str, PregelProtocol]]:
        for name, node in self.get_subgraphs(namespace=namespace, recurse=recurse):
            yield name, node

    def _prepare_state_snapshot(
        self,
        config: RunnableConfig,
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
            LoopProtocol(
                config=saved.config,
                step=saved.metadata.get("step", -1) + 1,
                stop=saved.metadata.get("step", -1) + 2,
            ),
            skip_context=True,
        ) as (channels, managed):
            # tasks for this checkpoint
            next_tasks = prepare_next_tasks(
                saved.checkpoint,
                saved.pending_writes or [],
                self.nodes,
                channels,
                managed,
                saved.config,
                saved.metadata.get("step", -1) + 1,
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

    async def _aprepare_state_snapshot(
        self,
        config: RunnableConfig,
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

        async with AsyncChannelsManager(
            self.channels,
            saved.checkpoint,
            LoopProtocol(
                config=saved.config,
                step=saved.metadata.get("step", -1) + 1,
                stop=saved.metadata.get("step", -1) + 2,
            ),
            skip_context=True,
        ) as (
            channels,
            managed,
        ):
            # tasks for this checkpoint
            next_tasks = prepare_next_tasks(
                saved.checkpoint,
                saved.pending_writes or [],
                self.nodes,
                channels,
                managed,
                saved.config,
                saved.metadata.get("step", -1) + 1,
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

    def get_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
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

    async def aget_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
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

    async def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[StateSnapshot]:
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
            {CONF: {CONFIG_KEY_CHECKPOINT_NS: checkpoint_ns}},
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
                        Each update is a tuple of the form `(values, as_node)`.

        Raises:
            ValueError: If no checkpointer is set or no updates are provided.
            InvalidUpdateError: If an invalid update is provided.

        Returns:
            RunnableConfig: The updated config.
        """

        checkpointer: Optional[BaseCheckpointSaver] = ensure_config(config)[CONF].get(
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
            checkpoint_metadata = config["metadata"]
            if saved:
                checkpoint_config = patch_configurable(config, saved.config[CONF])
                checkpoint_metadata = {**saved.metadata, **checkpoint_metadata}
            with ChannelsManager(
                self.channels,
                checkpoint,
                LoopProtocol(config=config, step=step + 1, stop=step + 2),
            ) as (channels, managed):
                values, as_node = updates[0]

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
                            saved.metadata.get("step", -1) + 1,
                            for_execution=True,
                            store=self.store,
                            checkpointer=self.checkpointer
                            if isinstance(self.checkpointer, BaseCheckpointSaver)
                            else None,
                            manager=None,
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
                            "parents": saved.metadata.get("parents", {})
                            if saved
                            else {},
                        },
                        {},
                    )
                    return patch_checkpoint_map(
                        next_config, saved.metadata if saved else None
                    )
                # no values, empty checkpoint
                if values is None and as_node is None:
                    if len(updates) > 1:
                        raise InvalidUpdateError(
                            "Cannot create empty checkpoint with multiple updates"
                        )

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
                            "parents": saved.metadata.get("parents", {})
                            if saved
                            else {},
                        },
                        {},
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
                                **checkpoint_metadata,
                                "source": "input",
                                "step": next_step,
                                "writes": dict(input_writes),
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

                # no values, copy checkpoint
                if values is None and as_node == "__copy__":
                    if len(updates) > 1:
                        raise InvalidUpdateError(
                            "Cannot copy checkpoint with multiple updates"
                        )

                    next_checkpoint = create_checkpoint(checkpoint, None, step)
                    # copy checkpoint
                    next_config = checkpointer.put(
                        saved.parent_config or saved.config
                        if saved
                        else checkpoint_config,
                        next_checkpoint,
                        {
                            **checkpoint_metadata,
                            "source": "fork",
                            "step": step + 1,
                            "parents": saved.metadata.get("parents", {})
                            if saved
                            else {},
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
                        managed,
                        saved.config,
                        saved.metadata.get("step", -1) + 1,
                        for_execution=True,
                        store=self.store,
                        checkpointer=(
                            self.checkpointer
                            if isinstance(self.checkpointer, BaseCheckpointSaver)
                            else None
                        ),
                        manager=None,
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
                    # apply writes
                    for tid, k, v in saved.pending_writes:
                        if k in (ERROR, INTERRUPT, SCHEDULED):
                            continue
                        if tid not in next_tasks:
                            continue
                        next_tasks[tid].writes.append((k, v))
                    if tasks := [t for t in next_tasks.values() if t.writes]:
                        apply_writes(checkpoint, channels, tasks, None)
            valid_updates: list[tuple[str, Optional[dict[str, Any]]]] = []
            if len(updates) == 1:
                values, as_node = updates[0]
                # find last node that updated the state, if not provided
                if as_node is None and not any(
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
                valid_updates.append((as_node, values))
            else:
                for values, as_node in updates:
                    if as_node is None:
                        raise InvalidUpdateError(
                            "as_node is required when applying multiple updates"
                        )
                    if as_node not in self.nodes:
                        raise InvalidUpdateError(f"Node {as_node} does not exist")

                    valid_updates.append((as_node, values))

            run_tasks: list[PregelTaskWrites] = []
            run_task_ids: list[str] = []

            for as_node, values in valid_updates:
                # create task to run all writers of the chosen node
                writers = self.nodes[as_node].flat_writers
                if not writers:
                    raise InvalidUpdateError(f"Node {as_node} has no writers")
                writes: deque[tuple[str, Any]] = deque()
                task = PregelTaskWrites((), as_node, writes, [INTERRUPT])
                task_id = str(uuid5(UUID(checkpoint["id"]), INTERRUPT))
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
                            CONFIG_KEY_SEND: partial(
                                local_write,
                                writes.extend,
                                self.nodes.keys(),
                            ),
                            CONFIG_KEY_READ: partial(
                                local_read,
                                step + 1,
                                checkpoint,
                                channels,
                                managed,
                                task,
                                config,
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
            mv_writes, _ = apply_writes(
                checkpoint, channels, run_tasks, checkpointer.get_next_version
            )
            assert not mv_writes, "Can't write to SharedValues from update_state"
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            next_config = checkpointer.put(
                checkpoint_config,
                checkpoint,
                {
                    **checkpoint_metadata,
                    "source": "update",
                    "step": step + 1,
                    "writes": {as_node: values for as_node, values in valid_updates},
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

        current_config = config
        for superstep in supersteps:
            current_config = perform_superstep(current_config, superstep)
        return current_config

    async def abulk_update_state(
        self,
        config: RunnableConfig,
        supersteps: Sequence[Sequence[StateUpdate]],
    ) -> RunnableConfig:
        """Apply updates to the graph state in bulk. Requires a checkpointer to be set.

        Args:
            config: The config to apply the updates to.
            supersteps: A list of supersteps, each including a list of updates to apply sequentially to a graph state.
                        Each update is a tuple of the form `(values, as_node)`.

        Raises:
            ValueError: If no checkpointer is set or no updates are provided.
            InvalidUpdateError: If an invalid update is provided.

        Returns:
            RunnableConfig: The updated config.
        """

        checkpointer: Optional[BaseCheckpointSaver] = ensure_config(config)[CONF].get(
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
            checkpoint_metadata = config["metadata"]
            if saved:
                checkpoint_config = patch_configurable(config, saved.config[CONF])
                checkpoint_metadata = {**saved.metadata, **checkpoint_metadata}
            async with AsyncChannelsManager(
                self.channels,
                checkpoint,
                LoopProtocol(config=config, step=step + 1, stop=step + 2),
            ) as (
                channels,
                managed,
            ):
                values, as_node = updates[0]
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
                            saved.metadata.get("step", -1) + 1,
                            for_execution=True,
                            store=self.store,
                            checkpointer=self.checkpointer
                            if isinstance(self.checkpointer, BaseCheckpointSaver)
                            else None,
                            manager=None,
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
                    next_config = await checkpointer.aput(
                        checkpoint_config,
                        create_checkpoint(checkpoint, None, step),
                        {
                            **checkpoint_metadata,
                            "source": "update",
                            "step": step + 1,
                            "writes": {},
                            "parents": saved.metadata.get("parents", {})
                            if saved
                            else {},
                        },
                        {},
                    )
                    return patch_checkpoint_map(
                        next_config, saved.metadata if saved else None
                    )
                # no values, empty checkpoint
                if values is None and as_node is None:
                    if len(updates) > 1:
                        raise InvalidUpdateError(
                            "Cannot create empty checkpoint with multiple updates"
                        )

                    next_checkpoint = create_checkpoint(checkpoint, None, step)
                    # copy checkpoint
                    next_config = await checkpointer.aput(
                        checkpoint_config,
                        next_checkpoint,
                        {
                            **checkpoint_metadata,
                            "source": "update",
                            "step": step + 1,
                            "writes": {},
                            "parents": saved.metadata.get("parents", {})
                            if saved
                            else {},
                        },
                        {},
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
                                **checkpoint_metadata,
                                "source": "input",
                                "step": next_step,
                                "writes": dict(input_writes),
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
                if values is None and as_node == "__copy__":
                    if len(updates) > 1:
                        raise InvalidUpdateError(
                            "Cannot copy checkpoint with multiple updates"
                        )

                    next_checkpoint = create_checkpoint(checkpoint, None, step)
                    # copy checkpoint
                    next_config = await checkpointer.aput(
                        saved.parent_config or saved.config
                        if saved
                        else checkpoint_config,
                        next_checkpoint,
                        {
                            **checkpoint_metadata,
                            "source": "fork",
                            "step": step + 1,
                            "parents": saved.metadata.get("parents", {})
                            if saved
                            else {},
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
                        managed,
                        saved.config,
                        saved.metadata.get("step", -1) + 1,
                        for_execution=True,
                        store=self.store,
                        checkpointer=(
                            self.checkpointer
                            if isinstance(self.checkpointer, BaseCheckpointSaver)
                            else None
                        ),
                        manager=None,
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
                    for tid, k, v in saved.pending_writes:
                        if k in (ERROR, INTERRUPT, SCHEDULED):
                            continue
                        if tid not in next_tasks:
                            continue
                        next_tasks[tid].writes.append((k, v))
                    if tasks := [t for t in next_tasks.values() if t.writes]:
                        apply_writes(checkpoint, channels, tasks, None)
            valid_updates: list[tuple[str, Optional[dict[str, Any]]]] = []
            if len(updates) == 1:
                values, as_node = updates[0]
                # find last node that updated the state, if not provided
                if as_node is None and not saved:
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
                valid_updates.append((as_node, values))
            else:
                for values, as_node in updates:
                    if as_node is None:
                        raise InvalidUpdateError(
                            "as_node is required when applying multiple updates"
                        )
                    if as_node not in self.nodes:
                        raise InvalidUpdateError(f"Node {as_node} does not exist")

                    valid_updates.append((as_node, values))

            run_tasks: list[PregelTaskWrites] = []
            run_task_ids: list[str] = []

            for as_node, values in valid_updates:
                # create task to run all writers of the chosen node
                writers = self.nodes[as_node].flat_writers
                if not writers:
                    raise InvalidUpdateError(f"Node {as_node} has no writers")
                writes: deque[tuple[str, Any]] = deque()
                task = PregelTaskWrites((), as_node, writes, [INTERRUPT])
                task_id = str(uuid5(UUID(checkpoint["id"]), INTERRUPT))
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
                            CONFIG_KEY_SEND: partial(
                                local_write,
                                writes.extend,
                                self.nodes.keys(),
                            ),
                            CONFIG_KEY_READ: partial(
                                local_read,
                                step + 1,
                                checkpoint,
                                channels,
                                managed,
                                task,
                                config,
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
            mv_writes, _ = apply_writes(
                checkpoint, channels, run_tasks, checkpointer.get_next_version
            )
            assert not mv_writes, "Can't write to SharedValues from update_state"
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            # save checkpoint, after applying writes
            next_config = await checkpointer.aput(
                checkpoint_config,
                checkpoint,
                {
                    **checkpoint_metadata,
                    "source": "update",
                    "step": step + 1,
                    "writes": {as_node: values for as_node, values in valid_updates},
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

        current_config = config
        for superstep in supersteps:
            current_config = await aperform_superstep(current_config, superstep)
        return current_config

    def update_state(
        self,
        config: RunnableConfig,
        values: Optional[Union[dict[str, Any], Any]],
        as_node: Optional[str] = None,
    ) -> RunnableConfig:
        """Update the state of the graph with the given values, as if they came from
        node `as_node`. If `as_node` is not provided, it will be set to the last node
        that updated the state, if not ambiguous.
        """
        return self.bulk_update_state(config, [[StateUpdate(values, as_node)]])

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any,
        as_node: Optional[str] = None,
    ) -> RunnableConfig:
        """Update the state of the graph asynchronously with the given values, as if they came from
        node `as_node`. If `as_node` is not provided, it will be set to the last node
        that updated the state, if not ambiguous.
        """
        return await self.abulk_update_state(config, [[StateUpdate(values, as_node)]])

    def _defaults(
        self,
        config: RunnableConfig,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]],
        output_keys: Optional[Union[str, Sequence[str]]],
        interrupt_before: Optional[Union[All, Sequence[str]]],
        interrupt_after: Optional[Union[All, Sequence[str]]],
        debug: Optional[bool],
    ) -> tuple[
        bool,
        set[StreamMode],
        Union[str, Sequence[str]],
        Union[All, Sequence[str]],
        Union[All, Sequence[str]],
        Optional[BaseCheckpointSaver],
        Optional[BaseStore],
    ]:
        if config["recursion_limit"] < 1:
            raise ValueError("recursion_limit must be at least 1")
        debug = debug if debug is not None else self.debug
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
                f"Checkpointer requires one or more of the following 'configurable' keys: {[s.id for s in checkpointer.config_specs]}"
            )
        if CONFIG_KEY_STORE in config.get(CONF, {}):
            store: Optional[BaseStore] = config[CONF][CONFIG_KEY_STORE]
        else:
            store = self.store
        return (
            debug,
            set(stream_mode),
            output_keys,
            interrupt_before,
            interrupt_after,
            checkpointer,
            store,
        )

    def stream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: Optional[bool] = None,
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
                - `"messages"`: Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes or tasks.
                - `"debug"`: Emit debug events with as much information as possible for each step.
            output_keys: The keys to stream, defaults to all non-context channels.
            interrupt_before: Nodes to interrupt before, defaults to all nodes in the graph.
            interrupt_after: Nodes to interrupt after, defaults to all nodes in the graph.
            debug: Whether to print debug information during execution, defaults to False.
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

            With stream_mode="messages":

            ```pycon
            >>> from typing_extensions import Annotated, TypedDict
            >>> from langgraph.graph import StateGraph, START
            >>> from langchain_openai import ChatOpenAI
            ...
            >>> llm = ChatOpenAI(model="gpt-4o-mini")
            ...
            >>> class State(TypedDict):
            ...     question: str
            ...     answer: str
            ...
            >>> def node_a(state: State):
            ...     response = llm.invoke(state["question"])
            ...     return {"answer": response.content}
            ...
            >>> builder = StateGraph(State)
            >>> builder.add_node("a", node_a)
            >>> builder.add_edge(START, "a")
            >>> graph = builder.compile()

            >>> for event in graph.stream({"question": "What is the capital of France?"}, stream_mode="messages"):
            ...     print(event)
            (AIMessageChunk(content='The', additional_kwargs={}, response_metadata={}, id='...'), {'langgraph_step': 1, 'langgraph_node': 'a', 'langgraph_triggers': ['start:a'], 'langgraph_path': ('__pregel_pull', 'a'), 'langgraph_checkpoint_ns': '...', 'checkpoint_ns': '...', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': 0.7})
            (AIMessageChunk(content=' capital', additional_kwargs={}, response_metadata={}, id='...'), {'langgraph_step': 1, 'langgraph_node': 'a', 'langgraph_triggers': ['start:a'], ...})
            (AIMessageChunk(content=' of', additional_kwargs={}, response_metadata={}, id='...'), {...})
            (AIMessageChunk(content=' France', additional_kwargs={}, response_metadata={}, id='...'), {...})
            (AIMessageChunk(content=' is', additional_kwargs={}, response_metadata={}, id='...'), {...})
            (AIMessageChunk(content=' Paris', additional_kwargs={}, response_metadata={}, id='...'), {...})
            ```
        """

        stream = SyncQueue()

        def output() -> Iterator:
            while True:
                try:
                    ns, mode, payload = stream.get(block=False)
                except queue.Empty:
                    break
                if subgraphs and isinstance(stream_mode, list):
                    yield (ns, mode, payload)
                elif isinstance(stream_mode, list):
                    yield (mode, payload)
                elif subgraphs:
                    yield (ns, payload)
                else:
                    yield payload

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
                debug,
                stream_modes,
                output_keys,
                interrupt_before_,
                interrupt_after_,
                checkpointer,
                store,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
            )
            # set up subgraph checkpointing
            if self.checkpointer is True:
                ns = cast(str, config[CONF][CONFIG_KEY_CHECKPOINT_NS])
                config[CONF][CONFIG_KEY_CHECKPOINT_NS] = recast_checkpoint_ns(ns)
            # set up messages stream mode
            if "messages" in stream_modes:
                run_manager.inheritable_handlers.append(
                    StreamMessagesHandler(stream.put)
                )
            # set up custom stream mode
            if "custom" in stream_modes:
                config[CONF][CONFIG_KEY_STREAM_WRITER] = lambda c: stream.put(
                    ((), "custom", c)
                )
            with SyncPregelLoop(
                input,
                input_model=self.input_model,
                stream=StreamProtocol(stream.put, stream_modes),
                config=config,
                store=store,
                checkpointer=checkpointer,
                nodes=self.nodes,
                specs=self.channels,
                output_keys=output_keys,
                stream_keys=self.stream_channels_asis,
                interrupt_before=interrupt_before_,
                interrupt_after=interrupt_after_,
                manager=run_manager,
                debug=debug,
                trigger_to_nodes=self.trigger_to_nodes,
            ) as loop:
                # create runner
                runner = PregelRunner(
                    submit=config[CONF].get(
                        CONFIG_KEY_RUNNER_SUBMIT, weakref.WeakMethod(loop.submit)
                    ),
                    put_writes=weakref.WeakMethod(loop.put_writes),
                    schedule_task=weakref.WeakMethod(loop.accept_push),
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
            # set final channel values as run output
            run_manager.on_chain_end(loop.output)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise

    async def astream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: Optional[bool] = None,
        subgraphs: bool = False,
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
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
                - `"messages"`: Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes or tasks.
                - `"debug"`: Emit debug events with as much information as possible for each step.
            output_keys: The keys to stream, defaults to all non-context channels.
            interrupt_before: Nodes to interrupt before, defaults to all nodes in the graph.
            interrupt_after: Nodes to interrupt after, defaults to all nodes in the graph.
            debug: Whether to print debug information during execution, defaults to False.
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
            >>> async for event in graph.astream({"alist": ['Ex for stream_mode="values"']}, stream_mode="values"):
            ...     print(event)
            {'alist': ['Ex for stream_mode="values"'], 'another_list': []}
            {'alist': ['Ex for stream_mode="values"'], 'another_list': ['hi']}
            {'alist': ['Ex for stream_mode="values"', 'there'], 'another_list': ['hi']}
            ```
            With stream_mode="updates":

            ```pycon
            >>> async for event in graph.astream({"alist": ['Ex for stream_mode="updates"']}, stream_mode="updates"):
            ...     print(event)
            {'a': {'another_list': ['hi']}}
            {'b': {'alist': ['there']}}
            ```
            With stream_mode="debug":

            ```pycon
            >>> async for event in graph.astream({"alist": ['Ex for stream_mode="debug"']}, stream_mode="debug"):
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
            >>> async def node_a(state: State, writer: StreamWriter):
            ...     writer({"custom_data": "foo"})
            ...     return {"alist": ["hi"]}
            ...
            >>> builder = StateGraph(State)
            >>> builder.add_node("a", node_a)
            >>> builder.add_edge(START, "a")
            >>> graph = builder.compile()
            ...
            >>> async for event in graph.astream({"alist": ['Ex for stream_mode="custom"']}, stream_mode="custom"):
            ...     print(event)
            {'custom_data': 'foo'}
            ```

            With stream_mode="messages":

            ```pycon
            >>> from typing_extensions import Annotated, TypedDict
            >>> from langgraph.graph import StateGraph, START
            >>> from langchain_openai import ChatOpenAI
            ...
            >>> llm = ChatOpenAI(model="gpt-4o-mini")
            ...
            >>> class State(TypedDict):
            ...     question: str
            ...     answer: str
            ...
            >>> async def node_a(state: State):
            ...     response = await llm.ainvoke(state["question"])
            ...     return {"answer": response.content}
            ...
            >>> builder = StateGraph(State)
            >>> builder.add_node("a", node_a)
            >>> builder.add_edge(START, "a")
            >>> graph = builder.compile()

            >>> for event in graph.stream({"question": "What is the capital of France?"}, stream_mode="messages"):
            ...     print(event)
            (AIMessageChunk(content='The', additional_kwargs={}, response_metadata={}, id='...'), {'langgraph_step': 1, 'langgraph_node': 'a', 'langgraph_triggers': ['start:a'], 'langgraph_path': ('__pregel_pull', 'a'), 'langgraph_checkpoint_ns': '...', 'checkpoint_ns': '...', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': 0.7})
            (AIMessageChunk(content=' capital', additional_kwargs={}, response_metadata={}, id='...'), {'langgraph_step': 1, 'langgraph_node': 'a', 'langgraph_triggers': ['start:a'], ...})
            (AIMessageChunk(content=' of', additional_kwargs={}, response_metadata={}, id='...'), {...})
            (AIMessageChunk(content=' France', additional_kwargs={}, response_metadata={}, id='...'), {...})
            (AIMessageChunk(content=' is', additional_kwargs={}, response_metadata={}, id='...'), {...})
            (AIMessageChunk(content=' Paris', additional_kwargs={}, response_metadata={}, id='...'), {...})
            ```
        """

        stream = AsyncQueue()
        aioloop = asyncio.get_running_loop()
        stream_put = cast(
            Callable[[StreamChunk], None],
            partial(aioloop.call_soon_threadsafe, stream.put_nowait),
        )

        def output() -> Iterator:
            while True:
                try:
                    ns, mode, payload = stream.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if subgraphs and isinstance(stream_mode, list):
                    yield (ns, mode, payload)
                elif isinstance(stream_mode, list):
                    yield (mode, payload)
                elif subgraphs:
                    yield (ns, payload)
                else:
                    yield payload

        config = ensure_config(self.config, config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name", self.get_name()),
            run_id=config.get("run_id"),
        )
        # if running from astream_log() run each proc with streaming
        do_stream = next(
            (
                cast(_StreamingCallbackHandler, h)
                for h in run_manager.handlers
                if isinstance(h, _StreamingCallbackHandler)
            ),
            None,
        )
        try:
            # assign defaults
            (
                debug,
                stream_modes,
                output_keys,
                interrupt_before_,
                interrupt_after_,
                checkpointer,
                store,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
            )
            # set up subgraph checkpointing
            if self.checkpointer is True:
                ns = cast(str, config[CONF][CONFIG_KEY_CHECKPOINT_NS])
                config[CONF][CONFIG_KEY_CHECKPOINT_NS] = recast_checkpoint_ns(ns)
            # set up messages stream mode
            if "messages" in stream_modes:
                run_manager.inheritable_handlers.append(
                    StreamMessagesHandler(stream_put)
                )
            # set up custom stream mode
            if "custom" in stream_modes:
                config[CONF][CONFIG_KEY_STREAM_WRITER] = (
                    lambda c: aioloop.call_soon_threadsafe(
                        stream.put_nowait, ((), "custom", c)
                    )
                )
            async with AsyncPregelLoop(
                input,
                input_model=self.input_model,
                stream=StreamProtocol(stream.put_nowait, stream_modes),
                config=config,
                store=store,
                checkpointer=checkpointer,
                nodes=self.nodes,
                specs=self.channels,
                output_keys=output_keys,
                stream_keys=self.stream_channels_asis,
                interrupt_before=interrupt_before_,
                interrupt_after=interrupt_after_,
                manager=run_manager,
                debug=debug,
                # `self.nodes` can be modified after creation of `Pregel`. For example,
                # that's how StateGraph compilation currently works.
                # For now, we recompute the trigger_to_nodes mapping every time the
                # loop is created. We could potentially memoize this if it becomes a
                # performance issue.
                trigger_to_nodes=_trigger_to_nodes(self.nodes),
            ) as loop:
                # create runner
                runner = PregelRunner(
                    submit=config[CONF].get(
                        CONFIG_KEY_RUNNER_SUBMIT, weakref.WeakMethod(loop.submit)
                    ),
                    put_writes=weakref.WeakMethod(loop.put_writes),
                    schedule_task=weakref.WeakMethod(loop.accept_push),
                    use_astream=do_stream is not None,
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
                while loop.tick(input_keys=self.input_channels):
                    async for _ in runner.atick(
                        loop.tasks.values(),
                        timeout=self.step_timeout,
                        retry_policy=self.retry_policy,
                        get_waiter=get_waiter,
                    ):
                        # emit output
                        for o in output():
                            yield o
            # emit output
            for o in output():
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
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: StreamMode = "values",
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: Optional[bool] = None,
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
            debug: Optional. Enable debug mode for the graph run.
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
            stream_mode=stream_mode,
            output_keys=output_keys,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
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

    async def ainvoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: StreamMode = "values",
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        """Asynchronously invoke the graph on a single input.

        Args:
            input: The input data for the computation. It can be a dictionary or any other type.
            config: Optional. The configuration for the computation.
            stream_mode: Optional. The stream mode for the computation. Default is "values".
            output_keys: Optional. The output keys to include in the result. Default is None.
            interrupt_before: Optional. The nodes to interrupt before. Default is None.
            interrupt_after: Optional. The nodes to interrupt after. Default is None.
            debug: Optional. Whether to enable debug mode. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the computation. If stream_mode is "values", it returns the latest value.
            If stream_mode is "chunks", it returns a list of chunks.
        """

        output_keys = output_keys if output_keys is not None else self.output_channels
        if stream_mode == "values":
            latest: Union[dict[str, Any], Any] = None
        else:
            chunks = []
        async for chunk in self.astream(
            input,
            config,
            stream_mode=stream_mode,
            output_keys=output_keys,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
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


def _trigger_to_nodes(nodes: dict[str, PregelNode]) -> Mapping[str, Sequence[str]]:
    """Index from a trigger to nodes that depend on it."""
    trigger_to_nodes: defaultdict[str, list[str]] = defaultdict(list)
    for name, node in nodes.items():
        for trigger in node.triggers:
            trigger_to_nodes[trigger].append(name)
    return dict(trigger_to_nodes)
