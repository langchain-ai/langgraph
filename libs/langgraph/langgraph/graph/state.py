import inspect
import logging
import typing
import warnings
from functools import partial
from inspect import isclass, isfunction, signature
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    Union,
    get_origin,
    get_type_hints,
    overload,
)

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.base import RunnableLike
from langchain_core.runnables.utils import create_model
from pydantic import BaseModel

from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.dynamic_barrier_value import DynamicBarrierValue, WaitForNames
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import NS_END, NS_SEP, TAG_HIDDEN
from langgraph.errors import InvalidUpdateError
from langgraph.graph.graph import END, START, Branch, CompiledGraph, Graph, Send
from langgraph.managed.base import (
    ChannelKeyPlaceholder,
    ChannelTypePlaceholder,
    ConfiguredManagedValue,
    ManagedValueSpec,
    is_managed_value,
    is_writable_managed_value,
)
from langgraph.pregel.read import ChannelRead, PregelNode
from langgraph.pregel.types import All, RetryPolicy
from langgraph.pregel.write import SKIP_WRITE, ChannelWrite, ChannelWriteEntry
from langgraph.store.base import BaseStore
from langgraph.utils import RunnableCallable, coerce_to_runnable, get_field_default

logger = logging.getLogger(__name__)


def _warn_invalid_state_schema(schema: Union[Type[Any], Any]) -> None:
    if isinstance(schema, type):
        return
    if typing.get_args(schema):
        return
    warnings.warn(
        f"Invalid state_schema: {schema}. Expected a type or Annotated[type, reducer]. "
        "Please provide a valid schema to ensure correct updates.\n"
        " See: https://langchain-ai.github.io/langgraph/reference/graphs/#stategraph"
    )


class StateNodeSpec(NamedTuple):
    runnable: Runnable
    metadata: dict[str, Any]
    input: Type[Any]
    retry_policy: Optional[RetryPolicy]


class StateGraph(Graph):
    """A graph whose nodes communicate by reading and writing to a shared state.
    The signature of each node is State -> Partial<State>.

    Each state key can optionally be annotated with a reducer function that
    will be used to aggregate the values of that key received from multiple nodes.
    The signature of a reducer function is (Value, Value) -> Value.

    Args:
        state_schema (Type[Any]): The schema class that defines the state.
        config_schema (Optional[Type[Any]]): The schema class that defines the configuration.
            Use this to expose configurable parameters in your API.


    Examples:
        >>> from langchain_core.runnables import RunnableConfig
        >>> from typing_extensions import Annotated, TypedDict
        >>> from langgraph.checkpoint.memory import MemorySaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> def reducer(a: list, b: int | None) -> int:
        ...     if b is not None:
        ...         return a + [b]
        ...     return a
        >>>
        >>> class State(TypedDict):
        ...     x: Annotated[list, reducer]
        >>>
        >>> class ConfigSchema(TypedDict):
        ...     r: float
        >>>
        >>> graph = StateGraph(State, config_schema=ConfigSchema)
        >>>
        >>> def node(state: State, config: RunnableConfig) -> dict:
        ...     r = config["configurable"].get("r", 1.0)
        ...     x = state["x"][-1]
        ...     next_value = x * r * (1 - x)
        ...     return {"x": next_value}
        >>>
        >>> graph.add_node("A", node)
        >>> graph.set_entry_point("A")
        >>> graph.set_finish_point("A")
        >>> compiled = graph.compile()
        >>>
        >>> print(compiled.config_specs)
        [ConfigurableFieldSpec(id='r', annotation=<class 'float'>, name=None, description=None, default=None, is_shared=False, dependencies=None)]
        >>>
        >>> step1 = compiled.invoke({"x": 0.5}, {"configurable": {"r": 3.0}})
        >>> print(step1)
        {'x': [0.5, 0.75]}"""

    nodes: dict[str, StateNodeSpec]
    channels: dict[str, BaseChannel]
    managed: dict[str, ManagedValueSpec]
    schemas: dict[Type[Any], dict[str, Union[BaseChannel, ManagedValueSpec]]]

    def __init__(
        self,
        state_schema: Optional[Type[Any]] = None,
        config_schema: Optional[Type[Any]] = None,
        *,
        input: Optional[Type[Any]] = None,
        output: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if state_schema is None:
            if input is None or output is None:
                raise ValueError("Must provide state_schema or input and output")
            state_schema = input
        else:
            if input is None:
                input = state_schema
            if output is None:
                output = state_schema
        self.schemas = {}
        self.channels = {}
        self.managed = {}
        self.schema = state_schema
        self.input = input
        self.output = output
        self._add_schema(state_schema)
        self._add_schema(input)
        self._add_schema(output)
        self.config_schema = config_schema
        self.waiting_edges: set[tuple[tuple[str, ...], str]] = set()

    @property
    def _all_edges(self) -> set[tuple[str, str]]:
        return self.edges | {
            (start, end) for starts, end in self.waiting_edges for start in starts
        }

    def _add_schema(self, schema: Type[Any]) -> None:
        if schema not in self.schemas:
            _warn_invalid_state_schema(schema)
            channels, managed = _get_channels(schema)
            self.schemas[schema] = {**channels, **managed}
            for key, channel in channels.items():
                if key in self.channels:
                    if self.channels[key] != channel:
                        if isinstance(channel, LastValue):
                            pass
                        else:
                            raise ValueError(
                                f"Channel '{key}' already exists with a different type"
                            )
                else:
                    self.channels[key] = channel
            for key, managed in managed.items():
                if key in self.managed:
                    if self.managed[key] != managed:
                        raise ValueError(
                            f"Managed value '{key}' already exists with a different type"
                        )
                else:
                    self.managed[key] = managed

    @overload
    def add_node(
        self,
        node: RunnableLike,
        *,
        metadata: Optional[dict[str, Any]] = None,
        input: Optional[Type[Any]] = None,
        retry: Optional[RetryPolicy] = None,
    ) -> None:
        """Adds a new node to the state graph.
        Will take the name of the function/runnable as the node name.

        Args:
            node (RunnableLike): The function or runnable this node will run.

        Raises:
            ValueError: If the key is already being used as a state key.

        Returns:
            None
        """
        ...

    @overload
    def add_node(
        self,
        node: str,
        action: RunnableLike,
        *,
        metadata: Optional[dict[str, Any]] = None,
        input: Optional[Type[Any]] = None,
        retry: Optional[RetryPolicy] = None,
    ) -> None:
        """Adds a new node to the state graph.

        Args:
            node (str): The key of the node.
            action (RunnableLike): The action associated with the node.

        Raises:
            ValueError: If the key is already being used as a state key.

        Returns:
            None
        """
        ...

    def add_node(
        self,
        node: Union[str, RunnableLike],
        action: Optional[RunnableLike] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
        input: Optional[Type[Any]] = None,
        retry: Optional[RetryPolicy] = None,
    ) -> None:
        """Adds a new node to the state graph.

        Will take the name of the function/runnable as the node name.

        Args:
            node (Union[str, RunnableLike)]: The function or runnable this node will run.
            action (Optional[RunnableLike]): The action associated with the node. (default: None)
            metadata (Optional[dict[str, Any]]): The metadata associated with the node. (default: None)
            input (Optional[Type[Any]]): The input schema for the node. (default: the graph's input schema)
            retry (Optional[RetryPolicy]): The policy for retrying the node. (default: None)
        Raises:
            ValueError: If the key is already being used as a state key.


        Examples:
            ```pycon
            >>> from langgraph.graph import START, StateGraph
            ...
            >>> def my_node(state, config):
            ...    return {"x": state["x"] + 1}
            ...
            >>> builder = StateGraph(dict)
            >>> builder.add_node(my_node)  # node name will be 'my_node'
            >>> builder.add_edge(START, "my_node")
            >>> graph = builder.compile()
            >>> graph.invoke({"x": 1})
            {'x': 2}
            ```
            Customize the name:

            ```pycon
            >>> builder = StateGraph(dict)
            >>> builder.add_node("my_fair_node", my_node)
            >>> builder.add_edge(START, "my_fair_node")
            >>> graph = builder.compile()
            >>> graph.invoke({"x": 1})
            {'x': 2}
            ```

        Returns:
            None
        """
        if not isinstance(node, str):
            action = node
            if isinstance(action, Runnable):
                node = action.name
            else:
                node = getattr(action, "__name__", action.__class__.__name__)
            if node is None:
                raise ValueError(
                    "Node name must be provided if action is not a function"
                )
        if node in self.channels:
            raise ValueError(f"'{node}' is already being used as a state key")
        if self.compiled:
            logger.warning(
                "Adding a node to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        if not isinstance(node, str):
            action = node
            node = getattr(action, "name", action.__name__)
        if node in self.nodes:
            raise ValueError(f"Node `{node}` already present.")
        if node == END or node == START:
            raise ValueError(f"Node `{node}` is reserved.")

        for character in (NS_SEP, NS_END):
            if character in node:
                raise ValueError(
                    f"'{character}' is a reserved character and is not allowed in the node names."
                )

        try:
            if isfunction(action) and (
                hints := get_type_hints(action.__call__) or get_type_hints(action)
            ):
                if input is None:
                    first_parameter_name = next(
                        iter(inspect.signature(action).parameters.keys())
                    )
                    if input_hint := hints.get(first_parameter_name):
                        if isinstance(input_hint, type) and get_type_hints(input_hint):
                            input = input_hint
        except (TypeError, StopIteration):
            pass
        if input is not None:
            self._add_schema(input)
        self.nodes[node] = StateNodeSpec(
            coerce_to_runnable(action, name=node, trace=False),
            metadata,
            input=input or self.schema,
            retry_policy=retry,
        )

    def add_edge(self, start_key: Union[str, list[str]], end_key: str) -> None:
        """Adds a directed edge from the start node to the end node.

        If the graph transitions to the start_key node, it will always transition to the end_key node next.

        Args:
            start_key (Union[str, list[str]]): The key(s) of the start node(s) of the edge.
            end_key (str): The key of the end node of the edge.

        Raises:
            ValueError: If the start key is 'END' or if the start key or end key is not present in the graph.

        Returns:
            None
        """
        if isinstance(start_key, str):
            return super().add_edge(start_key, end_key)

        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        for start in start_key:
            if start == END:
                raise ValueError("END cannot be a start node")
            if start not in self.nodes:
                raise ValueError(f"Need to add_node `{start}` first")
        if end_key == START:
            raise ValueError("START cannot be an end node")
        if end_key != END and end_key not in self.nodes:
            raise ValueError(f"Need to add_node `{end_key}` first")

        self.waiting_edges.add((tuple(start_key), end_key))

    def compile(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        *,
        store: Optional[BaseStore] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: bool = False,
    ) -> "CompiledStateGraph":
        """Compiles the state graph into a `CompiledGraph` object.

        The compiled graph implements the `Runnable` interface and can be invoked,
        streamed, batched, and run asynchronously.

        Args:
            checkpointer (Optional[BaseCheckpointSaver]): An optional checkpoint saver object.
                This serves as a fully versioned "memory" for the graph, allowing
                the graph to be paused and resumed, and replayed from any point.
            interrupt_before (Optional[Sequence[str]]): An optional list of node names to interrupt before.
            interrupt_after (Optional[Sequence[str]]): An optional list of node names to interrupt after.
            debug (bool): A flag indicating whether to enable debug mode.

        Returns:
            CompiledStateGraph: The compiled state graph.
        """
        # assign default values
        interrupt_before = interrupt_before or []
        interrupt_after = interrupt_after or []

        # validate the graph
        self.validate(
            interrupt=(
                (interrupt_before if interrupt_before != "*" else []) + interrupt_after
                if interrupt_after != "*"
                else []
            )
        )

        # prepare output channels
        output_channels = (
            "__root__"
            if len(self.schemas[self.output]) == 1
            and "__root__" in self.schemas[self.output]
            else [
                key
                for key, val in self.schemas[self.output].items()
                if not is_managed_value(val)
            ]
        )
        stream_channels = (
            "__root__"
            if len(self.channels) == 1 and "__root__" in self.channels
            else [
                key for key, val in self.channels.items() if not is_managed_value(val)
            ]
        )

        compiled = CompiledStateGraph(
            builder=self,
            config_type=self.config_schema,
            nodes={},
            channels={
                **self.channels,
                **self.managed,
                START: EphemeralValue(self.input),
            },
            input_channels=START,
            stream_mode="updates",
            output_channels=output_channels,
            stream_channels=stream_channels,
            checkpointer=checkpointer,
            interrupt_before_nodes=interrupt_before,
            interrupt_after_nodes=interrupt_after,
            auto_validate=False,
            debug=debug,
            store=store,
        )

        compiled.attach_node(START, None)
        for key, node in self.nodes.items():
            compiled.attach_node(key, node)

        for start, end in self.edges:
            compiled.attach_edge(start, end)

        for starts, end in self.waiting_edges:
            compiled.attach_edge(starts, end)

        for start, branches in self.branches.items():
            for name, branch in branches.items():
                compiled.attach_branch(start, name, branch)

        return compiled.validate()


class CompiledStateGraph(CompiledGraph):
    builder: StateGraph

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        if isclass(self.builder.input) and issubclass(self.builder.input, BaseModel):
            return self.builder.input
        else:
            keys = list(self.builder.schemas[self.builder.input].keys())
            if len(keys) == 1 and keys[0] == "__root__":
                return create_model(  # type: ignore[call-overload]
                    self.get_name("Input"),
                    __root__=(self.channels[keys[0]].UpdateType, None),
                )
            else:
                return create_model(  # type: ignore[call-overload]
                    self.get_name("Input"),
                    **{
                        k: (
                            self.channels[k].UpdateType,
                            (
                                get_field_default(
                                    k,
                                    self.channels[k].UpdateType,
                                    self.builder.input,
                                )
                            ),
                        )
                        for k in self.builder.schemas[self.builder.input]
                        if isinstance(self.channels[k], BaseChannel)
                    },
                )

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        if isclass(self.builder.output) and issubclass(self.builder.output, BaseModel):
            return self.builder.output

        return super().get_output_schema(config)

    def attach_node(self, key: str, node: Optional[StateNodeSpec]) -> None:
        if key == START:
            output_keys = [
                k
                for k, v in self.builder.schemas[self.builder.input].items()
                if not is_managed_value(v)
            ]
        else:
            output_keys = list(self.builder.channels) + [
                k
                for k, v in self.builder.managed.items()
                if is_writable_managed_value(v)
            ]

        def _get_state_key(
            input: Union[None, dict, Any], config: RunnableConfig, *, key: str
        ) -> Any:
            if input is None:
                return SKIP_WRITE
            elif isinstance(input, dict):
                return input.get(key, SKIP_WRITE)
            elif get_type_hints(type(input)):
                value = getattr(input, key, SKIP_WRITE)
                return value if value is not None else SKIP_WRITE
            else:
                raise InvalidUpdateError(f"Expected dict, got {input}")

        # state updaters
        write_entries = (
            [ChannelWriteEntry("__root__", skip_none=True)]
            if output_keys == ["__root__"]
            else [
                ChannelWriteEntry(
                    key,
                    mapper=RunnableCallable(
                        _get_state_key, key=key, trace=False, recurse=False
                    ),
                )
                for key in output_keys
            ]
        )

        # add node and output channel
        if key == START:
            self.nodes[key] = PregelNode(
                tags=[TAG_HIDDEN],
                triggers=[START],
                channels=[START],
                writers=[
                    ChannelWrite(
                        write_entries,
                        tags=[TAG_HIDDEN],
                        require_at_least_one_of=output_keys,
                    ),
                ],
            )
        else:
            input_schema = node.input if node else self.builder.schema
            input_values = {k: k for k in self.builder.schemas[input_schema]}
            is_single_input = len(input_values) == 1 and "__root__" in input_values

            self.channels[key] = EphemeralValue(Any, guard=False)
            self.nodes[key] = PregelNode(
                triggers=[],
                # read state keys and managed values
                channels=(list(input_values) if is_single_input else input_values),
                # coerce state dict to schema class (eg. pydantic model)
                mapper=(
                    None if is_single_input else partial(_coerce_state, input_schema)
                ),
                writers=[
                    # publish to this channel and state keys
                    ChannelWrite(
                        [ChannelWriteEntry(key, key)] + write_entries,
                        tags=[TAG_HIDDEN],
                        require_at_least_one_of=output_keys,
                    ),
                ],
                metadata=node.metadata,
                retry_policy=node.retry_policy,
            ).pipe(node.runnable)

    def attach_edge(self, starts: Union[str, Sequence[str]], end: str) -> None:
        if isinstance(starts, str):
            if starts == START:
                channel_name = f"start:{end}"
                # register channel
                self.channels[channel_name] = EphemeralValue(Any)
                # subscribe to channel
                self.nodes[end].triggers.append(channel_name)
                # publish to channel
                self.nodes[START] |= ChannelWrite(
                    [ChannelWriteEntry(channel_name, START)], tags=[TAG_HIDDEN]
                )
            elif end != END:
                # subscribe to start channel
                self.nodes[end].triggers.append(starts)
        elif end != END:
            channel_name = f"join:{'+'.join(starts)}:{end}"
            # register channel
            self.channels[channel_name] = NamedBarrierValue(str, set(starts))
            # subscribe to channel
            self.nodes[end].triggers.append(channel_name)
            # publish to channel
            for start in starts:
                self.nodes[start] |= ChannelWrite(
                    [ChannelWriteEntry(channel_name, start)], tags=[TAG_HIDDEN]
                )

    def attach_branch(self, start: str, name: str, branch: Branch) -> None:
        def branch_writer(packets: list[Union[str, Send]]) -> Optional[ChannelWrite]:
            if filtered := [p for p in packets if p != END]:
                writes = [
                    (
                        ChannelWriteEntry(f"branch:{start}:{name}:{p}", start)
                        if not isinstance(p, Send)
                        else p
                    )
                    for p in filtered
                ]
                if branch.then and branch.then != END:
                    writes.append(
                        ChannelWriteEntry(
                            f"branch:{start}:{name}::then",
                            WaitForNames(
                                {p.node if isinstance(p, Send) else p for p in filtered}
                            ),
                        )
                    )
                return ChannelWrite(writes, tags=[TAG_HIDDEN])

        # attach branch publisher
        schema = (
            self.builder.nodes[start].input
            if start in self.builder.nodes
            else self.builder.schema
        )
        self.nodes[start] |= branch.run(
            branch_writer, _get_state_reader(self.builder, schema)
        )

        # attach branch subscribers
        ends = (
            branch.ends.values()
            if branch.ends
            else [node for node in self.builder.nodes if node != branch.then]
        )
        for end in ends:
            if end != END:
                channel_name = f"branch:{start}:{name}:{end}"
                self.channels[channel_name] = EphemeralValue(Any, guard=False)
                self.nodes[end].triggers.append(channel_name)

        # attach then subscriber
        if branch.then and branch.then != END:
            channel_name = f"branch:{start}:{name}::then"
            self.channels[channel_name] = DynamicBarrierValue(str)
            self.nodes[branch.then].triggers.append(channel_name)
            for end in ends:
                if end != END:
                    self.nodes[end] |= ChannelWrite(
                        [ChannelWriteEntry(channel_name, end)], tags=[TAG_HIDDEN]
                    )


def _get_state_reader(
    builder: StateGraph, schema: Type[Any]
) -> Callable[[RunnableConfig], Any]:
    state_keys = list(builder.channels)
    select = list(builder.schemas[schema])
    return partial(
        ChannelRead.do_read,
        select=select[0] if select == ["__root__"] else select,
        fresh=True,
        # coerce state dict to schema class (eg. pydantic model)
        mapper=(None if state_keys == ["__root__"] else partial(_coerce_state, schema)),
    )


def _coerce_state(schema: Type[Any], input: dict[str, Any]) -> dict[str, Any]:
    return schema(**input)


def _get_channels(
    schema: Type[dict],
) -> tuple[dict[str, BaseChannel], dict[str, ManagedValueSpec]]:
    if not hasattr(schema, "__annotations__"):
        return {"__root__": _get_channel("__root__", schema, allow_managed=False)}, {}

    all_keys = {
        name: _get_channel(name, typ)
        for name, typ in get_type_hints(schema, include_extras=True).items()
        if name != "__slots__"
    }
    return (
        {k: v for k, v in all_keys.items() if not is_managed_value(v)},
        {k: v for k, v in all_keys.items() if is_managed_value(v)},
    )


def _get_channel(
    name: str, annotation: Any, *, allow_managed: bool = True
) -> Union[BaseChannel, ManagedValueSpec]:
    if manager := _is_field_managed_value(name, annotation):
        if allow_managed:
            return manager
        else:
            raise ValueError(f"This {annotation} not allowed in this position")
    elif channel := _is_field_channel(annotation):
        channel.key = name
        return channel
    elif channel := _is_field_binop(annotation):
        channel.key = name
        return channel

    fallback = LastValue(annotation)
    fallback.key = name
    return fallback


def _is_field_channel(typ: Type[Any]) -> Optional[BaseChannel]:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and isinstance(meta[-1], BaseChannel):
            return meta[-1]
        elif len(meta) >= 1 and isclass(meta[-1]) and issubclass(meta[-1], BaseChannel):
            return meta[-1](typ.__origin__ if hasattr(typ, "__origin__") else typ)
    return None


def _is_field_binop(typ: Type[Any]) -> Optional[BinaryOperatorAggregate]:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and callable(meta[-1]):
            sig = signature(meta[0])
            params = list(sig.parameters.values())
            if len(params) == 2 and all(
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params
            ):
                return BinaryOperatorAggregate(typ, meta[0])
            else:
                raise ValueError(
                    f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
                )
    return None


def _is_field_managed_value(name: str, typ: Type[Any]) -> Optional[ManagedValueSpec]:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1:
            decoration = get_origin(meta[-1]) or meta[-1]
            if is_managed_value(decoration):
                if isinstance(decoration, ConfiguredManagedValue):
                    for k, v in decoration.kwargs.items():
                        if v is ChannelKeyPlaceholder:
                            decoration.kwargs[k] = name
                        if v is ChannelTypePlaceholder:
                            decoration.kwargs[k] = typ.__origin__
                return decoration

    return None
