import inspect
import logging
import typing
import warnings
from collections import defaultdict
from functools import partial
from inspect import isclass, isfunction, ismethod, signature
from types import FunctionType
from typing import (
    Any,
    Callable,
    Hashable,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from typing_extensions import Self

from langgraph._api.deprecation import LangGraphDeprecationWarning
from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.dynamic_barrier_value import DynamicBarrierValue, WaitForNames
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.constants import (
    EMPTY_SEQ,
    END,
    MISSING,
    NS_END,
    NS_SEP,
    SELF,
    START,
    TAG_HIDDEN,
)
from langgraph.errors import (
    ErrorCode,
    InvalidUpdateError,
    ParentCommand,
    create_error_message,
)
from langgraph.pregel import Pregel
from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.read import ChannelRead, PregelNode
from langgraph.pregel.write import (
    ChannelWrite,
    ChannelWriteEntry,
    ChannelWriteTupleEntry,
)
from langgraph.store.base import BaseStore
from langgraph.types import All, Checkpointer, Command, RetryPolicy, Send
from langgraph.utils.config import RunnableConfig
from langgraph.utils.runnable import (
    Runnable,
    RunnableCallable,
    RunnableLike,
    coerce_to_runnable,
)

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


def _get_node_name(node: RunnableLike) -> str:
    if isinstance(node, Runnable):
        return node.get_name()
    elif callable(node):
        return getattr(node, "__name__", node.__class__.__name__)
    else:
        raise TypeError(f"Unsupported node type: {type(node)}")


class Branch(NamedTuple):
    path: Runnable[Any, Union[Hashable, list[Hashable]]]
    ends: Optional[dict[Hashable, str]]
    then: Optional[str] = None

    def run(
        self,
        writer: Callable[
            [Sequence[Union[str, Send]], RunnableConfig], Optional[ChannelWrite]
        ],
        reader: Optional[Callable[[RunnableConfig], Any]] = None,
    ) -> RunnableCallable:
        return RunnableCallable(
            func=self._route,
            writer=writer,
            reader=reader,
            name=None,
            trace=False,
        )

    def _route(
        self,
        input: Any,
        config: RunnableConfig,
        *,
        reader: Optional[Callable[[RunnableConfig], Any]],
        writer: Callable[
            [Sequence[Union[str, Send]], RunnableConfig], Optional[ChannelWrite]
        ],
    ) -> Runnable:
        if reader:
            value = reader(config)
            # passthrough additional keys from node to branch
            # only doable when using dict states
            if isinstance(value, dict) and isinstance(input, dict):
                value = {**input, **value}
        else:
            value = input
        result = self.path.invoke(value, config)
        return self._finish(writer, input, result, config)

    def _finish(
        self,
        writer: Callable[
            [Sequence[Union[str, Send]], RunnableConfig], Optional[ChannelWrite]
        ],
        input: Any,
        result: Any,
        config: RunnableConfig,
    ) -> Union[Runnable, Any]:
        if not isinstance(result, (list, tuple)):
            result = [result]
        if self.ends:
            destinations: Sequence[Union[Send, str]] = [
                r if isinstance(r, Send) else self.ends[r] for r in result
            ]
        else:
            destinations = cast(Sequence[Union[Send, str]], result)
        if any(dest is None or dest == START for dest in destinations):
            raise ValueError("Branch did not return a valid destination")
        if any(p.node == END for p in destinations if isinstance(p, Send)):
            raise InvalidUpdateError("Cannot send a packet to the END node")
        return writer(destinations, config) or input


class StateNodeSpec(NamedTuple):
    runnable: Runnable
    metadata: Optional[dict[str, Any]]
    input: Type[Any]
    retry_policy: Optional[RetryPolicy]
    ends: Optional[Union[tuple[str, ...], dict[str, str]]] = EMPTY_SEQ
    subgraphs: Optional[list[PregelProtocol]] = EMPTY_SEQ


class StateGraph:
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
        >>> from langgraph.utils.config import RunnableConfig
        >>> from typing_extensions import Annotated, TypedDict
        >>> from langgraph.checkpoint.memory import MemorySaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> def reducer(a: list, b: int | None) -> list:
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

    nodes: dict[str, StateNodeSpec]  # type: ignore[assignment]
    channels: dict[str, BaseChannel]
    schemas: dict[Type[Any], dict[str, BaseChannel]]

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
            warnings.warn(
                "Initializing StateGraph without state_schema is deprecated. "
                "Please pass in an explicit state_schema instead of just an input and output schema.",
                LangGraphDeprecationWarning,
                stacklevel=2,
            )
        else:
            if input is None:
                input = state_schema
            if output is None:
                output = state_schema

        self.nodes: dict[str, StateNodeSpec] = {}
        self.edges = set[tuple[str, str]]()
        self.branches: defaultdict[str, dict[str, Branch]] = defaultdict(dict)
        self.compiled = False

        self.schemas = {}
        self.channels = {}
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
            channels = _get_channels(schema)
            self.schemas[schema] = channels
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

    @overload
    def add_node(
        self,
        node: RunnableLike,
        *,
        metadata: Optional[dict[str, Any]] = None,
        input: Optional[Type[Any]] = None,
        retry: Optional[RetryPolicy] = None,
        destinations: Optional[Union[dict[str, str], tuple[str]]] = None,
    ) -> Self:
        """Adds a new node to the state graph.
        Will take the name of the function/runnable as the node name.

        Args:
            node (RunnableLike): The function or runnable this node will run.

        Raises:
            ValueError: If the key is already being used as a state key.

        Returns:
            Self: The instance of the state graph, allowing for method chaining.
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
        destinations: Optional[Union[dict[str, str], tuple[str]]] = None,
    ) -> Self:
        """Adds a new node to the state graph.

        Args:
            node (str): The key of the node.
            action (RunnableLike): The action associated with the node.

        Raises:
            ValueError: If the key is already being used as a state key.

        Returns:
            Self: The instance of the state graph, allowing for method chaining.
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
        destinations: Optional[Union[dict[str, str], tuple[str]]] = None,
        subgraphs: list[PregelProtocol] = EMPTY_SEQ,
    ) -> Self:
        """Adds a new node to the state graph.

        Will take the name of the function/runnable as the node name.

        Args:
            node (Union[str, RunnableLike]): The function or runnable this node will run.
            action (Optional[RunnableLike]): The action associated with the node. (default: None)
            metadata (Optional[dict[str, Any]]): The metadata associated with the node. (default: None)
            input (Optional[Type[Any]]): The input schema for the node. (default: the graph's input schema)
            retry (Optional[RetryPolicy]): The policy for retrying the node. (default: None)
            destinations (Optional[Union[dict[str, str], tuple[str]]]): Destinations that indicate where a node can route to.
                This is useful for edgeless graphs with nodes that return `Command` objects.
                If a dict is provided, the keys will be used as the target node names and the values will be used as the labels for the edges.
                If a tuple is provided, the values will be used as the target node names.
                NOTE: this is only used for graph rendering and doesn't have any effect on the graph execution.
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
            Self: The instance of the state graph, allowing for method chaining.
        """
        if not isinstance(node, str):
            action = node
            if isinstance(action, Runnable):
                node = action.get_name()
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
            node = cast(str, getattr(action, "name", getattr(action, "__name__", None)))
            if node is None:
                raise ValueError(
                    "Node name must be provided if action is not a function"
                )
        if action is None:
            raise RuntimeError
        if node in self.nodes:
            raise ValueError(f"Node `{node}` already present.")
        if node == END or node == START:
            raise ValueError(f"Node `{node}` is reserved.")

        for character in (NS_SEP, NS_END):
            if character in cast(str, node):
                raise ValueError(
                    f"'{character}' is a reserved character and is not allowed in the node names."
                )

        ends: Union[tuple[str, ...], dict[str, str]] = EMPTY_SEQ
        try:
            if (
                isfunction(action)
                or ismethod(action)
                or ismethod(getattr(action, "__call__", None))
            ) and (
                hints := get_type_hints(getattr(action, "__call__"))
                or get_type_hints(action)
            ):
                if input is None:
                    first_parameter_name = next(
                        iter(
                            inspect.signature(
                                cast(FunctionType, action)
                            ).parameters.keys()
                        )
                    )
                    if input_hint := hints.get(first_parameter_name):
                        if isinstance(input_hint, type) and get_type_hints(input_hint):
                            input = input_hint
                if rtn := hints.get("return"):
                    # Handle Union types
                    rtn_origin = get_origin(rtn)
                    if rtn_origin is Union:
                        rtn_args = get_args(rtn)
                        # Look for Command in the union
                        for arg in rtn_args:
                            arg_origin = get_origin(arg)
                            if arg_origin is Command:
                                rtn = arg
                                rtn_origin = arg_origin
                                break

                    # Check if it's a Command type
                    if (
                        rtn_origin is Command
                        and (rargs := get_args(rtn))
                        and get_origin(rargs[0]) is Literal
                        and (vals := get_args(rargs[0]))
                    ):
                        ends = vals
        except (TypeError, StopIteration):
            pass

        if destinations is not None:
            ends = destinations

        if input is not None:
            self._add_schema(input)
        self.nodes[cast(str, node)] = StateNodeSpec(
            coerce_to_runnable(action, name=cast(str, node), trace=False),
            metadata,
            input=input or self.schema,
            retry_policy=retry,
            ends=ends,
            subgraphs=subgraphs,
        )
        return self

    def add_edge(self, start_key: Union[str, list[str]], end_key: str) -> Self:
        """Adds a directed edge from the start node (or list of start nodes) to the end node.

        When a single start node is provided, the graph will wait for that node to complete
        before executing the end node. When multiple start nodes are provided,
        the graph will wait for ALL of the start nodes to complete before executing the end node.

        Args:
            start_key (Union[str, list[str]]): The key(s) of the start node(s) of the edge.
            end_key (str): The key of the end node of the edge.

        Raises:
            ValueError: If the start key is 'END' or if the start key or end key is not present in the graph.

        Returns:
            Self: The instance of the state graph, allowing for method chaining.
        """
        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        if isinstance(start_key, str):
            if start_key == END:
                raise ValueError("END cannot be a start node")
            if end_key == START:
                raise ValueError("START cannot be an end node")

            self.edges.add((start_key, end_key))
            return self

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
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Union[
            Callable[..., Union[Hashable, list[Hashable]]],
            Runnable[Any, Union[Hashable, list[Hashable]]],
        ],
        path_map: Optional[Union[dict[Hashable, str], list[str]]] = None,
        then: Optional[str] = None,
    ) -> Self:
        """Add a conditional edge from the starting node to any number of destination nodes.

        Args:
            source (str): The starting node. This conditional edge will run when
                exiting this node.
            path (Union[Callable, Runnable]): The callable that determines the next
                node or nodes. If not specifying `path_map` it should return one or
                more nodes. If it returns END, the graph will stop execution.
            path_map (Optional[dict[Hashable, str]]): Optional mapping of paths to node
                names. If omitted the paths returned by `path` should be node names.
            then (Optional[str]): The name of a node to execute after the nodes
                selected by `path`.

        Returns:
            Self: The instance of the graph, allowing for method chaining.

        Note: Without typehints on the `path` function's return value (e.g., `-> Literal["foo", "__end__"]:`)
            or a path_map, the graph visualization assumes the edge could transition to any node in the graph.

        """  # noqa: E501
        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        # coerce path_map to a dictionary
        try:
            if isinstance(path_map, dict):
                path_map_ = path_map.copy()
            elif isinstance(path_map, list):
                path_map_ = {name: name for name in path_map}
            elif isinstance(path, Runnable):
                path_map_ = None
            elif rtn_type := get_type_hints(path.__call__).get(  # type: ignore[operator]
                "return"
            ) or get_type_hints(path).get("return"):
                if get_origin(rtn_type) is Literal:
                    path_map_ = {name: name for name in get_args(rtn_type)}
                else:
                    path_map_ = None
            else:
                path_map_ = None
        except Exception:
            path_map_ = None
        # find a name for the condition
        path = coerce_to_runnable(path, name=None, trace=True)
        name = path.name or "condition"
        # validate the condition
        if name in self.branches[source]:
            raise ValueError(
                f"Branch with name `{path.name}` already exists for node " f"`{source}`"
            )
        # save it
        self.branches[source][name] = Branch(path, path_map_, then)
        return self

    def add_sequence(
        self,
        nodes: Sequence[Union[RunnableLike, tuple[str, RunnableLike]]],
    ) -> Self:
        """Add a sequence of nodes that will be executed in the provided order.

        Args:
            nodes: A sequence of RunnableLike objects (e.g. a LangChain Runnable or a callable) or (name, RunnableLike) tuples.
                If no names are provided, the name will be inferred from the node object (e.g. a runnable or a callable name).
                Each node will be executed in the order provided.

        Raises:
            ValueError: if the sequence is empty.
            ValueError: if the sequence contains duplicate node names.

        Returns:
            Self: The instance of the state graph, allowing for method chaining.
        """
        if len(nodes) < 1:
            raise ValueError("Sequence requires at least one node.")

        previous_name: Optional[str] = None
        for node in nodes:
            if isinstance(node, tuple) and len(node) == 2:
                name, node = node
            else:
                name = _get_node_name(node)

            if name in self.nodes:
                raise ValueError(
                    f"Node names must be unique: node with the name '{name}' already exists. "
                    "If you need to use two different runnables/callables with the same name (for example, using `lambda`), please provide them as tuples (name, runnable/callable)."
                )

            self.add_node(name, node)
            if previous_name is not None:
                self.add_edge(previous_name, name)

            previous_name = name

        return self

    def set_entry_point(self, key: str) -> Self:
        """Specifies the first node to be called in the graph.

        Equivalent to calling `add_edge(START, key)`.

        Parameters:
            key (str): The key of the node to set as the entry point.

        Returns:
            Self: The instance of the graph, allowing for method chaining.
        """
        return self.add_edge(START, key)

    def set_conditional_entry_point(
        self,
        path: Union[
            Callable[..., Union[Hashable, list[Hashable]]],
            Runnable[Any, Union[Hashable, list[Hashable]]],
        ],
        path_map: Optional[Union[dict[Hashable, str], list[str]]] = None,
        then: Optional[str] = None,
    ) -> Self:
        """Sets a conditional entry point in the graph.

        Args:
            path (Union[Callable, Runnable]): The callable that determines the next
                node or nodes. If not specifying `path_map` it should return one or
                more nodes. If it returns END, the graph will stop execution.
            path_map (Optional[dict[str, str]]): Optional mapping of paths to node
                names. If omitted the paths returned by `path` should be node names.
            then (Optional[str]): The name of a node to execute after the nodes
                selected by `path`.

        Returns:
            Self: The instance of the graph, allowing for method chaining.
        """
        return self.add_conditional_edges(START, path, path_map, then)

    def set_finish_point(self, key: str) -> Self:
        """Marks a node as a finish point of the graph.

        If the graph reaches this node, it will cease execution.

        Parameters:
            key (str): The key of the node to set as the finish point.

        Returns:
            Self: The instance of the graph, allowing for method chaining.
        """
        return self.add_edge(key, END)

    def validate(self, interrupt: Optional[Sequence[str]] = None) -> Self:
        # assemble sources
        all_sources = {src for src, _ in self._all_edges}
        for start, branches in self.branches.items():
            all_sources.add(start)
            for cond, branch in branches.items():
                if branch.then is not None:
                    if branch.ends is not None:
                        for end in branch.ends.values():
                            if end != END:
                                all_sources.add(end)
                    else:
                        for node in self.nodes:
                            if node != start and node != branch.then:
                                all_sources.add(node)
        for name, spec in self.nodes.items():
            if spec.ends:
                all_sources.add(name)
        # validate sources
        for source in all_sources:
            if source not in self.nodes and source != START:
                raise ValueError(f"Found edge starting at unknown node '{source}'")

        if START not in all_sources:
            raise ValueError(
                "Graph must have an entrypoint: add at least one edge from START to another node"
            )

        # assemble targets
        all_targets = {end for _, end in self._all_edges}
        for start, branches in self.branches.items():
            for cond, branch in branches.items():
                if branch.then is not None:
                    all_targets.add(branch.then)
                if branch.ends is not None:
                    for end in branch.ends.values():
                        if end not in self.nodes and end != END:
                            raise ValueError(
                                f"At '{start}' node, '{cond}' branch found unknown target '{end}'"
                            )
                        all_targets.add(end)
                else:
                    all_targets.add(END)
                    for node in self.nodes:
                        if node != start and node != branch.then:
                            all_targets.add(node)
        for name, spec in self.nodes.items():
            if spec.ends:
                all_targets.update(spec.ends)
        for target in all_targets:
            if target not in self.nodes and target != END:
                raise ValueError(f"Found edge ending at unknown node `{target}`")
        # validate interrupts
        if interrupt:
            for node in interrupt:
                if node not in self.nodes:
                    raise ValueError(f"Interrupt node `{node}` not found")

        self.compiled = True
        return self

    def compile(
        self,
        checkpointer: Checkpointer = None,
        *,
        store: Optional[BaseStore] = None,
        interrupt_before: Optional[Union[All, list[str]]] = None,
        interrupt_after: Optional[Union[All, list[str]]] = None,
        name: Optional[str] = None,
    ) -> "CompiledStateGraph":
        """Compiles the state graph into a `CompiledGraph` object.

        The compiled graph implements the `Runnable` interface and can be invoked,
        streamed, batched, and run asynchronously.

        Args:
            checkpointer (Optional[Union[Checkpointer, Literal[False]]]): A checkpoint saver object or flag.
                If provided, this Checkpointer serves as a fully versioned "short-term memory" for the graph,
                allowing it to be paused, resumed, and replayed from any point.
                If None, it may inherit the parent graph's checkpointer when used as a subgraph.
                If False, it will not use or inherit any checkpointer.
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
            else [key for key, val in self.schemas[self.output].items()]
        )
        stream_channels = (
            "__root__"
            if len(self.channels) == 1 and "__root__" in self.channels
            else [key for key, val in self.channels.items()]
        )

        compiled = CompiledStateGraph(
            builder=self,
            config_type=self.config_schema,
            nodes={},
            channels={
                **self.channels,
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
            store=store,
            name=name or "LangGraph",
        )

        compiled.attach_node(START, None)
        for key, node in self.nodes.items():
            compiled.attach_node(key, node)

        compiled.attach_branch(START, SELF, CONTROL_BRANCH, with_reader=False)
        for key, node in self.nodes.items():
            compiled.attach_branch(key, SELF, CONTROL_BRANCH, with_reader=False)

        for start, end in self.edges:
            compiled.attach_edge(start, end)

        for starts, end in self.waiting_edges:
            compiled.attach_edge(starts, end)

        for start, branches in self.branches.items():
            for name, branch in branches.items():
                compiled.attach_branch(start, name, branch)

        return compiled.validate()


class CompiledStateGraph(Pregel):
    builder: StateGraph

    def __init__(self, *, builder: StateGraph, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.builder = builder

    def attach_node(self, key: str, node: Optional[StateNodeSpec]) -> None:
        if key == START:
            output_keys = [
                k for k, v in self.builder.schemas[self.builder.input].items()
            ]
        else:
            output_keys = list(self.builder.channels)

        def _get_root(input: Any) -> Optional[Sequence[tuple[str, Any]]]:
            if isinstance(input, Command):
                if input.graph == Command.PARENT:
                    return ()
                return input._update_as_tuples()
            elif (
                isinstance(input, (list, tuple))
                and input
                and any(isinstance(i, Command) for i in input)
            ):
                updates: list[tuple[str, Any]] = []
                for i in input:
                    if isinstance(i, Command):
                        if i.graph == Command.PARENT:
                            continue
                        updates.extend(i._update_as_tuples())
                    else:
                        updates.append(("__root__", i))
                return updates
            elif input is not None:
                return [("__root__", input)]

        def _get_updates(
            input: Union[None, dict, Any],
        ) -> Optional[Sequence[tuple[str, Any]]]:
            if input is None:
                return None
            elif isinstance(input, dict):
                return [(k, v) for k, v in input.items() if k in output_keys]
            elif isinstance(input, Command):
                if input.graph == Command.PARENT:
                    return None
                return [
                    (k, v) for k, v in input._update_as_tuples() if k in output_keys
                ]
            elif (
                isinstance(input, (list, tuple))
                and input
                and any(isinstance(i, Command) for i in input)
            ):
                updates: list[tuple[str, Any]] = []
                for i in input:
                    if isinstance(i, Command):
                        if i.graph == Command.PARENT:
                            continue
                        updates.extend(
                            (k, v) for k, v in i._update_as_tuples() if k in output_keys
                        )
                    else:
                        updates.extend(_get_updates(i) or ())
                return updates
            elif get_type_hints(type(input)):
                # if input is a Pydantic model, only update values
                # for the keys that have been explicitly set by the users
                # (this is needed to avoid sending updates for fields with None defaults)
                output_keys_ = output_keys
                # Pydantic v2
                if hasattr(input, "model_fields_set"):
                    output_keys_ = [
                        k for k in output_keys if k in input.model_fields_set
                    ]
                # Pydantic v1
                elif hasattr(input, "__fields_set__"):
                    output_keys_ = [k for k in output_keys if k in input.__fields_set__]

                return [
                    (k, getattr(input, k))
                    for k in output_keys_
                    if getattr(input, k, MISSING) is not MISSING
                ]
            else:
                msg = create_error_message(
                    message=f"Expected dict, got {input}",
                    error_code=ErrorCode.INVALID_GRAPH_NODE_RETURN_VALUE,
                )
                raise InvalidUpdateError(msg)

        # state updaters
        write_entries: list[Union[ChannelWriteEntry, ChannelWriteTupleEntry]] = [
            ChannelWriteTupleEntry(
                mapper=_get_root if output_keys == ["__root__"] else _get_updates
            )
        ]

        # add node and output channel
        if key == START:
            self.nodes[key] = PregelNode(
                tags=[TAG_HIDDEN],
                triggers=[START],
                channels=[START],
                writers=[
                    ChannelWrite(write_entries, tags=[TAG_HIDDEN]),
                ],
            )
        elif node is not None:
            input_schema = node.input if node else self.builder.schema
            input_values = {k: k for k in self.builder.schemas[input_schema]}
            is_single_input = len(input_values) == 1 and "__root__" in input_values

            self.channels[key] = EphemeralValue(Any, guard=False)
            self.nodes[key] = PregelNode(
                triggers=[],
                # read state keys
                channels=(list(input_values) if is_single_input else input_values),
                # coerce state dict to schema class (eg. pydantic model)
                mapper=(
                    None
                    if is_single_input or issubclass(input_schema, dict)
                    else partial(_coerce_state, input_schema)
                ),
                writers=[
                    # publish to this channel and state keys
                    ChannelWrite(
                        write_entries + [ChannelWriteEntry(key, key)],
                        tags=[TAG_HIDDEN],
                    ),
                ],
                metadata=node.metadata,
                retry_policy=node.retry_policy,
                subgraphs=node.subgraphs,
                bound=node.runnable,
            )
        else:
            raise RuntimeError

    def attach_edge(self, starts: Union[str, Sequence[str]], end: str) -> None:
        if isinstance(starts, str):
            if starts == START:
                channel_name = f"start:{end}"
                # register channel
                self.channels[channel_name] = EphemeralValue(Any)
                # subscribe to channel
                self.nodes[end].triggers.append(channel_name)
                # publish to channel
                self.nodes[START].writers.append(
                    ChannelWrite(
                        [ChannelWriteEntry(channel_name, START)], tags=[TAG_HIDDEN]
                    )
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
                self.nodes[start].writers.append(
                    ChannelWrite(
                        [ChannelWriteEntry(channel_name, start)], tags=[TAG_HIDDEN]
                    )
                )

    def attach_branch(
        self, start: str, name: str, branch: Branch, *, with_reader: bool = True
    ) -> None:
        def branch_writer(
            packets: Sequence[Union[str, Send]], config: RunnableConfig
        ) -> None:
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
                ChannelWrite.do_write(
                    config, cast(Sequence[Union[Send, ChannelWriteEntry]], writes)
                )

        # attach branch publisher
        schema = (
            self.builder.nodes[start].input
            if start in self.builder.nodes
            else self.builder.schema
        )
        self.nodes[start].writers.append(
            branch.run(
                branch_writer,
                _get_state_reader(self.builder, schema) if with_reader else None,
            )
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
                    self.nodes[end].writers.append(
                        ChannelWrite(
                            [ChannelWriteEntry(channel_name, end)], tags=[TAG_HIDDEN]
                        )
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
        mapper=(
            None
            if state_keys == ["__root__"] or issubclass(schema, dict)
            else partial(_coerce_state, schema)
        ),
    )


def _coerce_state(schema: Type[Any], input: dict[str, Any]) -> dict[str, Any]:
    return schema(**input)


def _control_branch(value: Any) -> Sequence[Union[str, Send]]:
    if isinstance(value, Send):
        return [value]
    commands: list[Command] = []
    if isinstance(value, Command):
        commands.append(value)
    elif isinstance(value, (list, tuple)):
        for cmd in value:
            if isinstance(cmd, Command):
                commands.append(cmd)
    rtn: list[Union[str, Send]] = []
    for command in commands:
        if command.graph == Command.PARENT:
            raise ParentCommand(command)
        if isinstance(command.goto, Send):
            rtn.append(command.goto)
        elif isinstance(command.goto, str):
            rtn.append(command.goto)
        else:
            rtn.extend(command.goto)
    return rtn


CONTROL_BRANCH_PATH = RunnableCallable(_control_branch, tags=[TAG_HIDDEN], trace=False)
CONTROL_BRANCH = Branch(CONTROL_BRANCH_PATH, None)


def _get_channels(
    schema: Type[dict],
) -> dict[str, BaseChannel]:
    all_keys = {
        name: _get_channel(name, typ)
        for name, typ in get_type_hints(schema, include_extras=True).items()
        if name != "__slots__"
    }
    return {k: v for k, v in all_keys.items() if isinstance(v, BaseChannel)}


def _get_channel(name: str, annotation: Any) -> BaseChannel:
    if channel := _is_field_channel(annotation):
        channel.key = name
        return channel
    elif channel := _is_field_binop(annotation):
        channel.key = name
        return channel

    fallback: LastValue = LastValue(annotation)
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
            sig = signature(meta[-1])
            params = list(sig.parameters.values())
            if (
                sum(
                    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    for p in params
                )
                == 2
            ):
                return BinaryOperatorAggregate(typ, meta[-1])
            else:
                raise ValueError(
                    f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
                )
    return None
