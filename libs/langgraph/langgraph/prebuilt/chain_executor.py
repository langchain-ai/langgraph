from typing import Any, Optional, Type, Union, cast

from langchain_core.runnables.base import Runnable, RunnableLike

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import END, START, CompiledStateGraph, StateGraph
from langgraph.store.base import BaseStore


def _get_name(step: RunnableLike) -> str:
    if isinstance(step, Runnable):
        if step.name is None:
            raise ValueError(
                f"Runnable ({step}) needs to have a name attribute. "
                "Consider setting the name or passing it as a tuple (name, runnable)."
            )
        return step.name
    elif callable(step):
        return getattr(step, "__name__", step.__class__.__name__)
    else:
        raise TypeError(f"Unsupported step type: {step}")


def create_chain_executor(
    *steps: Union[RunnableLike, tuple[str, RunnableLike]],
    state_schema: Type[Any],
    return_compiled: bool = True,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
) -> Union[CompiledStateGraph, StateGraph]:
    """Creates a chain executor graph that runs a series of provided steps in order.

    Args:
        *steps: A sequence of RunnableLike objects or (name, RunnableLike) tuples.
            If no names are provided, the name will be inferred from the step object (e.g. a runnable or a callable name).
            Each step will be executed in the order provided.
        state_schema: The state schema for the graph.
        return_compiled: Whether to return the compiled graph or the builder object.
            If False, all of the arguments except `steps` and `state_schema` will be ignored.
            Defaults to True (return compiled graph).
        checkpointer: An optional checkpoint saver object. This is used for persisting
            the state of the graph (e.g., as chat memory) for a single thread (e.g., a single conversation).
        store: An optional store object. This is used for persisting data
            across multiple threads (e.g., multiple conversations / users).
        interrupt_before: An optional list of step names to interrupt before execution.
        interrupt_after: An optional list of step names to interrupt after execution.
        debug: A flag to enable debug mode.

    Returns:
        A CompiledStateGraph object if `return_compiled` is True, otherwise a StateGraph object.
    """
    if len(steps) < 1:
        raise ValueError("Sequential executor requires at least one step.")

    node_names = set()
    builder = StateGraph(state_schema)
    previous_name: Optional[str] = None
    for step in steps:
        if isinstance(step, tuple) and len(step) == 2:
            name, step = step
        else:
            name = _get_name(step)

        if name in node_names:
            raise ValueError(f"Node name {name} already exists.")

        node_names.add(name)
        builder.add_node(name, step)
        if previous_name is None:
            builder.add_edge(START, name)
        else:
            builder.add_edge(previous_name, name)

        previous_name = name

    builder.add_edge(cast(str, previous_name), END)
    if not return_compiled:
        return builder

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )
