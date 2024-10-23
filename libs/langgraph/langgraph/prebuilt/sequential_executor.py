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


def create_sequential_executor(
    *steps: Union[RunnableLike, tuple[str, RunnableLike]],
    state_schema: Type[Any],
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
) -> CompiledStateGraph:
    """Creates a sequential executor graph that runs a series of provided steps in order.

    Args:
        *steps: A sequence of RunnableLike objects or (name, RunnableLike) tuples.
            If no names are provided, the name will be inferred from the step object (e.g. a runnable or a callable name).
            Each step will be executed in the order provided.
        state_schema: The state schema for the graph.
        checkpointer: An optional checkpoint saver object. This is used for persisting
            the state of the graph (e.g., as chat memory) for a single thread (e.g., a single conversation).
        store: An optional store object. This is used for persisting data
            across multiple threads (e.g., multiple conversations / users).
        interrupt_before: An optional list of step names to interrupt before execution.
        interrupt_after: An optional list of step names to interrupt after execution.
        debug: A flag to enable debug mode.

    Returns:
        A CompiledStateGraph object.
    """
    if len(steps) < 2:
        raise ValueError("Sequential executor requires at least two steps.")

    builder = StateGraph(state_schema)
    previous_name: Optional[str] = None
    for step in steps:
        if isinstance(step, tuple) and len(step) == 2:
            name, step = step
        else:
            name = _get_name(step)

        builder.add_node(name, step)
        if previous_name is None:
            builder.add_edge(START, name)
        else:
            builder.add_edge(previous_name, name)

        previous_name = name

    builder.add_edge(cast(str, previous_name), END)
    return builder.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )
