from typing import Any, Optional, Sequence, Type, Union, cast

from langchain_core.runnables.base import Runnable, RunnableLike

from langgraph.graph.state import END, START, StateGraph


def _get_step_name(step: RunnableLike) -> str:
    if isinstance(step, Runnable):
        return step.get_name()
    elif callable(step):
        return getattr(step, "__name__", step.__class__.__name__)
    else:
        raise TypeError(f"Unsupported step type: {step}")


def create_pipeline(
    steps: Sequence[Union[RunnableLike, tuple[str, RunnableLike]]],
    *,
    state_schema: Type[Any],
    input_schema: Optional[Type[Any]] = None,
    output_schema: Optional[Type[Any]] = None,
) -> StateGraph:
    """Create a pipeline graph that runs a series of provided steps in order.

    Args:
        steps: A sequence of RunnableLike objects (e.g. a LangChain Runnable or a callable) or (name, RunnableLike) tuples.
            If no names are provided, the name will be inferred from the step object (e.g. a runnable or a callable name).
            Each step will be executed in the order provided.
        state_schema: The state schema for the graph.
        input_schema: The input schema for the graph.
        output_schema: The output schema for the graph. Will only be used when calling `graph.invoke()`.

    Returns:
        A StateGraph object.
    """
    if len(steps) < 1:
        raise ValueError("Pipeline requires at least one step.")

    node_names = set()
    builder = StateGraph(state_schema, input=input_schema, output=output_schema)
    previous_name: Optional[str] = None
    for step in steps:
        if isinstance(step, tuple) and len(step) == 2:
            name, step = step
        else:
            name = _get_step_name(step)

        if name in node_names:
            raise ValueError(
                f"Step names must be unique: step with the name '{name}' already exists. "
                "If you need to use two different runnables/callables with the same name (for example, using `lambda`), please provide them as tuples (name, runnable/callable)."
            )

        node_names.add(name)
        builder.add_node(name, step)
        if previous_name is None:
            builder.add_edge(START, name)
        else:
            builder.add_edge(previous_name, name)

        previous_name = name

    builder.add_edge(cast(str, previous_name), END)
    return builder
