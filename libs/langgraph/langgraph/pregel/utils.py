from typing import Optional

from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.runnables.utils import get_function_nonlocals

from langgraph.checkpoint.base import ChannelVersions
from langgraph.pregel.protocol import PregelProtocol
from langgraph.utils.runnable import Runnable, RunnableCallable, RunnableSeq


def get_new_channel_versions(
    previous_versions: ChannelVersions, current_versions: ChannelVersions
) -> ChannelVersions:
    """Get subset of current_versions that are newer than previous_versions."""
    if previous_versions:
        version_type = type(next(iter(current_versions.values()), None))
        null_version = version_type()  # type: ignore[misc]
        new_versions = {
            k: v
            for k, v in current_versions.items()
            if v > previous_versions.get(k, null_version)  # type: ignore[operator]
        }
    else:
        new_versions = current_versions

    return new_versions


def find_subgraph_pregel(candidate: Runnable) -> Optional[Runnable]:
    from langgraph.pregel import Pregel

    candidates: list[Runnable] = [candidate]

    for c in candidates:
        if (
            isinstance(c, PregelProtocol)
            # subgraphs that disabled checkpointing are not considered
            and (not isinstance(c, Pregel) or c.checkpointer is not False)
        ):
            return c
        elif isinstance(c, RunnableSequence) or isinstance(c, RunnableSeq):
            candidates.extend(c.steps)
        elif isinstance(c, RunnableLambda):
            candidates.extend(c.deps)
        elif isinstance(c, RunnableCallable):
            if c.func is not None:
                candidates.extend(
                    nl.__self__ if hasattr(nl, "__self__") else nl
                    for nl in get_function_nonlocals(c.func)
                )
            if c.afunc is not None:
                candidates.extend(
                    nl.__self__ if hasattr(nl, "__self__") else nl
                    for nl in get_function_nonlocals(c.afunc)
                )

    return None
