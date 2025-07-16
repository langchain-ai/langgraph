import sys
from typing import Any
from warnings import warn

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINTER,
    TASKS,
)
from langgraph.warnings import LangGraphDeprecatedSinceV10

__all__ = (
    "TAG_NOSTREAM",
    "TAG_HIDDEN",
    "START",
    "END",
    # retained for backwards compatibility (mostly langgraph-api), should be removed in v2 (or earlier)
    "CONF",
    "TASKS",
    "CONFIG_KEY_CHECKPOINTER",
)

# --- Public constants ---
TAG_NOSTREAM = sys.intern("nostream")
"""Tag to disable streaming for a chat model."""
TAG_HIDDEN = sys.intern("langsmith:hidden")
"""Tag to hide a node/edge from certain tracing/streaming environments."""
END = sys.intern("__end__")
"""The last (maybe virtual) node in graph-style Pregel."""
START = sys.intern("__start__")
"""The first (maybe virtual) node in graph-style Pregel."""


def __getattr__(name: str) -> Any:
    if name in ["Send", "Interrupt"]:
        warn(
            f"Importing {name} from langgraph.constants is deprecated. "
            f"Please use 'from langgraph.types import {name}' instead.",
            LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )

        from importlib import import_module

        module = import_module("langgraph.types")
        return getattr(module, name)

    try:
        from importlib import import_module

        private_constants = import_module("langgraph._internal._constants")
        attr = getattr(private_constants, name)
        warn(
            f"Importing {name} from langgraph.constants is deprecated. "
            f"This constant is now private and should not be used directly. "
            "Please let the LangGraph team know if you need this value.",
            LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        return attr
    except AttributeError:
        pass

    raise AttributeError(f"module has no attribute '{name}'")
