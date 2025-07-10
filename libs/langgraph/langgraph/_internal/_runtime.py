"""Internal utilities for the Runtime class."""

from dataclasses import replace
from typing import Any, cast

from typing_extensions import TypedDict, Unpack

from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter


class RuntimePatch(TypedDict, total=False):
    """Patch structure for the Runtime class."""

    context: Any
    store: BaseStore | None
    stream_writer: StreamWriter
    previous: Any


def patch_runtime(runtime: Runtime, **overrides: Unpack[RuntimePatch]) -> Runtime:
    """Patch the runtime with the given overrides, returning a new instance."""
    return replace(runtime, **overrides)


def patch_runtime_non_null(
    runtime: Runtime, **overrides: Unpack[RuntimePatch]
) -> Runtime:
    """Patch the runtime with the given overrides, returning a new instance.

    Only patch fields with overrides that are not None.
    """
    return replace(
        runtime,
        **cast(dict[str, Any], {k: v for k, v in overrides.items() if v is not None}),
    )
