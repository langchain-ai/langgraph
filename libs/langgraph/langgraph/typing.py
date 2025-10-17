from __future__ import annotations

from typing_extensions import TypeVar

from langgraph._internal._typing import StateLike

__all__ = (
    "StateT",
    "StateT_co",
    "StateT_contra",
    "InputT",
    "OutputT",
    "ContextT",
)

StateT = TypeVar("StateT", bound=StateLike)
"""Type variable used to represent the state in a graph."""

StateT_co = TypeVar("StateT_co", bound=StateLike, covariant=True)

StateT_contra = TypeVar("StateT_contra", bound=StateLike, contravariant=True)

ContextT = TypeVar("ContextT", bound=StateLike | None, default=None)
"""Type variable used to represent graph run scoped context.

Defaults to `None`.
"""

ContextT_contra = TypeVar(
    "ContextT_contra", bound=StateLike | None, contravariant=True, default=None
)

InputT = TypeVar("InputT", bound=StateLike, default=StateT)
"""Type variable used to represent the input to a state graph.

Defaults to `StateT`.
"""

OutputT = TypeVar("OutputT", bound=StateLike, default=StateT)
"""Type variable used to represent the output of a state graph.

Defaults to `StateT`.
"""

NodeInputT = TypeVar("NodeInputT", bound=StateLike)
"""Type variable used to represent the input to a node."""

NodeInputT_contra = TypeVar("NodeInputT_contra", bound=StateLike, contravariant=True)
