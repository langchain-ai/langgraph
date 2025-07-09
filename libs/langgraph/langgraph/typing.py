from __future__ import annotations

from typing import Union

from typing_extensions import TypeVar

from langgraph._internal._typing import StateLike

__all__ = (
    "StateT",
    "StateT_co",
    "StateT_contra",
    "InputT",
    "OutputT",
)

StateT = TypeVar("StateT", bound=StateLike)
"""Type variable used to represent the state in a graph."""

StateT_co = TypeVar("StateT_co", bound=StateLike, covariant=True)

StateT_contra = TypeVar("StateT_contra", bound=StateLike, contravariant=True)

InputT = TypeVar("InputT", bound=StateLike, default=StateT)
"""Type variable used to represent the input to a state graph.

Defaults to `StateT`.
"""

OutputT = TypeVar("OutputT", bound=Union[StateLike, None], default=StateT)
"""Type variable used to represent the output of a state graph."""

NodeInputT = TypeVar("NodeInputT", bound=StateLike)

NodeInputT_contra = TypeVar("NodeInputT_contra", bound=StateLike, contravariant=True)
