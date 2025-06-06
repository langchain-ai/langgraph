from __future__ import annotations

from typing import Union

from typing_extensions import TypeVar

from langgraph._typing import StateLike

StateT = TypeVar("StateT", bound=StateLike)
"""Type variable used to represent the state in a graph."""

StateT_co = TypeVar("StateT_co", bound=StateLike, covariant=True)

StateT_contra = TypeVar("StateT_contra", bound=StateLike, contravariant=True)

InputT = TypeVar("InputT", bound=StateLike, default=StateT)
"""Type variable used to represent the input to a state graph.

Defaults to `StateT`.
"""

ResolvedInputT = TypeVar("ResolvedInputT", bound=StateLike)
"""Type variable used to represent the resolved input to a state graph.

No default.
"""


OutputT = TypeVar("OutputT", bound=Union[StateLike, None], default=StateT)
"""Type variable used to represent the output of a state graph."""
