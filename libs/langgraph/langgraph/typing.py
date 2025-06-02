from __future__ import annotations

from typing import Any, TypeVar, Union

from pydantic import BaseModel
from typing_extensions import ParamSpec, TypeAlias

StateLike: TypeAlias = Union[dict[str, Any], BaseModel, object]
"""Type alias for state-like types.

It can either be a `TypedDict`, `dataclass`, or Pydantic `BaseModel`.
Note: we cannot use either `TypedDict` or `dataclass` directly due to limitations in type checking."""

StateT = TypeVar("StateT", bound=StateLike)
"""Type variable used to represent the state in a graph."""

StateT_co = TypeVar("StateT_co", bound=StateLike, covariant=True)

StateT_contra = TypeVar("StateT_contra", bound=StateLike, contravariant=True)

InputT = TypeVar("InputT", bound=StateLike)
"""Type variable used to represent the input to a graph."""

OutputT = TypeVar("OutputT", bound=StateLike)
"""Type variable used to represent the output of a graph."""

P = ParamSpec("P")
