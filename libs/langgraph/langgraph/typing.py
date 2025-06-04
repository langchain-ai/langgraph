from __future__ import annotations

from dataclasses import Field
from typing import (
    Any,
    ClassVar,
    Protocol,
    Union,
)

from pydantic import BaseModel
from typing_extensions import TypeAlias, TypedDict, TypeVar


class _TypedDictLikeV1(Protocol):
    """Protocol to represent types that behave like TypedDicts

    Version 1: using `ClassVar` for keys."""

    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]


class _TypedDictLikeV2(Protocol):
    """Protocol to represent types that behave like TypedDicts

    Version 2: not using `ClassVar` for keys."""

    __required_keys__: frozenset[str]
    __optional_keys__: frozenset[str]


class _DataclassLike(Protocol):
    """Protocol to represent types that behave like dataclasses.

    Inspired by the private _DataclassT from dataclasses that uses a similar protocol as a bound."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


StateLike: TypeAlias = Union[
    _TypedDictLikeV1, _TypedDictLikeV2, _DataclassLike, BaseModel
]
"""Type alias for state-like types.

It can either be a `TypedDict`, `dataclass`, or Pydantic `BaseModel`.
Note: we cannot use either `TypedDict` or `dataclass` directly due to limitations in type checking."""


class Unset:
    """Sentinel representing an unset value."""


UNSET = Unset()

StateT = TypeVar("StateT", bound=StateLike)
"""Type variable used to represent the state in a graph."""

StateT_co = TypeVar("StateT_co", bound=StateLike, covariant=True)

StateT_contra = TypeVar("StateT_contra", bound=StateLike, contravariant=True)

InputT = TypeVar("InputT", bound=Union[StateLike, Unset], default=Unset)
"""Type variable used to represent the input to a graph.

In practice, InputT might be represented by either the `input_type` or `state_type` of a graph.
If `input_type` is not specified, it defaults to `StateType`."""

OutputT = TypeVar("OutputT", bound=Union[StateLike, Unset], default=Unset)
"""Type variable used to represent the output of a graph."""


class DeprecatedKwargs(TypedDict):
    """TypedDict to use for extra keyword arguments, enabling type checking warnings for deprecated arguments."""
