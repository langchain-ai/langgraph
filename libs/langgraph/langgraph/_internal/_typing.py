"""Private typing utilities for LangGraph."""

from __future__ import annotations

from dataclasses import Field
from typing import Any, ClassVar, Protocol, TypeAlias

from pydantic import BaseModel
from typing_extensions import TypedDict


class TypedDictLikeV1(Protocol):
    """Protocol to represent types that behave like TypedDicts

    Version 1: using `ClassVar` for keys."""

    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]


class TypedDictLikeV2(Protocol):
    """Protocol to represent types that behave like TypedDicts

    Version 2: not using `ClassVar` for keys."""

    __required_keys__: frozenset[str]
    __optional_keys__: frozenset[str]


class DataclassLike(Protocol):
    """Protocol to represent types that behave like dataclasses.

    Inspired by the private _DataclassT from dataclasses that uses a similar protocol as a bound."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


StateLike: TypeAlias = TypedDictLikeV1 | TypedDictLikeV2 | DataclassLike | BaseModel
"""Type alias for state-like types.

It can either be a `TypedDict`, `dataclass`, or Pydantic `BaseModel`.
Note: we cannot use either `TypedDict` or `dataclass` directly due to limitations in type checking.
"""

MISSING = object()
"""Unset sentinel value."""


class DeprecatedKwargs(TypedDict):
    """TypedDict to use for extra keyword arguments, enabling type checking warnings for deprecated arguments."""


EMPTY_SEQ: tuple[str, ...] = tuple()
"""An empty sequence of strings."""
