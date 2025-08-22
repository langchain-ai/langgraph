from __future__ import annotations

from typing import Awaitable, Callable, TypeVar, Union

from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

SyncOrAsync = Callable[P, Union[R, Awaitable[R]]]
