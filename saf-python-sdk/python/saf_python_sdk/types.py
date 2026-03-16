from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

N = TypeVar("N")


@dataclass
class Send(Generic[N]):
    node: N
    arg: Any = None


@dataclass
class Command:
    update: Any = None
    goto: Any = field(default=None)

