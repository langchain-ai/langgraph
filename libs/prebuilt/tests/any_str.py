import re
from typing import Any, Sequence, Union

from typing_extensions import Self


class FloatBetween(float):
    def __new__(cls, min_value: float, max_value: float) -> Self:
        return super().__new__(cls, min_value)

    def __init__(self, min_value: float, max_value: float) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, float)
            and other >= self.min_value
            and other <= self.max_value
        )

    def __hash__(self) -> int:
        return hash((float(self), self.min_value, self.max_value))


class AnyStr(str):
    def __init__(self, prefix: Union[str, re.Pattern] = "") -> None:
        super().__init__()
        self.prefix = prefix

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and (
            other.startswith(self.prefix)
            if isinstance(self.prefix, str)
            else self.prefix.match(other)
        )

    def __hash__(self) -> int:
        return hash((str(self), self.prefix))


class AnyDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, dict) or len(self) != len(other):
            return False
        for k, v in self.items():
            if kk := next((kk for kk in other if kk == k), None):
                if v == other[kk]:
                    continue
                else:
                    return False
        else:
            return True


class AnyVersion:
    def __init__(self) -> None:
        super().__init__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (str, int, float))

    def __hash__(self) -> int:
        return hash(str(self))


class UnsortedSequence:
    def __init__(self, *values: Any) -> None:
        self.seq = values

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Sequence)
            and len(self.seq) == len(value)
            and all(a in value for a in self.seq)
        )

    def __hash__(self) -> int:
        return hash(frozenset(self.seq))

    def __repr__(self) -> str:
        return repr(self.seq)
