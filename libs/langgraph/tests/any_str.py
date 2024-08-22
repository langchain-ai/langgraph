from typing import Any, Sequence


class AnyStr(str):
    def __init__(self) -> None:
        super().__init__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str)

    def __hash__(self) -> int:
        return hash(str(self))


class AnyVersion:
    def __init__(self) -> None:
        super().__init__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (str, int, float))

    def __hash__(self) -> int:
        return hash(str(self))


class ExceptionLike:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Exception)
            and self.exc.__class__ == value.__class__
            and str(self.exc) == str(value)
        )

    def __hash__(self) -> int:
        return hash((self.exc.__class__, str(self.exc)))

    def __repr__(self) -> str:
        return str(self.exc)


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
