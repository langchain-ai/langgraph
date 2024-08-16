class AnyStr(str):
    def __init__(self) -> None:
        super().__init__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str)

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
