class AnyInt(int):
    def __init__(self) -> None:
        super().__init__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, int)
