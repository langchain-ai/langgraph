"""LangGraph specific warnings."""

from __future__ import annotations

__all__ = (
    "LangGraphDeprecationWarning",
    "LangGraphDeprecatedSinceV05",
    "LangGraphDeprecatedSinceV10",
)


class LangGraphDeprecationWarning(DeprecationWarning):
    """A LangGraph specific deprecation warning.

    Attributes:
        message: Description of the warning.
        since: LangGraph version in which the deprecation was introduced.
        expected_removal: LangGraph version in what the corresponding functionality expected to be removed.

    Inspired by the Pydantic `PydanticDeprecationWarning` class, which sets a great standard
    for deprecation warnings with clear versioning information.
    """

    message: str
    since: tuple[int, int]
    expected_removal: tuple[int, int]

    def __init__(
        self,
        message: str,
        *args: object,
        since: tuple[int, int],
        expected_removal: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(message, *args)
        self.message = message.rstrip(".")
        self.since = since
        self.expected_removal = (
            expected_removal if expected_removal is not None else (since[0] + 1, 0)
        )

    def __str__(self) -> str:
        message = (
            f"{self.message}. Deprecated in LangGraph V{self.since[0]}.{self.since[1]}"
            f" to be removed in V{self.expected_removal[0]}.{self.expected_removal[1]}."
        )
        return message


class LangGraphDeprecatedSinceV05(LangGraphDeprecationWarning):
    """A specific `LangGraphDeprecationWarning` subclass defining functionality deprecated since LangGraph v0.5.0"""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args, since=(0, 5), expected_removal=(2, 0))


class LangGraphDeprecatedSinceV10(LangGraphDeprecationWarning):
    """A specific `LangGraphDeprecationWarning` subclass defining functionality deprecated since LangGraph v1.0.0"""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args, since=(1, 0), expected_removal=(2, 0))
