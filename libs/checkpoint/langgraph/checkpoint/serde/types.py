from typing import Any, Protocol, TypeVar, runtime_checkable

ERROR = "__error__"
SCHEDULED = "__scheduled__"
INTERRUPT = "__interrupt__"
RESUME = "__resume__"
TASKS = "__pregel_tasks"

Value = TypeVar("Value", covariant=True)
Update = TypeVar("Update", contravariant=True)
C = TypeVar("C")


@runtime_checkable
class SendProtocol(Protocol):
    # Mirrors langgraph.constants.Send
    node: str
    arg: Any

    def __hash__(self) -> int: ...

    def __repr__(self) -> str: ...

    def __eq__(self, value: object) -> bool: ...
