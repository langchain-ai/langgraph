from typing import Any, Protocol, Sequence, Generator, AsyncGenerator, Optional, TypeVar, runtime_checkable
from typing_extensions import Self

from langchain_core.runnables import RunnableConfig


Value = TypeVar("Value")
Update = TypeVar("Update")
C = TypeVar("C")


class ChannelProtocol(Protocol[Value, Update, C]):
    @property
    def ValueType(self) -> Any:
        ...

    @property
    def UpdateType(self) -> Any:
        ...

    def checkpoint(self) -> Optional[C]:
        ...

    def from_checkpoint(
        self, checkpoint: Optional[C], config: RunnableConfig
    ) -> Generator[Self, None, None]:
        ...

    async def afrom_checkpoint(
        self, checkpoint: Optional[C], config: RunnableConfig
    ) -> AsyncGenerator[Self, None]:
        ...

    def update(self, values: Sequence[Update]) -> bool:
        ...

    def get(self) -> Value:
        ...

    def consume(self) -> bool:
        ...


@runtime_checkable
class SendProtocol(Protocol):
    """A message or packet to send to a specific node in the graph."""

    node: str
    arg: Any

    def __hash__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, value: object) -> bool:
        ...