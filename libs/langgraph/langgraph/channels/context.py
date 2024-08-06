from contextlib import asynccontextmanager, contextmanager
from inspect import signature
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    ContextManager,
    Generator,
    Generic,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError, InvalidUpdateError


class Context(Generic[Value], BaseChannel[Value, None, None]):
    """Exposes the value of a context manager, for the duration of an invocation.
    Context manager is entered before the first step, and exited after the last step.
    Optionally, provide an equivalent async context manager, which will be used
    instead for async invocations.

    ```python
    import httpx

    client = Channels.Context(httpx.Client, httpx.AsyncClient)
    ```
    """

    value: Value

    def __init__(
        self,
        ctx: Union[
            None, Type[ContextManager[Value]], Type[AsyncContextManager[Value]]
        ] = None,
        actx: Optional[Type[AsyncContextManager[Value]]] = None,
    ) -> None:
        if ctx is None and actx is None:
            raise ValueError("Must provide either sync or async context manager.")
        self.ctx = ctx
        self.actx = actx

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Context)
            and value.ctx == self.ctx
            and value.actx == self.actx
        )

    @property
    def ValueType(self) -> Any:
        """The type of the value stored in the channel."""
        return None

    @property
    def UpdateType(self) -> Type[None]:
        """The type of the update received by the channel."""
        return None

    def checkpoint(self) -> None:
        raise EmptyChannelError()

    @contextmanager
    def from_checkpoint(
        self, checkpoint: None, config: RunnableConfig
    ) -> Generator[Self, None, None]:
        if self.ctx is None:
            raise ValueError("Cannot enter sync context manager.")

        empty = self.__class__(ctx=self.ctx, actx=self.actx)
        ctx = (
            self.ctx(config)
            if signature(self.ctx).parameters.get("config")
            else self.ctx()
        )
        with ctx as value:
            empty.value = value
            yield empty

    @asynccontextmanager
    async def afrom_checkpoint(
        self, checkpoint: None, config: RunnableConfig
    ) -> AsyncGenerator[Self, None]:
        empty = self.__class__(ctx=self.ctx, actx=self.actx)
        if self.actx is not None:
            ctx = (
                self.actx(config)
                if signature(self.actx).parameters.get("config")
                else self.actx()
            )
        else:
            ctx = (
                self.ctx(config)
                if signature(self.ctx).parameters.get("config")
                else self.ctx()
            )
        if hasattr(ctx, "__aenter__"):
            async with ctx as value:
                empty.value = value
                yield empty
        else:
            with ctx as value:
                empty.value = value
                yield empty

    def update(self, values: Sequence[None]) -> bool:
        if values:
            raise InvalidUpdateError("Context channel does not accept writes.")
        return False

    def get(self) -> Value:
        try:
            return self.value
        except AttributeError:
            raise EmptyChannelError()
