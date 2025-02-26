import asyncio
import contextvars
from _typeshed import Incomplete
from typing import Coroutine, TypeVar

T = TypeVar('T')
AnyFuture: Incomplete
CONTEXT_NOT_SUPPORTED: Incomplete

def chain_future(source: AnyFuture, destination: AnyFuture) -> AnyFuture: ...
def run_coroutine_threadsafe(coro: Coroutine[None, None, T], loop: asyncio.AbstractEventLoop, name: str | None = None, context: contextvars.Context | None = None) -> asyncio.Future[T]: ...
