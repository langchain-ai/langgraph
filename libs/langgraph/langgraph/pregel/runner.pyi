import asyncio
import concurrent.futures
import threading
from _typeshed import Incomplete
from langchain_core.callbacks import Callbacks as Callbacks
from langgraph.pregel.algo import Call
from langgraph.pregel.executor import Submit as Submit
from langgraph.types import PregelExecutableTask, PregelScratchpad as PregelScratchpad, RetryPolicy as RetryPolicy
from typing import Any, AsyncIterator, Callable, Generic, Iterable, Iterator, Sequence, TypeVar

F = TypeVar('F', concurrent.futures.Future, asyncio.Future)
E = TypeVar('E', threading.Event, asyncio.Event)

class FuturesDict(dict[F, PregelExecutableTask | None], Generic[F, E]):
    event: E
    callback: Callable[[PregelExecutableTask, BaseException | None], None]
    counter: int
    done: set[F]
    lock: threading.Lock
    def __init__(self, event: E, callback: Callable[[PregelExecutableTask, BaseException | None], None], future_type: type[F]) -> None: ...
    def __setitem__(self, key: F, value: PregelExecutableTask | None) -> None: ...
    def on_done(self, task: PregelExecutableTask, fut: F) -> None: ...

class PregelRunner:
    submit: Incomplete
    put_writes: Incomplete
    use_astream: Incomplete
    node_finished: Incomplete
    schedule_task: Incomplete
    def __init__(self, *, submit: Submit, put_writes: Callable[[str, Sequence[tuple[str, Any]]], None], schedule_task: Callable[[PregelExecutableTask, int, Call | None], PregelExecutableTask | None], use_astream: bool = False, node_finished: Callable[[str], None] | None = None) -> None: ...
    def tick(self, tasks: Iterable[PregelExecutableTask], *, reraise: bool = True, timeout: float | None = None, retry_policy: RetryPolicy | None = None, get_waiter: Callable[[], concurrent.futures.Future[None]] | None = None) -> Iterator[None]: ...
    async def atick(self, tasks: Iterable[PregelExecutableTask], *, reraise: bool = True, timeout: float | None = None, retry_policy: RetryPolicy | None = None, get_waiter: Callable[[], asyncio.Future[None]] | None = None) -> AsyncIterator[None]: ...
    def commit(self, task: PregelExecutableTask, exception: BaseException | None) -> None: ...
