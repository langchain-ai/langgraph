import concurrent.futures
from langchain_core.runnables import Runnable as Runnable
from langgraph.types import RetryPolicy as RetryPolicy
from langgraph.utils.runnable import RunnableSeq
from typing import Any, Callable, Generator, Generic, TypeVar
from typing_extensions import ParamSpec

def get_runnable_for_entrypoint(func: Callable[..., Any]) -> RunnableSeq: ...
def get_runnable_for_task(func: Callable[..., Any]) -> RunnableSeq: ...

CACHE: dict[tuple[Callable[..., Any], bool], Runnable]
P = ParamSpec('P')
P1 = TypeVar('P1')
T = TypeVar('T')

class SyncAsyncFuture(concurrent.futures.Future[T], Generic[T]):
    def __await__(self) -> Generator[T, None, T]: ...

def call(func: Callable[P, T], *args: Any, retry: RetryPolicy | None = None, **kwargs: Any) -> SyncAsyncFuture[T]: ...
