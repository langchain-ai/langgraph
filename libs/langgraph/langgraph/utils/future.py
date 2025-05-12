import asyncio
import concurrent.futures
import contextvars
import inspect
import sys
import types
from collections.abc import Awaitable, Coroutine, Generator
from typing import Optional, TypeVar, Union, cast

T = TypeVar("T")
AnyFuture = Union[asyncio.Future, concurrent.futures.Future]

CONTEXT_NOT_SUPPORTED = sys.version_info < (3, 11)
EAGER_NOT_SUPPORTED = sys.version_info < (3, 12)


def _get_loop(fut: asyncio.Future) -> asyncio.AbstractEventLoop:
    # Tries to call Future.get_loop() if it's available.
    # Otherwise fallbacks to using the old '_loop' property.
    try:
        get_loop = fut.get_loop
    except AttributeError:
        pass
    else:
        return get_loop()
    return fut._loop


def _convert_future_exc(exc: BaseException) -> BaseException:
    exc_class = type(exc)
    if exc_class is concurrent.futures.CancelledError:
        return asyncio.CancelledError(*exc.args)
    elif exc_class is concurrent.futures.TimeoutError:
        return asyncio.TimeoutError(*exc.args)
    elif exc_class is concurrent.futures.InvalidStateError:
        return asyncio.InvalidStateError(*exc.args)
    else:
        return exc


def _set_concurrent_future_state(
    concurrent: concurrent.futures.Future,
    source: AnyFuture,
) -> None:
    """Copy state from a future to a concurrent.futures.Future."""
    assert source.done()
    if source.cancelled():
        concurrent.cancel()
    if not concurrent.set_running_or_notify_cancel():
        return
    exception = source.exception()
    if exception is not None:
        concurrent.set_exception(_convert_future_exc(exception))
    else:
        result = source.result()
        concurrent.set_result(result)


def _copy_future_state(source: AnyFuture, dest: asyncio.Future) -> None:
    """Internal helper to copy state from another Future.

    The other Future may be a concurrent.futures.Future.
    """
    if dest.done():
        return
    assert source.done()
    if dest.cancelled():
        return
    if source.cancelled():
        dest.cancel()
    else:
        exception = source.exception()
        if exception is not None:
            dest.set_exception(_convert_future_exc(exception))
        else:
            result = source.result()
            dest.set_result(result)


def _chain_future(source: AnyFuture, destination: AnyFuture) -> None:
    """Chain two futures so that when one completes, so does the other.

    The result (or exception) of source will be copied to destination.
    If destination is cancelled, source gets cancelled too.
    Compatible with both asyncio.Future and concurrent.futures.Future.
    """
    if not asyncio.isfuture(source) and not isinstance(
        source, concurrent.futures.Future
    ):
        raise TypeError("A future is required for source argument")
    if not asyncio.isfuture(destination) and not isinstance(
        destination, concurrent.futures.Future
    ):
        raise TypeError("A future is required for destination argument")
    source_loop = _get_loop(source) if asyncio.isfuture(source) else None
    dest_loop = _get_loop(destination) if asyncio.isfuture(destination) else None

    def _set_state(future: AnyFuture, other: AnyFuture) -> None:
        if asyncio.isfuture(future):
            _copy_future_state(other, future)
        else:
            _set_concurrent_future_state(future, other)

    def _call_check_cancel(destination: AnyFuture) -> None:
        if destination.cancelled():
            if source_loop is None or source_loop is dest_loop:
                source.cancel()
            else:
                source_loop.call_soon_threadsafe(source.cancel)

    def _call_set_state(source: AnyFuture) -> None:
        if destination.cancelled() and dest_loop is not None and dest_loop.is_closed():
            return
        if dest_loop is None or dest_loop is source_loop:
            _set_state(destination, source)
        else:
            if dest_loop.is_closed():
                return
            dest_loop.call_soon_threadsafe(_set_state, destination, source)

    destination.add_done_callback(_call_check_cancel)
    source.add_done_callback(_call_set_state)


def chain_future(source: AnyFuture, destination: AnyFuture) -> AnyFuture:
    # adapted from asyncio.run_coroutine_threadsafe
    try:
        _chain_future(source, destination)
        return destination
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        if isinstance(destination, concurrent.futures.Future):
            if destination.set_running_or_notify_cancel():
                destination.set_exception(exc)
        else:
            destination.set_exception(exc)
        raise


def _ensure_future(
    coro_or_future: Union[Coroutine[None, None, T], Awaitable[T]],
    *,
    loop: asyncio.AbstractEventLoop,
    name: Optional[str] = None,
    context: Optional[contextvars.Context] = None,
    lazy: bool = True,
) -> asyncio.Task[T]:
    called_wrap_awaitable = False
    if not asyncio.iscoroutine(coro_or_future):
        if inspect.isawaitable(coro_or_future):
            coro_or_future = cast(
                Coroutine[None, None, T], _wrap_awaitable(coro_or_future)
            )
            called_wrap_awaitable = True
        else:
            raise TypeError(
                "An asyncio.Future, a coroutine or an awaitable is required."
                f" Got {type(coro_or_future).__name__} instead."
            )

    try:
        if CONTEXT_NOT_SUPPORTED:
            return loop.create_task(coro_or_future, name=name)
        elif EAGER_NOT_SUPPORTED or lazy:
            return loop.create_task(coro_or_future, name=name, context=context)
        else:
            return asyncio.eager_task_factory(
                loop, coro_or_future, name=name, context=context
            )
    except RuntimeError:
        if not called_wrap_awaitable:
            coro_or_future.close()
        raise


@types.coroutine
def _wrap_awaitable(awaitable: Awaitable[T]) -> Generator[None, None, T]:
    """Helper for asyncio.ensure_future().

    Wraps awaitable (an object with __await__) into a coroutine
    that will later be wrapped in a Task by ensure_future().
    """
    return (yield from awaitable.__await__())


def run_coroutine_threadsafe(
    coro: Coroutine[None, None, T],
    loop: asyncio.AbstractEventLoop,
    *,
    lazy: bool,
    name: Optional[str] = None,
    context: Optional[contextvars.Context] = None,
) -> asyncio.Future[T]:
    """Submit a coroutine object to a given event loop.

    Return a asyncio.Future to access the result.
    """

    if asyncio._get_running_loop() is loop:
        return _ensure_future(coro, loop=loop, name=name, context=context, lazy=lazy)
    else:
        future: asyncio.Future[T] = asyncio.Future(loop=loop)

        def callback() -> None:
            try:
                chain_future(
                    _ensure_future(coro, loop=loop, name=name, context=context),
                    future,
                )
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException as exc:
                future.set_exception(exc)
                raise

        loop.call_soon_threadsafe(callback, context=context)
        return future
