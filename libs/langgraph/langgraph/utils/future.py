import asyncio
import concurrent.futures
from typing import Union

AnyFuture = Union[asyncio.Future, concurrent.futures.Future]


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
    assert source.done()
    if dest.cancelled():
        return
    assert not dest.done()
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


def chain_future(source: AnyFuture, destination: concurrent.futures.Future) -> None:
    # adapted from asyncio.run_coroutine_threadsafe
    try:
        _chain_future(source, destination)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        if destination.set_running_or_notify_cancel():
            destination.set_exception(exc)
        raise
