import functools

from langgraph.utils import _isgenerator, _iscoroutinefunction


def test_is_async() -> None:
    async def func() -> None:
        pass

    assert _iscoroutinefunction(func)
    wrapped_func = functools.wraps(func)(func)
    assert _iscoroutinefunction(wrapped_func)

    def sync_func() -> None:
        pass

    assert not _iscoroutinefunction(sync_func)
    wrapped_sync_func = functools.wraps(sync_func)(sync_func)
    assert not _iscoroutinefunction(wrapped_sync_func)

    class AsyncFuncCallable:
        async def __call__(self) -> None:
            pass

    runnable = AsyncFuncCallable()
    assert _iscoroutinefunction(runnable)
    wrapped_runnable = functools.wraps(runnable)(runnable)
    assert _iscoroutinefunction(wrapped_runnable)

    class SyncFuncCallable:
        def __call__(self) -> None:
            pass

    sync_runnable = SyncFuncCallable()
    assert not _iscoroutinefunction(sync_runnable)
    wrapped_sync_runnable = functools.wraps(sync_runnable)(sync_runnable)
    assert not _iscoroutinefunction(wrapped_sync_runnable)


def test_is_generator() -> None:
    async def gen():
        yield

    assert _isgenerator(gen)

    wrapped_gen = functools.wraps(gen)(gen)
    assert _isgenerator(wrapped_gen)

    def sync_gen():
        yield

    assert _isgenerator(sync_gen)
    wrapped_sync_gen = functools.wraps(sync_gen)(sync_gen)
    assert _isgenerator(wrapped_sync_gen)

    class AsyncGenCallable:
        async def __call__(self):
            yield

    runnable = AsyncGenCallable()
    assert _isgenerator(runnable)
    wrapped_runnable = functools.wraps(runnable)(runnable)
    assert _isgenerator(wrapped_runnable)

    class SyncGenCallable:
        def __call__(self):
            yield

    sync_runnable = SyncGenCallable()
    assert _isgenerator(sync_runnable)
    wrapped_sync_runnable = functools.wraps(sync_runnable)(sync_runnable)
    assert _isgenerator(wrapped_sync_runnable)
