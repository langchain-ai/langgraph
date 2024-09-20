import asyncio
import enum
import inspect
import sys
from contextlib import AsyncExitStack
from contextvars import copy_context
from functools import partial, wraps
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    Optional,
    Sequence,
    Union,
    cast,
)

from langchain_core.runnables.base import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableLike,
    RunnableParallel,
    RunnableSequence,
)
from langchain_core.runnables.config import (
    run_in_executor,
    var_child_runnable_config,
)
from langchain_core.runnables.utils import Input, accepts_config
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from typing_extensions import TypeGuard

from langgraph.utils.config import (
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
)

try:
    from langchain_core.runnables.config import _set_config_context
except ImportError:
    # For forwards compatibility
    def _set_config_context(context: RunnableConfig) -> None:  # type: ignore
        """Set the context for the current thread."""
        var_child_runnable_config.set(context)


# Before Python 3.11 native StrEnum is not available
class StrEnum(str, enum.Enum):
    """A string enum."""


ASYNCIO_ACCEPTS_CONTEXT = sys.version_info >= (3, 11)


class RunnableCallable(Runnable):
    """A much simpler version of RunnableLambda that requires sync and async functions."""

    def __init__(
        self,
        func: Optional[Callable[..., Union[Any, Runnable]]],
        afunc: Optional[Callable[..., Awaitable[Union[Any, Runnable]]]] = None,
        *,
        name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        trace: bool = True,
        recurse: bool = True,
        **kwargs: Any,
    ) -> None:
        self.name = name
        if self.name is None:
            if func:
                try:
                    if func.__name__ != "<lambda>":
                        self.name = func.__name__
                except AttributeError:
                    pass
            elif afunc:
                try:
                    self.name = afunc.__name__
                except AttributeError:
                    pass
        self.func = func
        if func is not None:
            self.func_accepts_config = accepts_config(func)
        self.afunc = afunc
        if afunc is not None:
            self.afunc_accepts_config = accepts_config(afunc)
        self.tags = tags
        self.kwargs = kwargs
        self.trace = trace
        self.recurse = recurse

    def __repr__(self) -> str:
        repr_args = {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"name", "func", "afunc", "config", "kwargs", "trace"}
        }
        return f"{self.get_name()}({', '.join(f'{k}={v!r}' for k, v in repr_args.items())})"

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        if self.func is None:
            raise TypeError(
                f'No synchronous function provided to "{self.name}".'
                "\nEither initialize with a synchronous function or invoke"
                " via the async API (ainvoke, astream, etc.)"
            )
        kwargs = {**self.kwargs, **kwargs}
        if self.func_accepts_config:
            kwargs["config"] = config
        if config is None:
            config = ensure_config()
        context = copy_context()
        if self.trace:
            callback_manager = get_callback_manager_for_config(config, self.tags)
            run_manager = callback_manager.on_chain_start(
                None,
                input,
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            try:
                child_config = patch_config(config, callbacks=run_manager.get_child())
                context = copy_context()
                context.run(_set_config_context, child_config)
                ret = context.run(self.func, input, **kwargs)
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise
            else:
                run_manager.on_chain_end(ret)
        else:
            context.run(_set_config_context, config)
            ret = context.run(self.func, input, **kwargs)
        if isinstance(ret, Runnable) and self.recurse:
            return ret.invoke(input, config)
        return ret

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        if not self.afunc:
            return self.invoke(input, config)
        kwargs = {**self.kwargs, **kwargs}
        if self.afunc_accepts_config:
            kwargs["config"] = config
        if config is None:
            config = ensure_config()
        context = copy_context()
        if self.trace:
            callback_manager = get_async_callback_manager_for_config(config, self.tags)
            run_manager = await callback_manager.on_chain_start(
                None,
                input,
                name=config.get("run_name") or self.name,
                run_id=config.pop("run_id", None),
            )
            try:
                child_config = patch_config(config, callbacks=run_manager.get_child())
                context.run(_set_config_context, child_config)
                coro = cast(Coroutine[None, None, Any], self.afunc(input, **kwargs))
                if ASYNCIO_ACCEPTS_CONTEXT:
                    ret = await asyncio.create_task(coro, context=context)
                else:
                    ret = await coro
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise
            else:
                await run_manager.on_chain_end(ret)
        else:
            context.run(_set_config_context, config)
            if ASYNCIO_ACCEPTS_CONTEXT:
                coro = cast(Coroutine[None, None, Any], self.afunc(input, **kwargs))
                ret = await asyncio.create_task(coro, context=context)
            else:
                ret = await self.afunc(input, **kwargs)
        if isinstance(ret, Runnable) and self.recurse:
            return await ret.ainvoke(input, config)
        return ret


def is_async_callable(
    func: Any,
) -> TypeGuard[Callable[..., Awaitable]]:
    """Check if a function is async."""
    return (
        asyncio.iscoroutinefunction(func)
        or hasattr(func, "__call__")
        and asyncio.iscoroutinefunction(func.__call__)
    )


def is_async_generator(
    func: Any,
) -> TypeGuard[Callable[..., AsyncIterator]]:
    """Check if a function is an async generator."""
    return (
        inspect.isasyncgenfunction(func)
        or hasattr(func, "__call__")
        and inspect.isasyncgenfunction(func.__call__)
    )


def coerce_to_runnable(
    thing: RunnableLike, *, name: Optional[str], trace: bool
) -> Runnable:
    """Coerce a runnable-like object into a Runnable.

    Args:
        thing: A runnable-like object.

    Returns:
        A Runnable.
    """
    if isinstance(thing, Runnable):
        return thing
    elif is_async_generator(thing) or inspect.isgeneratorfunction(thing):
        return RunnableLambda(thing, name=name)
    elif callable(thing):
        if is_async_callable(thing):
            return RunnableCallable(None, thing, name=name, trace=trace)
        else:
            return RunnableCallable(
                thing,
                wraps(thing)(partial(run_in_executor, None, thing)),  # type: ignore[arg-type]
                name=name,
                trace=trace,
            )
    elif isinstance(thing, dict):
        return RunnableParallel(thing)
    else:
        raise TypeError(
            f"Expected a Runnable, callable or dict."
            f"Instead got an unsupported type: {type(thing)}"
        )


class RunnableSeq(Runnable):
    """A simpler version of RunnableSequence."""

    def __init__(
        self,
        *steps: RunnableLike,
        name: Optional[str] = None,
    ) -> None:
        """Create a new RunnableSequence.

        Args:
            steps: The steps to include in the sequence.
            name: The name of the Runnable. Defaults to None.
            first: The first Runnable in the sequence. Defaults to None.
            middle: The middle Runnables in the sequence. Defaults to None.
            last: The last Runnable in the sequence. Defaults to None.

        Raises:
            ValueError: If the sequence has less than 2 steps.
        """
        steps_flat: list[Runnable] = []
        for step in steps:
            if isinstance(step, RunnableSequence):
                steps_flat.extend(step.steps)
            elif isinstance(step, RunnableSeq):
                steps_flat.extend(step.steps)
            else:
                steps_flat.append(coerce_to_runnable(step, name=None, trace=True))
        if len(steps_flat) < 2:
            raise ValueError(
                f"RunnableSeq must have at least 2 steps, got {len(steps_flat)}"
            )
        self.steps = steps_flat
        self.name = name

    def __or__(
        self,
        other: Any,
    ) -> Runnable:
        if isinstance(other, RunnableSequence):
            return RunnableSeq(
                *self.steps,
                other.first,
                *other.middle,
                other.last,
                name=self.name or other.name,
            )
        elif isinstance(other, RunnableSeq):
            return RunnableSeq(
                *self.steps,
                *other.steps,
                name=self.name or other.name,
            )
        else:
            return RunnableSeq(
                *self.steps,
                coerce_to_runnable(other, name=None, trace=True),
                name=self.name,
            )

    def __ror__(
        self,
        other: Any,
    ) -> Runnable:
        if isinstance(other, RunnableSequence):
            return RunnableSequence(
                other.first,
                *other.middle,
                other.last,
                *self.steps,
                name=other.name or self.name,
            )
        elif isinstance(other, RunnableSeq):
            return RunnableSeq(
                *other.steps,
                *self.steps,
                name=other.name or self.name,
            )
        else:
            return RunnableSequence(
                coerce_to_runnable(other, name=None, trace=True),
                *self.steps,
                name=self.name,
            )

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        if config is None:
            config = ensure_config()
        # setup callbacks and context
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # invoke all steps in sequence
        try:
            for i, step in enumerate(self.steps):
                # mark each step as a child run
                config = patch_config(
                    config, callbacks=run_manager.get_child(f"seq:step:{i+1}")
                )
                context = copy_context()
                context.run(_set_config_context, config)
                if i == 0:
                    input = context.run(step.invoke, input, config, **kwargs)
                else:
                    input = context.run(step.invoke, input, config)
        # finish the root run
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(input)
            return input

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Any:
        if config is None:
            config = ensure_config()
        # setup callbacks
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # invoke all steps in sequence
        try:
            for i, step in enumerate(self.steps):
                # mark each step as a child run
                config = patch_config(
                    config, callbacks=run_manager.get_child(f"seq:step:{i+1}")
                )
                context = copy_context()
                context.run(_set_config_context, config)
                if i == 0:
                    coro = step.ainvoke(input, config, **kwargs)
                else:
                    coro = step.ainvoke(input, config)
                if ASYNCIO_ACCEPTS_CONTEXT:
                    input = await asyncio.create_task(coro, context=context)
                else:
                    input = await asyncio.create_task(coro)
        # finish the root run
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(input)
            return input

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Any]:
        if config is None:
            config = ensure_config()
        # setup callbacks
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        try:
            # stream the last steps
            # transform the input stream of each step with the next
            # steps that don't natively support transforming an input stream will
            # buffer input in memory until all available, and then start emitting output
            for idx, step in enumerate(self.steps):
                config = patch_config(
                    config,
                    callbacks=run_manager.get_child(f"seq:step:{idx+1}"),
                )
                if idx == 0:
                    iterator = step.stream(input, config, **kwargs)
                else:
                    iterator = step.transform(iterator, config)
            if stream_handler := next(
                (
                    cast(_StreamingCallbackHandler, h)
                    for h in run_manager.handlers
                    if isinstance(h, _StreamingCallbackHandler)
                ),
                None,
            ):
                # populates streamed_output in astream_log() output if needed
                iterator = stream_handler.tap_output_iter(run_manager.run_id, iterator)
            output: Any = None
            add_supported = False
            for chunk in iterator:
                yield chunk
                # collect final output
                if output is None:
                    output = chunk
                elif add_supported:
                    try:
                        output = output + chunk
                    except TypeError:
                        output = chunk
                        add_supported = False
                else:
                    output = chunk
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(output)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Any]:
        if config is None:
            config = ensure_config()
        # setup callbacks
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        try:
            async with AsyncExitStack() as stack:
                # stream the last steps
                # transform the input stream of each step with the next
                # steps that don't natively support transforming an input stream will
                # buffer input in memory until all available, and then start emitting output
                for idx, step in enumerate(self.steps):
                    config = patch_config(
                        config,
                        callbacks=run_manager.get_child(f"seq:step:{idx+1}"),
                    )
                    if idx == 0:
                        aiterator = step.astream(input, config, **kwargs)
                    else:
                        aiterator = step.atransform(aiterator, config)
                    if hasattr(aiterator, "aclose"):
                        stack.push_async_callback(aiterator.aclose)
                if stream_handler := next(
                    (
                        cast(_StreamingCallbackHandler, h)
                        for h in run_manager.handlers
                        if isinstance(h, _StreamingCallbackHandler)
                    ),
                    None,
                ):
                    # populates streamed_output in astream_log() output if needed
                    aiterator = stream_handler.tap_output_aiter(
                        run_manager.run_id, aiterator
                    )
                output: Any = None
                add_supported = False
                async for chunk in aiterator:
                    yield chunk
                    # collect final output
                    if add_supported:
                        try:
                            output = output + chunk
                        except TypeError:
                            output = chunk
                            add_supported = False
                    else:
                        output = chunk
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(output)
