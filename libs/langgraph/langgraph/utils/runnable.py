import asyncio
import enum
import inspect
import sys
from contextlib import AsyncExitStack, contextmanager
from contextvars import Context, Token, copy_context
from functools import partial, wraps
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
)

from langchain_core.runnables.base import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
)
from langchain_core.runnables.base import (
    RunnableLike as LCRunnableLike,
)
from langchain_core.runnables.config import (
    run_in_executor,
    var_child_runnable_config,
)
from langchain_core.runnables.utils import Input, Output
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from typing_extensions import TypeGuard

from langgraph.constants import (
    CONF,
    CONFIG_KEY_PREVIOUS,
    CONFIG_KEY_STORE,
    CONFIG_KEY_STREAM_WRITER,
)
from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter
from langgraph.utils.config import (
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
)


def _set_config_context(
    config: RunnableConfig,
) -> tuple[Token[Optional[RunnableConfig]], Optional[dict[str, Any]]]:
    """Set the child Runnable config + tracing context.

    Args:
        config (RunnableConfig): The config to set.
    """
    from langchain_core.tracers.langchain import LangChainTracer

    config_token = var_child_runnable_config.set(config)
    current_context = None
    if (
        (callbacks := config.get("callbacks"))
        and (
            parent_run_id := getattr(callbacks, "parent_run_id", None)
        )  # Is callback manager
        and (
            tracer := next(
                (
                    handler
                    for handler in getattr(callbacks, "handlers", [])
                    if isinstance(handler, LangChainTracer)
                ),
                None,
            )
        )
        and (run := tracer.run_map.get(str(parent_run_id)))
    ):
        from langsmith.run_helpers import _set_tracing_context, get_tracing_context

        current_context = get_tracing_context()
        _set_tracing_context({"parent": run})
    return config_token, current_context


@contextmanager
def set_config_context(config: RunnableConfig) -> Generator[Context, None, None]:
    """Set the child Runnable config + tracing context.

    Args:
        config (RunnableConfig): The config to set.
    """
    from langsmith.run_helpers import _set_tracing_context

    ctx = copy_context()
    config_token, _ = ctx.run(_set_config_context, config)
    try:
        yield ctx
    finally:
        ctx.run(var_child_runnable_config.reset, config_token)
        ctx.run(
            _set_tracing_context,
            {
                "parent": None,
                "project_name": None,
                "tags": None,
                "metadata": None,
                "enabled": None,
                "client": None,
            },
        )


# Before Python 3.11 native StrEnum is not available
class StrEnum(str, enum.Enum):
    """A string enum."""


# Special type to denote any type is accepted
ANY_TYPE = object()

ASYNCIO_ACCEPTS_CONTEXT = sys.version_info >= (3, 11)

# List of keyword arguments that can be injected at runtime from the config object.
# A named argument may appear multiple times if it appears with distinct types.
KWARGS_CONFIG_KEYS: tuple[tuple[str, tuple[Any, ...], str, Any], ...] = (
    (
        sys.intern("writer"),
        (StreamWriter, "StreamWriter", inspect.Parameter.empty),
        CONFIG_KEY_STREAM_WRITER,
        lambda _: None,
    ),
    (
        # Covers store that is not optional (will raise an error if a store
        # cannot be injected).
        sys.intern("store"),
        (
            BaseStore,
            "BaseStore",
            inspect.Parameter.empty,
        ),
        CONFIG_KEY_STORE,
        inspect.Parameter.empty,
    ),
    (
        # Covers store that is optional. Will set to None if not found in config.
        sys.intern("store"),
        (
            Optional[BaseStore],
            # Best effort to catch some forward references.
            # This will not work for cases like `"Union[None, BaseStore]"`,
            # we'll need to re-write logic to use get_type_hints()
            # to resolve forward references.
            "Optional[BaseStore]",
        ),
        CONFIG_KEY_STORE,
        None,
    ),
    (
        sys.intern("previous"),
        (ANY_TYPE,),
        CONFIG_KEY_PREVIOUS,
        inspect.Parameter.empty,
    ),
)
"""List of kwargs that can be passed to functions, and their corresponding
config keys, default values and type annotations.

Used to configure keyword arguments that can be injected at runtime
from the config object as kwargs to `invoke`, `ainvoke`, `stream` and `astream`.

For a keyword to be injected from the config object, the function signature
must contain a kwarg with the same name and a matching type annotation.

Each tuple contains:
- the name of the kwarg in the function signature
- the type annotation(s) for the kwarg
- the config key to look for the value in
- the default value for the kwarg
"""

VALID_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


class _RunnableWithWriter(Protocol[Input, Output]):
    def __call__(self, state: Input, *, writer: StreamWriter) -> Output: ...


class _RunnableWithStore(Protocol[Input, Output]):
    def __call__(self, state: Input, *, store: BaseStore) -> Output: ...


class _RunnableWithWriterStore(Protocol[Input, Output]):
    def __call__(
        self, state: Input, *, writer: StreamWriter, store: BaseStore
    ) -> Output: ...


class _RunnableWithConfigWriter(Protocol[Input, Output]):
    def __call__(
        self, state: Input, *, config: RunnableConfig, writer: StreamWriter
    ) -> Output: ...


class _RunnableWithConfigStore(Protocol[Input, Output]):
    def __call__(
        self, state: Input, *, config: RunnableConfig, store: BaseStore
    ) -> Output: ...


class _RunnableWithConfigWriterStore(Protocol[Input, Output]):
    def __call__(
        self,
        state: Input,
        *,
        config: RunnableConfig,
        writer: StreamWriter,
        store: BaseStore,
    ) -> Output: ...


RunnableLike = Union[
    LCRunnableLike,
    _RunnableWithWriter[Input, Output],
    _RunnableWithStore[Input, Output],
    _RunnableWithWriterStore[Input, Output],
    _RunnableWithConfigWriter[Input, Output],
    _RunnableWithConfigStore[Input, Output],
    _RunnableWithConfigWriterStore[Input, Output],
]


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
        explode_args: bool = False,
        func_accepts_config: Optional[bool] = None,
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
        self.afunc = afunc
        self.tags = tags
        self.kwargs = kwargs
        self.trace = trace
        self.recurse = recurse
        self.explode_args = explode_args
        # check signature
        if func is None and afunc is None:
            raise ValueError("At least one of func or afunc must be provided.")

        if func_accepts_config is not None:
            self.func_accepts_config = func_accepts_config
            self.func_accepts: dict[str, Tuple[str, Any]] = {}
        else:
            params = inspect.signature(cast(Callable, func or afunc)).parameters

            self.func_accepts_config = "config" in params
            # Mapping from kwarg name to (config key, default value) to be used.
            # The default value is used if the config key is not found in the config.
            self.func_accepts = {}

            for kw, typ, config_key, default in KWARGS_CONFIG_KEYS:
                p = params.get(kw)

                if p is None or p.kind not in VALID_KINDS:
                    # If parameter is not found or is not a valid kind, skip
                    continue

                if typ != (ANY_TYPE,) and p.annotation not in typ:
                    # A specific type is required, but the function annotation does
                    # not match the expected type.
                    continue

                # If the kwarg is accepted by the function, store the default value
                self.func_accepts[kw] = (config_key, default)

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
        if config is None:
            config = ensure_config()
        if self.explode_args:
            args, _kwargs = input
            kwargs = {**self.kwargs, **_kwargs, **kwargs}
        else:
            args = (input,)
            kwargs = {**self.kwargs, **kwargs}
        if self.func_accepts_config:
            kwargs["config"] = config
        _conf = config[CONF]

        for kw, (config_key, default_value) in self.func_accepts.items():
            # If the kwarg is already set, use the set value
            if kw in kwargs:
                continue

            if (
                # If the kwarg is requested, but isn't in the config AND has no
                # default value, raise an error
                config_key not in _conf and default_value is inspect.Parameter.empty
            ):
                raise ValueError(
                    f"Missing required config key '{config_key}' for '{self.name}'."
                )

            kwargs[kw] = _conf.get(config_key, default_value)

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
                with set_config_context(child_config) as context:
                    ret = context.run(self.func, *args, **kwargs)
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise
            else:
                run_manager.on_chain_end(ret)
        else:
            with set_config_context(config) as context:
                ret = context.run(self.func, *args, **kwargs)
        if isinstance(ret, Runnable) and self.recurse:
            return ret.invoke(input, config)
        return ret

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        if not self.afunc:
            return self.invoke(input, config)
        if config is None:
            config = ensure_config()
        if self.explode_args:
            args, _kwargs = input
            kwargs = {**self.kwargs, **_kwargs, **kwargs}
        else:
            args = (input,)
            kwargs = {**self.kwargs, **kwargs}
        if self.func_accepts_config:
            kwargs["config"] = config
        _conf = config[CONF]
        for kw, (config_key, default_value) in self.func_accepts.items():
            # If the kwarg has already been set, use the set value
            if kw in kwargs:
                continue

            if (
                # If the kwarg is requested, but isn't in the config AND has no
                # default value, raise an error
                config_key not in _conf and default_value is inspect.Parameter.empty
            ):
                raise ValueError(
                    f"Missing required config key '{config_key}' for '{self.name}'."
                )
            kwargs[kw] = _conf.get(config_key, default_value)
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
                with set_config_context(child_config) as context:
                    coro = cast(Coroutine[None, None, Any], self.afunc(*args, **kwargs))
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
            with set_config_context(config) as context:
                if ASYNCIO_ACCEPTS_CONTEXT:
                    coro = cast(Coroutine[None, None, Any], self.afunc(*args, **kwargs))
                    ret = await asyncio.create_task(coro, context=context)
                else:
                    ret = await self.afunc(*args, **kwargs)
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
    """Sequence of Runnables, where the output of each is the input of the next.

    RunnableSeq is a simpler version of RunnableSequence that is internal to
    LangGraph.
    """

    def __init__(
        self,
        *steps: RunnableLike,
        name: Optional[str] = None,
        trace_inputs: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Create a new RunnableSeq.

        Args:
            steps: The steps to include in the sequence.
            name: The name of the Runnable. Defaults to None.

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
        self.trace_inputs = trace_inputs

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
            self.trace_inputs(input) if self.trace_inputs is not None else input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # invoke all steps in sequence
        try:
            for i, step in enumerate(self.steps):
                # mark each step as a child run
                config = patch_config(
                    config, callbacks=run_manager.get_child(f"seq:step:{i + 1}")
                )
                if i == 0:
                    input = step.invoke(input, config, **kwargs)
                else:
                    input = step.invoke(input, config)
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
            self.trace_inputs(input) if self.trace_inputs is not None else input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # invoke all steps in sequence
        try:
            for i, step in enumerate(self.steps):
                # mark each step as a child run
                config = patch_config(
                    config, callbacks=run_manager.get_child(f"seq:step:{i + 1}")
                )
                if i == 0:
                    input = await step.ainvoke(input, config, **kwargs)
                else:
                    input = await step.ainvoke(input, config)
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
            self.trace_inputs(input) if self.trace_inputs is not None else input,
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
                    callbacks=run_manager.get_child(f"seq:step:{idx + 1}"),
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
            self.trace_inputs(input) if self.trace_inputs is not None else input,
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
                        callbacks=run_manager.get_child(f"seq:step:{idx + 1}"),
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
