import enum
import inspect
from abc import ABC, abstractmethod
from contextvars import copy_context
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

from langgraph.utils.config import (
    AnyConfig,
    ensure_config,
    get_runtree_for_config,
    patch_config,
    set_config_in_context,
)


# Before Python 3.11 native StrEnum is not available
class StrEnum(str, enum.Enum):
    """A string enum."""


Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)


class Runnable(Generic[Input, Output], ABC):
    """A unit of work that can be invoked, batched, streamed, transformed and composed."""  # noqa: E501

    name: Optional[str]
    """The name of the Runnable. Used for debugging and tracing."""

    """ --- Public API --- """

    def get_name(self, name: Optional[str] = None) -> str:
        """Get the name of the Runnable.

        Returns:
            The name of the Runnable.
        """
        return name or self.name or self.__class__.__name__

    @abstractmethod
    def invoke(
        self, input: Input, config: Optional[AnyConfig] = None, **kwargs: Any
    ) -> Output:
        """Transform a single input into an output. Override to implement.

        Args:
            input: The input to the Runnable.
            config: A config to use when invoking the Runnable.
               The config supports standard keys like 'tags', 'metadata' for tracing
               purposes, 'max_concurrency' for controlling how much work to do
               in parallel, and other keys. Please refer to the RunnableConfig
               for more details.

        Returns:
            The output of the Runnable.
        """


RunnableLike = Union[
    Runnable[Input, Output],
    Callable[[Input], Output],
    Callable[[Input, AnyConfig], Output],
]


class RunnableCallable(Runnable):
    """Wraps a callable as a Runnable."""

    def __init__(
        self,
        func: Optional[Callable[..., Union[Any, Runnable]]],
        *,
        name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        trace: bool = True,
        recurse: bool = True,
        explode_args: bool = False,
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
        self.func = func
        self.tags = tags
        self.kwargs = kwargs
        self.trace = trace
        self.explode_args = explode_args
        self.func_accepts_config = "config" in inspect.signature(func).parameters
        self.recurse = recurse

    def __repr__(self) -> str:
        repr_args = {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"name", "func", "afunc", "config", "kwargs", "trace"}
        }
        return f"{self.get_name()}({', '.join(f'{k}={v!r}' for k, v in repr_args.items())})"

    def invoke(
        self, input: Any, config: Optional[AnyConfig] = None, **kwargs: Any
    ) -> Any:
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

        context = copy_context()
        if self.trace:
            runtree = get_runtree_for_config(
                config,
                input,
                name=cast(str, config.get("run_name", self.get_name())),
                tags=self.tags,
            )
            try:
                child_config = patch_config(config, runtree=runtree)
                context = copy_context()
                context.run(set_config_in_context, child_config)
                ret = context.run(self.func, *args, **kwargs)
            except BaseException as e:
                runtree.end(error=str(e))
                raise
            else:
                runtree.end(outputs=ret if isinstance(ret, dict) else {"output": ret})
        else:
            context.run(set_config_in_context, config)
            ret = context.run(self.func, *args, **kwargs)
        if isinstance(ret, Runnable) and self.recurse:
            return ret.invoke(input, config)
        return ret


def coerce_to_runnable(
    thing: RunnableLike, *, name: Optional[str] = None, trace: bool = True
) -> Runnable:
    """Coerce a runnable-like object into a Runnable.

    Args:
        thing: A runnable-like object.

    Returns:
        A Runnable.
    """
    try:
        from langchain_core.runnables import Runnable as LC_Runnable
    except ImportError:
        LC_Runnable = None
    if LC_Runnable and isinstance(thing, LC_Runnable):
        return thing
    elif isinstance(thing, Runnable):
        return thing
    elif callable(thing):
        return RunnableCallable(
            thing,
            name=name,
            trace=trace,
        )
    else:
        raise TypeError(
            f"Expected a Runnable, callable or dict."
            f"Instead got an unsupported type: {type(thing)}"
        )


class RunnableSeq(Runnable):
    """Sequence of Runnables, where the output of each is the input of the next."""

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
            if isinstance(step, RunnableSeq):
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

    def invoke(
        self, input: Input, config: Optional[AnyConfig] = None, **kwargs: Any
    ) -> Any:
        if config is None:
            config = ensure_config()
        # setup runtree and context
        runtree = get_runtree_for_config(
            config,
            self.trace_inputs(input) if self.trace_inputs is not None else input,
            name=cast(str, config.get("run_name", self.get_name())),
        )

        # invoke all steps in sequence
        try:
            for i, step in enumerate(self.steps):
                # mark each step as a child run
                config = patch_config(config, runtree=runtree)
                if i == 0:
                    input = step.invoke(input, config, **kwargs)
                else:
                    input = step.invoke(input, config)
        # finish the root run
        except BaseException as e:
            runtree.end(error=str(e))
            raise
        else:
            runtree.end(outputs=input if isinstance(input, dict) else {"output": input})
            return input
