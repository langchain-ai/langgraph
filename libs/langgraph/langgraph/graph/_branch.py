from __future__ import annotations

from collections.abc import Awaitable, Hashable, Sequence
from inspect import (
    isfunction,
    ismethod,
    signature,
)
from itertools import zip_longest
from types import FunctionType
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
)

from langgraph._internal._runnable import (
    RunnableCallable,
)
from langgraph.constants import END, START
from langgraph.errors import InvalidUpdateError
from langgraph.pregel._write import PASSTHROUGH, ChannelWrite, ChannelWriteEntry
from langgraph.types import Send

_Writer = Callable[
    [Sequence[Union[str, Send]], bool],
    Sequence[Union[ChannelWriteEntry, Send]],
]


def _get_branch_path_input_schema(
    path: Callable[..., Hashable | Sequence[Hashable]]
    | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
    | Runnable[Any, Hashable | Sequence[Hashable]],
) -> type[Any] | None:
    input = None
    # detect input schema annotation in the branch callable
    try:
        callable_: (
            Callable[..., Hashable | Sequence[Hashable]]
            | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
            | None
        ) = None
        if isinstance(path, (RunnableCallable, RunnableLambda)):
            if isfunction(path.func) or ismethod(path.func):
                callable_ = path.func
            elif (callable_method := getattr(path.func, "__call__", None)) and ismethod(
                callable_method
            ):
                callable_ = callable_method
            elif isfunction(path.afunc) or ismethod(path.afunc):
                callable_ = path.afunc
            elif (
                callable_method := getattr(path.afunc, "__call__", None)
            ) and ismethod(callable_method):
                callable_ = callable_method
        elif callable(path):
            callable_ = path

        if callable_ is not None and (hints := get_type_hints(callable_)):
            first_parameter_name = next(
                iter(signature(cast(FunctionType, callable_)).parameters.keys())
            )
            if input_hint := hints.get(first_parameter_name):
                if isinstance(input_hint, type) and get_type_hints(input_hint):
                    input = input_hint
    except (TypeError, StopIteration):
        pass

    return input


class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None
    input_schema: type[Any] | None = None

    @classmethod
    def from_path(
        cls,
        path: Runnable[Any, Hashable | list[Hashable]],
        path_map: dict[Hashable, str] | list[str] | None,
        infer_schema: bool = False,
    ) -> BranchSpec:
        # coerce path_map to a dictionary
        path_map_: dict[Hashable, str] | None = None
        try:
            if isinstance(path_map, dict):
                path_map_ = path_map.copy()
            elif isinstance(path_map, list):
                path_map_ = {name: name for name in path_map}
            else:
                # find func
                func: Callable | None = None
                if isinstance(path, (RunnableCallable, RunnableLambda)):
                    func = path.func or path.afunc
                if func is not None:
                    # find callable method
                    if (cal := getattr(path, "__call__", None)) and ismethod(cal):
                        func = cal
                    # get the return type
                    if rtn_type := get_type_hints(func).get("return"):
                        if get_origin(rtn_type) is Literal:
                            path_map_ = {name: name for name in get_args(rtn_type)}
        except Exception:
            pass
        # infer input schema
        input_schema = _get_branch_path_input_schema(path) if infer_schema else None
        # create branch
        return cls(path=path, ends=path_map_, input_schema=input_schema)

    def run(
        self,
        writer: _Writer,
        reader: Callable[[RunnableConfig], Any] | None = None,
    ) -> RunnableCallable:
        return ChannelWrite.register_writer(
            RunnableCallable(
                func=self._route,
                afunc=self._aroute,
                writer=writer,
                reader=reader,
                name=None,
                trace=False,
            ),
            list(
                zip_longest(
                    writer([e for e in self.ends.values()], True),
                    [str(la) for la, e in self.ends.items()],
                )
            )
            if self.ends
            else None,
        )

    def _route(
        self,
        input: Any,
        config: RunnableConfig,
        *,
        reader: Callable[[RunnableConfig], Any] | None,
        writer: _Writer,
    ) -> Runnable:
        if reader:
            value = reader(config)
            # passthrough additional keys from node to branch
            # only doable when using dict states
            if (
                isinstance(value, dict)
                and isinstance(input, dict)
                and self.input_schema is None
            ):
                value = {**input, **value}
        else:
            value = input
        result = self.path.invoke(value, config)
        return self._finish(writer, input, result, config)

    async def _aroute(
        self,
        input: Any,
        config: RunnableConfig,
        *,
        reader: Callable[[RunnableConfig], Any] | None,
        writer: _Writer,
    ) -> Runnable:
        if reader:
            value = reader(config)
            # passthrough additional keys from node to branch
            # only doable when using dict states
            if (
                isinstance(value, dict)
                and isinstance(input, dict)
                and self.input_schema is None
            ):
                value = {**input, **value}
        else:
            value = input
        result = await self.path.ainvoke(value, config)
        return self._finish(writer, input, result, config)

    def _finish(
        self,
        writer: _Writer,
        input: Any,
        result: Any,
        config: RunnableConfig,
    ) -> Runnable | Any:
        if not isinstance(result, (list, tuple)):
            result = [result]
        if self.ends:
            destinations: Sequence[Send | str] = [
                r if isinstance(r, Send) else self.ends[r] for r in result
            ]
        else:
            destinations = cast(Sequence[Union[Send, str]], result)
        if any(dest is None or dest == START for dest in destinations):
            raise ValueError("Branch did not return a valid destination")
        if any(p.node == END for p in destinations if isinstance(p, Send)):
            raise InvalidUpdateError("Cannot send a packet to the END node")
        entries = writer(destinations, False)
        if not entries:
            return input
        else:
            need_passthrough = False
            for e in entries:
                if isinstance(e, ChannelWriteEntry):
                    if e.value is PASSTHROUGH:
                        need_passthrough = True
                        break
            if need_passthrough:
                return ChannelWrite(entries)
            else:
                ChannelWrite.do_write(config, entries)
                return input
