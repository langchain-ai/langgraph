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
    Optional,
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

from langgraph.constants import END, START
from langgraph.errors import InvalidUpdateError
from langgraph.pregel.write import PASSTHROUGH, ChannelWrite, ChannelWriteEntry
from langgraph.types import Send
from langgraph.utils.runnable import (
    RunnableCallable,
)

Writer = Callable[
    [Sequence[Union[str, Send]], bool],
    Sequence[Union[ChannelWriteEntry, Send]],
]


def _get_branch_path_input_schema(
    path: Union[
        Callable[..., Union[Hashable, list[Hashable]]],
        Callable[..., Awaitable[Union[Hashable, list[Hashable]]]],
        Runnable[Any, Union[Hashable, list[Hashable]]],
    ],
) -> Optional[type[Any]]:
    input = None
    # detect input schema annotation in the branch callable
    try:
        callable_: Optional[
            Union[
                Callable[..., Union[Hashable, list[Hashable]]],
                Callable[..., Awaitable[Union[Hashable, list[Hashable]]]],
            ]
        ] = None
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


class Branch(NamedTuple):
    path: Runnable[Any, Union[Hashable, list[Hashable]]]
    ends: Optional[dict[Hashable, str]]
    then: Optional[str] = None
    input_schema: Optional[type[Any]] = None

    @classmethod
    def from_path(
        cls,
        path: Runnable[Any, Union[Hashable, list[Hashable]]],
        path_map: Optional[Union[dict[Hashable, str], list[str]]],
        then: Optional[str] = None,
        infer_schema: bool = False,
    ) -> "Branch":
        # coerce path_map to a dictionary
        path_map_: Optional[dict[Hashable, str]] = None
        try:
            if isinstance(path_map, dict):
                path_map_ = path_map.copy()
            elif isinstance(path_map, list):
                path_map_ = {name: name for name in path_map}
            else:
                # find func
                func: Optional[Callable] = None
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
        return cls(path=path, ends=path_map_, then=then, input_schema=input_schema)

    def run(
        self,
        writer: Writer,
        reader: Optional[Callable[[RunnableConfig], Any]] = None,
    ) -> RunnableCallable:
        return ChannelWrite.register_writer(
            RunnableCallable(
                func=self._route,
                afunc=self._aroute,
                writer=writer,
                reader=reader,
                name=None,
                trace=False,
                func_accepts_config=True,
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
        reader: Optional[Callable[[RunnableConfig], Any]],
        writer: Writer,
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
        reader: Optional[Callable[[RunnableConfig], Any]],
        writer: Writer,
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
        writer: Writer,
        input: Any,
        result: Any,
        config: RunnableConfig,
    ) -> Union[Runnable, Any]:
        if not isinstance(result, (list, tuple)):
            result = [result]
        if self.ends:
            destinations: Sequence[Union[Send, str]] = [
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
