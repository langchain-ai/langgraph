import collections.abc
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Sequence,
    Type,
)

from typing_extensions import NotRequired, Required, Self

from langgraph.constants import CONF
from langgraph.errors import InvalidUpdateError
from langgraph.managed.base import (
    ChannelKeyPlaceholder,
    ChannelTypePlaceholder,
    ConfiguredManagedValue,
    WritableManagedValue,
)
from langgraph.store.base import PutOp
from langgraph.types import LoopProtocol

V = dict[str, Any]


Value = dict[str, V]
Update = dict[str, Optional[V]]


# Adapted from typing_extensions
def _strip_extras(t):  # type: ignore[no-untyped-def]
    """Strips Annotated, Required and NotRequired from a given type."""
    if hasattr(t, "__origin__"):
        return _strip_extras(t.__origin__)
    if hasattr(t, "__origin__") and t.__origin__ in (Required, NotRequired):
        return _strip_extras(t.__args__[0])

    return t


class SharedValue(WritableManagedValue[Value, Update]):
    @staticmethod
    def on(scope: str) -> ConfiguredManagedValue:
        return ConfiguredManagedValue(
            SharedValue,
            {
                "scope": scope,
                "key": ChannelKeyPlaceholder,
                "typ": ChannelTypePlaceholder,
            },
        )

    @classmethod
    @contextmanager
    def enter(cls, loop: LoopProtocol, **kwargs: Any) -> Iterator[Self]:
        with super().enter(loop, **kwargs) as value:
            if loop.store is not None:
                saved = loop.store.search(value.ns)
                value.value = {it.key: it.value for it in saved}
            yield value

    @classmethod
    @asynccontextmanager
    async def aenter(cls, loop: LoopProtocol, **kwargs: Any) -> AsyncIterator[Self]:
        async with super().aenter(loop, **kwargs) as value:
            if loop.store is not None:
                saved = await loop.store.asearch(value.ns)
                value.value = {it.key: it.value for it in saved}
            yield value

    def __init__(
        self, loop: LoopProtocol, *, typ: Type[Any], scope: str, key: str
    ) -> None:
        super().__init__(loop)
        if typ := _strip_extras(typ):
            if typ not in (
                dict,
                collections.abc.Mapping,
                collections.abc.MutableMapping,
            ):
                raise ValueError("SharedValue must be a dict")
        self.scope = scope
        self.value: Value = {}
        if self.loop.store is None:
            pass
        elif scope_value := self.loop.config[CONF].get(self.scope):
            self.ns = ("scoped", scope, key, scope_value)
        else:
            raise ValueError(
                f"Scope {scope} for shared state key not in config.configurable"
            )

    def __call__(self) -> Value:
        return self.value

    def _process_update(self, values: Sequence[Update]) -> list[PutOp]:
        writes: list[PutOp] = []
        for vv in values:
            for k, v in vv.items():
                if v is None:
                    if k in self.value:
                        del self.value[k]
                        writes.append(PutOp(self.ns, k, None))
                elif not isinstance(v, dict):
                    raise InvalidUpdateError("Received a non-dict value")
                else:
                    self.value[k] = v
                    writes.append(PutOp(self.ns, k, v))
        return writes

    def update(self, values: Sequence[Update]) -> None:
        if self.loop.store is None:
            self._process_update(values)
        else:
            return self.loop.store.batch(self._process_update(values))

    async def aupdate(self, writes: Sequence[Update]) -> None:
        if self.loop.store is None:
            self._process_update(writes)
        else:
            return await self.loop.store.abatch(self._process_update(writes))
