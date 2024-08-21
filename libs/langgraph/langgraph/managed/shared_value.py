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

from langchain_core.runnables import RunnableConfig
from typing_extensions import NotRequired, Required, Self

from langgraph.constants import CONFIG_KEY_STORE
from langgraph.errors import InvalidUpdateError
from langgraph.managed.base import (
    ChannelKeyPlaceholder,
    ChannelTypePlaceholder,
    ConfiguredManagedValue,
    WritableManagedValue,
)
from langgraph.store.base import BaseStore

V = dict[str, Any]


Value = dict[str, V]
Update = dict[str, Optional[V]]


# Adapted from typing_extensions
def _strip_extras(t):
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
    def enter(cls, config: RunnableConfig, **kwargs: Any) -> Iterator[Self]:
        with super().enter(config, **kwargs) as value:
            if value.store is not None:
                saved = value.store.list([value.ns])
                value.value = saved[value.ns] or {}
            yield value

    @classmethod
    @asynccontextmanager
    async def aenter(cls, config: RunnableConfig, **kwargs: Any) -> AsyncIterator[Self]:
        async with super().aenter(config, **kwargs) as value:
            if value.store is not None:
                saved = await value.store.alist([value.ns])
                value.value = saved[value.ns] or {}
            yield value

    def __init__(
        self, config: RunnableConfig, *, typ: Type[Any], scope: str, key: str
    ) -> None:
        if typ := _strip_extras(typ):
            if typ not in (
                dict,
                collections.abc.Mapping,
                collections.abc.MutableMapping,
            ):
                raise ValueError("SharedValue must be a dict")
        self.scope = scope
        self.config = config
        self.value: Value = {}
        self.store: BaseStore = config["configurable"].get(CONFIG_KEY_STORE)
        if self.store is None:
            self.ns: Optional[str] = None
        elif scope_value := config["configurable"].get(self.scope):
            self.ns = f"scoped:{scope}:{key}:{scope_value}"
        else:
            raise ValueError(
                f"Scope {scope} for shared state key not in config.configurable"
            )

    def __call__(self, step: int) -> Value:
        return self.value.copy()

    def _process_update(
        self, values: Sequence[Update]
    ) -> list[tuple[str, str, Optional[dict[str, Any]]]]:
        writes = []
        for vv in values:
            for k, v in vv.items():
                if v is None:
                    if k in self.value:
                        self.value[k] = None
                        writes.append((self.ns, k, None))
                elif not isinstance(v, dict):
                    raise InvalidUpdateError("Received a non-dict value")
                else:
                    self.value[k] = v
                    writes.append((self.ns, k, v))
        return writes

    def update(self, values: Sequence[Update]) -> None:
        if self.store is None:
            self._process_update(values)
        else:
            return self.store.put(self._process_update(values))

    async def aupdate(self, writes: Sequence[Update]) -> None:
        if self.store is None:
            self._process_update(writes)
        else:
            return await self.store.aput(self._process_update(writes))
