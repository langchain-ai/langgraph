from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Self,
    Sequence,
)

from langchain_core.runnables import RunnableConfig

from langgraph.constants import CONFIG_KEY_KV
from langgraph.errors import InvalidUpdateError
from langgraph.kv.base import BaseKV
from langgraph.managed.base import (
    ChannelKeyPlaceholder,
    ConfiguredManagedValue,
    WritableManagedValue,
)

V = dict[str, Any]


Value = dict[str, V]
Update = dict[str, Optional[V]]


class ScopedValue(WritableManagedValue[Value, Update]):
    @staticmethod
    def configure(scope: str) -> ConfiguredManagedValue:
        return ConfiguredManagedValue(
            ScopedValue, {"scope": scope, "key": ChannelKeyPlaceholder}
        )

    @classmethod
    @contextmanager
    def enter(cls, config: RunnableConfig, **kwargs: Any) -> Iterator[Self]:
        with super().enter(config, **kwargs) as value:
            if value.kv is not None:
                saved = value.kv.list([value.ns])
                value.value = saved[value.ns]
            yield value

    @classmethod
    @asynccontextmanager
    async def aenter(cls, config: RunnableConfig, **kwargs: Any) -> AsyncIterator[Self]:
        async with super().aenter(config, **kwargs) as value:
            if value.kv is not None:
                saved = await value.kv.alist([value.ns])
                value.value = saved[value.ns]
            yield value

    def __init__(self, config: RunnableConfig, *, scope: str, key: str) -> None:
        self.scope = scope
        self.config = config
        self.value: Value = {}
        self.kv: BaseKV = config["configurable"].get(CONFIG_KEY_KV)
        if self.kv is None:
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
        if self.kv is None:
            self._process_update(values)
        else:
            return self.kv.put(self._process_update(values))

    async def aupdate(self, writes: Sequence[Update]) -> None:
        if self.kv is None:
            self._process_update(writes)
        else:
            return await self.kv.aput(self._process_update(writes))
