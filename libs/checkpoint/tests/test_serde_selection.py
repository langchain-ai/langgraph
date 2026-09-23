from typing import Any

from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import BaseCheckpointSaver


class FalsySerializer:
    def __bool__(self) -> bool:
        return False

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "value", str(obj).encode()

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return data[1].decode()

    async def adumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return self.dumps_typed(obj)

    async def aloads_typed(self, data: tuple[str, bytes]) -> Any:
        return self.loads_typed(data)


class NoopCache(BaseCache[int]):
    def get(self, keys):
        return {}

    async def aget(self, keys):
        return {}

    def set(self, pairs):
        return None

    async def aset(self, pairs):
        return None

    def clear(self, namespaces=None):
        return None

    async def aclear(self, namespaces=None):
        return None


def test_checkpoint_saver_preserves_falsy_serializer():
    serde = FalsySerializer()
    assert BaseCheckpointSaver[str](serde=serde).serde is serde


def test_cache_preserves_falsy_serializer():
    serde = FalsySerializer()
    assert NoopCache(serde=serde).serde is serde
