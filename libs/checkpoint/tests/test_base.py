from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver


class FalsySerializer:
    def __bool__(self) -> bool:
        return False

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return type(obj).__name__, b""

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return data[1]


def test_base_checkpoint_saver_preserves_falsy_serializer() -> None:
    serde = FalsySerializer()

    saver = BaseCheckpointSaver(serde=serde)

    assert saver.serde is serde
