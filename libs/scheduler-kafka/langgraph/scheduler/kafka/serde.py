from typing import Any

import orjson

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

SERIALIZER = JsonPlusSerializer()


def loads(v: bytes) -> Any:
    return SERIALIZER.loads(v)


def dumps(v: Any) -> bytes:
    return orjson.dumps(v, default=_default)


def _default(v: Any) -> Any:
    # things we don't know how to serialize (eg. functions) ignore
    return None
