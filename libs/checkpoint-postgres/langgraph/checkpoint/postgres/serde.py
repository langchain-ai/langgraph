from base64 import b64encode
from typing import Any

import orjson

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def default(obj):
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    elif hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, (bytes, bytearray)):
        return b64encode(obj).decode()
    return None


def json_dumpb(obj):
    return orjson.dumps(
        obj,
        default=default,
        option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )


def json_dumps(obj):
    return (
        orjson.dumps(
            obj,
            default=default,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
        )
        .replace(b"\u0000", b"")  # null unicode char not allowed in json
        .decode()
    )


class JsonAndBinarySerializer(JsonPlusSerializer):
    def _default(self, obj):
        if isinstance(obj, (bytes, bytearray)):
            return self._encode_constructor_args(
                obj.__class__, method="fromhex", args=[obj.hex()]
            )
        return super()._default(obj)

    # TODO: rename this to dumps_typed / loads_typed for consistency
    def dumps(self, obj: Any) -> tuple[str, bytes]:
        if isinstance(obj, bytes):
            return "bytes", obj
        elif isinstance(obj, bytearray):
            return "bytearray", obj

        return "json", super().dumps(obj)

    def loads(self, s: tuple[str, bytes]) -> Any:
        if s[0] == "bytes":
            return s[1]
        elif s[0] == "bytearray":
            return bytearray(s[1])
        elif s[0] == "json":
            return super().loads(s[1])
        else:
            raise NotImplementedError(f"Unknown serialization type: {s[0]}")
