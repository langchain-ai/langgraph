from typing import Any

import orjson


def loads(v: bytes) -> Any:
    return orjson.loads(v)


def dumps(v: Any) -> bytes:
    return orjson.dumps(v, default=_default)


def _default(v: Any) -> Any:
    # things we don't know how to serialize (eg. functions) ignore
    return None
