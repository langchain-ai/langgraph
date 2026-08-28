from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def _freeze(obj: Any, depth: int = 10) -> Hashable:
    if depth <= 0:
        return obj
    # Plain tuples are Hashable even when they contain unhashable values
    # (`isinstance` is type-based). Walk them so positional args and keyword
    # values take the same path.
    if type(obj) is tuple:
        return tuple(_freeze(x, depth - 1) for x in obj)
    if isinstance(obj, Hashable):
        return obj
    if isinstance(obj, Mapping):
        # sort keys so {"a":1,"b":2} == {"b":2,"a":1}
        return tuple(sorted((k, _freeze(v, depth - 1)) for k, v in obj.items()))
    if isinstance(obj, Sequence):
        return tuple(_freeze(x, depth - 1) for x in obj)
    # numpy / pandas / PIL etc. can provide their own .tobytes()
    if hasattr(obj, "tobytes"):
        dtype = getattr(obj, "dtype", None)
        key: tuple[Any, ...] = (
            type(obj).__name__,
            obj.tobytes(),
            getattr(obj, "shape", None),
            str(dtype) if dtype is not None else None,
            getattr(obj, "strides", None),
        )
        # PIL Image: mode is a str like "P" / "RGB". ndarray.size is an int.
        mode = getattr(obj, "mode", None)
        if isinstance(mode, str):
            palette = None
            getpalette = getattr(obj, "getpalette", None)
            if callable(getpalette):
                raw = getpalette()
                if raw is not None:
                    palette = tuple(raw)
            key += (mode, getattr(obj, "size", None), palette)
        return key
    return obj


def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    """Default cache key function that uses the arguments and keyword arguments to generate a hashable key."""
    import pickle

    # protocol 5 strikes a good balance between speed and size
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)
