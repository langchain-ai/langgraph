from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def _pil_palette(obj: Any) -> tuple[int, ...] | None:
    """Best-effort read of a PIL-style palette (``getpalette()``).

    Returns ``None`` for objects that don't expose a palette, can't be read
    without raising, or report an empty palette. All access is defensive so
    this never adds a hard dependency on PIL.
    """
    getpalette = getattr(obj, "getpalette", None)
    if not callable(getpalette):
        return None
    try:
        palette = getpalette()
    except Exception:
        return None
    return tuple(palette) if palette else None


def _freeze(obj: Any, depth: int = 10) -> Hashable:
    if depth <= 0:
        # Depth exhausted: stop recursing and use a stable, hashable summary so
        # the caller (which pickle.dumps the result) never receives an
        # unhashable value. Two deeply-equal structures still freeze equally.
        return (type(obj).__qualname__, "max-depth")
    # Recurse into tuples so their contents are frozen consistently on both the
    # positional and keyword paths (otherwise the same value passed positionally
    # vs as a keyword would produce different cache keys).
    if isinstance(obj, tuple):
        return tuple(_freeze(x, depth - 1) for x in obj)
    elif isinstance(obj, Mapping):
        # sort keys so {"a":1,"b":2} == {"b":2,"a":1}
        return tuple(sorted((k, _freeze(v, depth - 1)) for k, v in obj.items()))
    elif isinstance(obj, Hashable):
        # already hashable, no need to freeze
        return obj
    elif isinstance(obj, Sequence):
        return tuple(_freeze(x, depth - 1) for x in obj)
    # Objects exposing .tobytes() (numpy arrays, PIL images, torch/jax tensors,
    # pandas DataFrames, …). Include the available distinguishing metadata so
    # byte-identical-but-semantically-distinct inputs (e.g. uint8 [200] vs
    # int8 [-56]; same-indices/different-palette PIL images) no longer collide.
    elif hasattr(obj, "tobytes"):
        return (
            type(obj).__qualname__,
            getattr(obj, "shape", None),
            str(getattr(obj, "dtype", "")) or None,  # numpy/torch/jax/cupy
            getattr(obj, "mode", None),  # PIL
            getattr(obj, "size", None),  # PIL
            _pil_palette(obj),  # PIL P-mode
            getattr(obj, "strides", None),
            obj.tobytes(),
        )
    return obj  # strings, ints, dataclasses with frozen=True, etc.


def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    """Default cache key function that uses the arguments and keyword arguments to generate a hashable key."""
    import pickle

    # protocol 5 strikes a good balance between speed and size
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)
