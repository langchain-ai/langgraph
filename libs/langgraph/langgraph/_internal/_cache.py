from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def _freeze(obj: Any, depth: int = 10) -> Hashable:
    if isinstance(obj, Hashable) or depth <= 0:
        # already hashable, no need to freeze
        return obj
    elif isinstance(obj, Mapping):
        # sort keys so {"a":1,"b":2} == {"b":2,"a":1}
        items = [(k, _freeze(v, depth - 1)) for k, v in obj.items()]
        try:
            items.sort()
        except TypeError:
            # Hashable keys are not necessarily orderable (Enum members, mixed key
            # types), and a cache key must not raise for a valid call. Fall back to
            # a stable serialization of the key; mappings whose keys are comparable
            # keep the cache keys they have today (issue #9566).
            import pickle

            items.sort(
                key=lambda item: pickle.dumps(item[0], protocol=5, fix_imports=False)
            )
        return tuple(items)
    elif isinstance(obj, Sequence):
        return tuple(_freeze(x, depth - 1) for x in obj)
    # numpy / pandas etc. can provide their own .tobytes()
    elif hasattr(obj, "tobytes"):
        return (
            type(obj).__name__,
            obj.tobytes(),
            obj.shape if hasattr(obj, "shape") else None,
        )
    return obj  # strings, ints, dataclasses with frozen=True, etc.


def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    """Default cache key function that uses the arguments and keyword arguments to generate a hashable key."""
    import pickle

    # protocol 5 strikes a good balance between speed and size
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)
