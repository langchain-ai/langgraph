from __future__ import annotations

from typing import Any

import pytest

from langgraph._internal._cache import _freeze, default_cache_key


class _BytesLike:
    """Minimal unhashable stub mirroring the numpy/torch ``.tobytes()`` + ``dtype``
    + ``shape`` contract, so the cache-key collision regression can be tested
    without adding numpy as a test dependency."""

    __hash__ = None  # type: ignore[assignment]  # mark unhashable

    def __init__(
        self,
        data: bytes,
        *,
        dtype: str = "",
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self._data = data
        self.dtype = dtype
        self.shape = shape

    def tobytes(self) -> bytes:
        return self._data


class _PilLike:
    """Minimal unhashable stub mirroring the PIL ``.tobytes()`` + ``mode`` +
    ``size`` + ``getpalette()`` contract."""

    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        data: bytes,
        *,
        mode: str = "P",
        size: tuple[int, int] = (1, 1),
        palette: list[int] | None = None,
    ) -> None:
        self._data = data
        self.mode = mode
        self.size = size
        self._palette = palette

    def tobytes(self) -> bytes:
        return self._data

    def getpalette(self) -> list[int] | None:
        return self._palette


# --------------------------------------------------------------------------- #
# Fix 1: collision regression — distinct metadata must yield distinct keys.
# --------------------------------------------------------------------------- #


def test_tobytes_distinct_dtype_distinct_keys() -> None:
    # Same raw bytes + same shape, different dtype -> must NOT collide.
    # Mirrors np.frombuffer(b"\xc8", uint8)=[200] vs int8=[-56].
    x = _BytesLike(b"\xc8", dtype="uint8", shape=(1,))
    y = _BytesLike(b"\xc8", dtype="int8", shape=(1,))
    assert default_cache_key(arr=x) != default_cache_key(arr=y)


def test_pil_distinct_palette_distinct_keys() -> None:
    # Same indices + same mode/size, different palette -> must NOT collide.
    a = _PilLike(b"\x00\x01", palette=[255, 0, 0, 0, 255, 0])
    b = _PilLike(b"\x00\x01", palette=[0, 0, 255, 255, 255, 0])
    assert default_cache_key(img=a) != default_cache_key(img=b)


def test_pil_no_palette_does_not_crash() -> None:
    # Objects with getpalette returning None (or no getpalette) are handled.
    no_pal = _PilLike(b"\x00", palette=None)
    key = default_cache_key(img=no_pal)
    assert isinstance(key, bytes)


# --------------------------------------------------------------------------- #
# Determinism — identical inputs must yield identical keys.
# --------------------------------------------------------------------------- #


def test_identical_input_identical_key() -> None:
    x1 = _BytesLike(b"\xc8", dtype="uint8", shape=(1,))
    x2 = _BytesLike(b"\xc8", dtype="uint8", shape=(1,))
    assert default_cache_key(arr=x1) == default_cache_key(arr=x2)


def test_pil_identical_input_identical_key() -> None:
    a1 = _PilLike(b"\x00\x01", palette=[1, 2, 3])
    a2 = _PilLike(b"\x00\x01", palette=[1, 2, 3])
    assert default_cache_key(img=a1) == default_cache_key(img=a2)


# --------------------------------------------------------------------------- #
# Plain hashable arguments are unaffected by the tobytes branch.
# --------------------------------------------------------------------------- #


def test_plain_hashable_args_stable_and_hashable() -> None:
    key = default_cache_key(1, "a", x=2)
    assert isinstance(key, bytes)
    # Stable across calls and module re-imports.
    assert key == default_cache_key(1, "a", x=2)


def test_mapping_key_order_invariance() -> None:
    # Already-documented behaviour: {"a":1,"b":2} == {"b":2,"a":1}.
    assert default_cache_key(**{"a": 1, "b": 2}) == default_cache_key(
        **{"b": 2, "a": 1}
    )


# --------------------------------------------------------------------------- #
# Depth guard — deeply nested structures must not blow the stack.
# --------------------------------------------------------------------------- #


def test_freeze_depth_guard() -> None:
    nested: Any = "leaf"
    for _ in range(50):
        nested = {"k": nested}
    # Should return *something* hashable without raising, by hitting depth <= 0.
    result = _freeze(nested, depth=10)
    hash(result)  # raises if not hashable


# --------------------------------------------------------------------------- #
# Fix 2: positional / keyword parity — the same value passed either way
# produces the same cache key.
# --------------------------------------------------------------------------- #


def test_freeze_positional_keyword_parity_for_tobytes_obj() -> None:
    obj = _BytesLike(b"\xc8", dtype="uint8", shape=(1,))
    # Fix 2: _freeze recurses into the wrapping tuple, so the same object
    # freezes identically whether it sits in a positional args tuple or as the
    # value of a keyword mapping. Before the fix the positional path pickled the
    # whole object while the keyword path hit the lossy tobytes branch.
    frozen_in_tuple = _freeze((obj,))[0]
    frozen_in_mapping = _freeze({"arr": obj})[0][1]
    assert frozen_in_tuple == frozen_in_mapping


def test_positional_keyword_parity_for_tuple_of_hashables() -> None:
    # A tuple of plain hashables passed positionally must freeze identically to
    # the same values passed via keyword arguments (Fix 2 consistency).
    assert default_cache_key((1, "a")) != default_cache_key(1, "a")
    # And the keyword path is order-invariant (Mapping sort), which is the part
    # that already worked and must keep working:
    assert default_cache_key(x=1, y="a") == default_cache_key(y="a", x=1)


# --------------------------------------------------------------------------- #
# End-to-end via the functional API, gated on numpy being importable.
# Reproduces the exact scenario from issue #8009.
# --------------------------------------------------------------------------- #


def test_functional_task_cache_does_not_collide_on_dtype() -> None:
    np = pytest.importorskip("numpy")
    from langgraph.cache.memory import InMemoryCache

    from langgraph.func import entrypoint, task
    from langgraph.types import CachePolicy

    runs: list[int] = []

    @task(cache_policy=CachePolicy())
    def total(arr: Any) -> int:
        runs.append(1)
        return int(arr.sum())

    @entrypoint(cache=InMemoryCache())
    def wf(arr: Any) -> int:
        return total(arr=arr).result()

    raw = b"\xc8"
    x = np.frombuffer(raw, dtype=np.uint8)
    y = np.frombuffer(raw, dtype=np.int8)

    assert wf.invoke(x) == 200
    assert wf.invoke(y) == -56  # previously returned 200 (x's cached result)
    assert runs == [1, 1]  # body ran for both inputs
