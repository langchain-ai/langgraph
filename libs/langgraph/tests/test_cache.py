from __future__ import annotations

from typing import Any

import pytest

from langgraph._internal._cache import default_cache_key


class _Array:
    """Minimal tobytes() object with numpy-like metadata."""

    def __init__(
        self,
        data: bytes,
        *,
        dtype: str,
        shape: tuple[int, ...],
        strides: tuple[int, ...] = (1,),
    ) -> None:
        self._data = data
        self.dtype = dtype
        self.shape = shape
        self.strides = strides

    def tobytes(self) -> bytes:
        return self._data


class _Image:
    """Minimal tobytes() object with PIL-like metadata."""

    def __init__(
        self,
        data: bytes,
        *,
        mode: str,
        size: tuple[int, int],
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


def test_tobytes_dtype_is_part_of_key() -> None:
    same_bytes = b"\xc8"
    uint8 = _Array(same_bytes, dtype="uint8", shape=(1,))
    int8 = _Array(same_bytes, dtype="int8", shape=(1,))
    assert default_cache_key(arr=uint8) != default_cache_key(arr=int8)


def test_tobytes_identical_inputs_share_a_key() -> None:
    a = _Array(b"\xc8", dtype="uint8", shape=(1,))
    b = _Array(b"\xc8", dtype="uint8", shape=(1,))
    assert default_cache_key(arr=a) == default_cache_key(arr=b)


def test_positional_tobytes_dtype_is_part_of_key() -> None:
    same_bytes = b"\xc8"
    uint8 = _Array(same_bytes, dtype="uint8", shape=(1,))
    int8 = _Array(same_bytes, dtype="int8", shape=(1,))
    assert default_cache_key(uint8) != default_cache_key(int8)


def test_tobytes_inside_tuple_uses_dtype() -> None:
    same_bytes = b"\xc8"
    uint8 = _Array(same_bytes, dtype="uint8", shape=(1,))
    int8 = _Array(same_bytes, dtype="int8", shape=(1,))
    assert default_cache_key(payload=(uint8,)) != default_cache_key(payload=(int8,))


def test_mapping_key_order_does_not_change_key() -> None:
    assert default_cache_key(payload={"a": 1, "b": 2}) == default_cache_key(
        payload={"b": 2, "a": 1}
    )


def test_pil_palette_is_part_of_key() -> None:
    data = b"\x00\x01"
    size = (1, 2)
    red = _Image(data, mode="P", size=size, palette=[255, 0, 0])
    blue = _Image(data, mode="P", size=size, palette=[0, 0, 255])
    assert default_cache_key(img=red) != default_cache_key(img=blue)


def test_depth_guard_does_not_raise() -> None:
    nested: Any = []
    cur = nested
    for _ in range(20):
        nxt: list[Any] = []
        cur.append(nxt)
        cur = nxt
    default_cache_key(nested)


def test_numpy_dtype_collision_from_issue() -> None:
    np = pytest.importorskip("numpy")
    a = np.array([200], dtype=np.uint8)
    b = np.array([-56], dtype=np.int8)
    assert a.tobytes() == b.tobytes()
    assert default_cache_key(arr=a) != default_cache_key(arr=b)
    assert default_cache_key(a) != default_cache_key(b)
