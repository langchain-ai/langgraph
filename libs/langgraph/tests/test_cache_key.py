"""Unit tests for the default cache key function (`langgraph._internal._cache`)."""

from langgraph._internal._cache import _freeze, default_cache_key


class _FakeArray:
    """Minimal numpy-like object: exposes ``tobytes()``, ``shape`` and ``dtype``.

    Like real numpy arrays, instances are unhashable, so ``_freeze`` reaches the
    ``tobytes()`` branch instead of returning the object unchanged.
    """

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, data: bytes, dtype: str, shape: tuple[int, ...]) -> None:
        self._data = data
        self.dtype = dtype
        self.shape = shape

    def tobytes(self) -> bytes:
        return self._data


class _FakeImage:
    """Minimal PIL-like object: same pixel bytes, distinguished by mode/size/palette."""

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, data: bytes, mode: str, size: tuple[int, int], palette: list[int]) -> None:
        self._data = data
        self.mode = mode
        self.size = size
        self._palette = palette

    def tobytes(self) -> bytes:
        return self._data

    def getpalette(self) -> list[int]:
        return self._palette


def test_tobytes_objects_with_different_dtype_get_distinct_keys() -> None:
    # Regression for #8009: identical bytes + shape but a different dtype must
    # not collide to the same cache key.
    x = _FakeArray(b"\xc8", dtype="uint8", shape=(1,))
    y = _FakeArray(b"\xc8", dtype="int8", shape=(1,))
    assert _freeze(x) != _freeze(y)
    assert default_cache_key(arr=x) != default_cache_key(arr=y)


def test_identical_tobytes_objects_get_identical_keys() -> None:
    x = _FakeArray(b"\xc8", dtype="uint8", shape=(1,))
    x2 = _FakeArray(b"\xc8", dtype="uint8", shape=(1,))
    assert default_cache_key(arr=x) == default_cache_key(arr=x2)


def test_image_like_objects_distinguished_by_palette() -> None:
    a = _FakeImage(b"\x00\x01", mode="P", size=(1, 2), palette=[0, 0, 0, 255, 255, 255])
    b = _FakeImage(b"\x00\x01", mode="P", size=(1, 2), palette=[255, 0, 0, 0, 255, 0])
    assert default_cache_key(img=a) != default_cache_key(img=b)


def test_plain_arguments_unaffected() -> None:
    assert default_cache_key(1, "a", flag=True) == default_cache_key(1, "a", flag=True)
    assert default_cache_key(1) != default_cache_key(2)
