from langgraph._internal._cache import default_cache_key


class FakeArray:
    __hash__ = None

    def __init__(self, data: bytes, shape: tuple[int, ...], dtype: str) -> None:
        self.data = data
        self.shape = shape
        self.dtype = dtype

    def __eq__(self, other: object) -> bool:
        return self is other

    def tobytes(self) -> bytes:
        return self.data


class FakeImage:
    __hash__ = None

    def __init__(
        self,
        data: bytes,
        mode: str,
        size: tuple[int, int],
        palette: list[int] | None = None,
    ) -> None:
        self.data = data
        self.mode = mode
        self.size = size
        self.palette = palette

    def __eq__(self, other: object) -> bool:
        return self is other

    def tobytes(self) -> bytes:
        return self.data

    def getpalette(self) -> list[int] | None:
        return self.palette


def test_default_cache_key_distinguishes_array_dtype() -> None:
    uint8 = FakeArray(b"\xc8", (1,), "uint8")
    int8 = FakeArray(b"\xc8", (1,), "int8")

    assert default_cache_key(array=uint8) != default_cache_key(array=int8)


def test_default_cache_key_distinguishes_array_shape() -> None:
    row = FakeArray(b"\x01\x02", (2,), "uint8")
    matrix = FakeArray(b"\x01\x02", (1, 2), "uint8")

    assert default_cache_key(array=row) != default_cache_key(array=matrix)


def test_default_cache_key_is_stable_for_identical_arrays() -> None:
    first = FakeArray(b"\x01\x02", (2,), "uint8")
    second = FakeArray(b"\x01\x02", (2,), "uint8")

    assert default_cache_key(array=first) == default_cache_key(array=second)


def test_default_cache_key_distinguishes_image_mode_and_size() -> None:
    grayscale = FakeImage(b"\x00\x01", "L", (2, 1))
    palette = FakeImage(b"\x00\x01", "P", (2, 1))
    resized = FakeImage(b"\x00\x01", "L", (1, 2))

    assert default_cache_key(image=grayscale) != default_cache_key(image=palette)
    assert default_cache_key(image=grayscale) != default_cache_key(image=resized)


def test_default_cache_key_distinguishes_image_palette() -> None:
    red = FakeImage(b"\x00", "P", (1, 1), [255, 0, 0])
    blue = FakeImage(b"\x00", "P", (1, 1), [0, 0, 255])

    assert default_cache_key(image=red) != default_cache_key(image=blue)


def test_default_cache_key_is_stable_for_identical_images() -> None:
    first = FakeImage(b"\x00", "P", (1, 1), [255, 0, 0])
    second = FakeImage(b"\x00", "P", (1, 1), [255, 0, 0])

    assert default_cache_key(image=first) == default_cache_key(image=second)


def test_default_cache_key_sorts_nested_dict_keys() -> None:
    first = {"a": 1, "b": 2}
    second = {"b": 2, "a": 1}

    assert default_cache_key(value=first) == default_cache_key(value=second)
