from typing import Any, Protocol, TypeVar


class SerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects.

    - `dumps`: Serialize an object to bytes.
    - `dumps_typed`: Serialize an object to a tuple (type, bytes).
    - `loads`: Deserialize an object from bytes.
    - `loads_typed`: Deserialize an object from a tuple (type, bytes).

    Valid implementations include the `pickle`, `json` and `orjson` modules.
    """

    def dumps(self, obj: Any) -> bytes:
        ...

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        ...

    def loads(self, data: bytes) -> Any:
        ...

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        ...


T = TypeVar("T", bound=SerializerProtocol)


def maybe_add_typed_methods(serde: T) -> T:
    """Add loads_typed and dumps_typed to old serde implementations for backwards compatibility."""

    if not hasattr(serde, "loads_typed"):

        def loads_typed(data: tuple[str, bytes]) -> Any:
            return serde.loads(data[1])

        serde.loads_typed = loads_typed

    if not hasattr(serde, "dumps_typed"):

        def dumps_typed(obj: Any) -> tuple[str, bytes]:
            return "type", serde.dumps(obj)

        serde.dumps_typed = dumps_typed

    return serde
