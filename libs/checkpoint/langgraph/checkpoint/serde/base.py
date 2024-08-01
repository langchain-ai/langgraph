from typing import Any, Protocol


class SerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects.

    - `dumps`: Serialize an object to bytes.
    - `dumps_typed`: Serialize an object to a typle (type, bytes).
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
