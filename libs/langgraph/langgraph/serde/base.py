from typing import Any, Protocol


class SerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects.

    - `dumps`: Serialize an object to bytes.
    - `loads`: Deserialize an object from bytes.

    Valid implementations include the `pickle`, `json` and `orjson` modules.
    """

    def dumps(self, obj: Any) -> bytes:
        ...

    def loads(self, data: bytes) -> Any:
        ...
