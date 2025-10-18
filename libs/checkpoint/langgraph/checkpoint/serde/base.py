from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class UntypedSerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects."""

    def dumps(self, obj: Any) -> bytes: ...

    def loads(self, data: bytes) -> Any: ...


@runtime_checkable
class SerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects.

    - `dumps`: Serialize an object to bytes.
    - `dumps_typed`: Serialize an object to a tuple `(type, bytes)`.
    - `loads`: Deserialize an object from bytes.
    - `loads_typed`: Deserialize an object from a tuple `(type, bytes)`.

    Valid implementations include the `pickle`, `json` and `orjson` modules.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...

    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...


class SerializerCompat(SerializerProtocol):
    def __init__(self, serde: UntypedSerializerProtocol) -> None:
        self.serde = serde

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return type(obj).__name__, self.serde.dumps(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return self.serde.loads(data[1])


def maybe_add_typed_methods(
    serde: SerializerProtocol | UntypedSerializerProtocol,
) -> SerializerProtocol:
    """Wrap serde old serde implementations in a class with loads_typed and dumps_typed for backwards compatibility."""

    if not isinstance(serde, SerializerProtocol):
        return SerializerCompat(serde)

    return serde


class CipherProtocol(Protocol):
    """Protocol for encryption and decryption of data.
    - `encrypt`: Encrypt plaintext.
    - `decrypt`: Decrypt ciphertext.
    """

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        """Encrypt plaintext. Returns a tuple (cipher name, ciphertext)."""
        ...

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        """Decrypt ciphertext. Returns the plaintext."""
        ...
