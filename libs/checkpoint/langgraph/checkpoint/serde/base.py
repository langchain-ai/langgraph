from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class UntypedSerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects."""

    def dumps(self, obj: Any) -> bytes:
        """Serialize an object to bytes."""
        ...

    def loads(self, data: bytes) -> Any:
        """Deserialize an object from bytes."""
        ...


@runtime_checkable
class SerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects.

    - `dumps_typed`: Serialize an object to a tuple `(type, bytes)`.
    - `loads_typed`: Deserialize an object from a tuple `(type, bytes)`.

    Valid implementations include the `pickle`, `json` and `orjson` modules.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize an object to a `(type, bytes)` tuple."""
        ...

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Deserialize an object from a `(type, bytes)` tuple."""
        ...


class SerializerCompat(SerializerProtocol):
    """Adapter that wraps an `UntypedSerializerProtocol` to provide `SerializerProtocol`.

    Adds type information by using the object's class name as the type tag.

    Args:
        serde: The untyped serializer to wrap.
    """

    def __init__(self, serde: UntypedSerializerProtocol) -> None:
        self.serde = serde

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize an object, tagging it with its class name.

        Args:
            obj: The object to serialize.

        Returns:
            A tuple of `(class_name, serialized_bytes)`.
        """
        return type(obj).__name__, self.serde.dumps(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Deserialize an object, ignoring the type tag.

        Args:
            data: A `(type, bytes)` tuple as produced by `dumps_typed`.

        Returns:
            The deserialized object.
        """
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
        """Encrypt plaintext. Returns a tuple `(cipher name, ciphertext)`."""
        ...

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        """Decrypt ciphertext. Returns the plaintext."""
        ...
