"""Encryption and decryption types for LangGraph.

This module defines the core types used for custom at-rest encryption
in LangGraph. It includes context types and typed dictionaries for
encryption operations.
"""

from __future__ import annotations

import typing
from collections.abc import Awaitable, Callable

Json = dict[str, typing.Any]
"""JSON-serializable dictionary type for structured data encryption."""


class EncryptionContext:
    """Context passed to encryption/decryption handlers.

    Contains arbitrary non-secret key-values that will be stored on encrypt.
    These key-values are intended to be sent to an external service that
    manages keys and handles the actual encryption and decryption of data.

    Attributes:
        model: The model type being encrypted (e.g., "assistant", "thread", "run", "checkpoint")
        field: The specific field being encrypted (e.g., "metadata", "context", "kwargs", "values")
        metadata: Additional context metadata that can be used for encryption decisions
    """

    __slots__ = ("field", "metadata", "model")

    def __init__(
        self,
        model: str | None = None,
        metadata: dict[str, typing.Any] | None = None,
        field: str | None = None,
    ):
        self.model = model
        self.field = field
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"EncryptionContext(model={self.model!r}, field={self.field!r}, metadata={self.metadata!r})"


BlobEncryptor = Callable[[EncryptionContext, bytes], Awaitable[bytes]]
"""Handler for encrypting opaque blob data like checkpoints.

Note: Must be an async function. Encryption typically involves I/O operations
(calling external KMS services), which should be async.

Args:
    ctx: Encryption context with model type and metadata
    blob: The raw bytes to encrypt

Returns:
    Awaitable that resolves to encrypted bytes
"""

BlobDecryptor = Callable[[EncryptionContext, bytes], Awaitable[bytes]]
"""Handler for decrypting opaque blob data like checkpoints.

Note: Must be an async function. Decryption typically involves I/O operations
(calling external KMS services), which should be async.

Args:
    ctx: Encryption context with model type and metadata
    blob: The encrypted bytes to decrypt

Returns:
    Awaitable that resolves to decrypted bytes
"""

JsonEncryptor = Callable[[EncryptionContext, Json], Awaitable[Json]]
"""Handler for encrypting structured JSON data.

Note: Must be an async function. Encryption typically involves I/O operations
(calling external KMS services), which should be async.

Used for encrypting structured data like metadata, context, kwargs, values,
and other JSON-serializable fields across different model types.

Maps plaintext fields to encrypted fields. A practical approach:
- Keep "owner" field unencrypted for search/filtering
- Encrypt VALUES (not keys) for fields with specific prefix (e.g., "my.customer.org/")
- Pass through all other fields unencrypted

Example:
    Input:  {"owner": "user123", "my.customer.org/email": "john@example.com", "tenant_id": "t-456"}
    Output: {"owner": "user123", "my.customer.org/email": "ENCRYPTED", "tenant_id": "t-456"}

Note: Encrypted field VALUES cannot be reliably searched, as most real-world
encryption implementations use nonces (non-deterministic encryption).
Only unencrypted fields can be used in search queries.

Args:
    ctx: Encryption context with model type, field name, and metadata
    data: The plaintext JSON dictionary

Returns:
    Awaitable that resolves to encrypted JSON dictionary
"""

JsonDecryptor = Callable[[EncryptionContext, Json], Awaitable[Json]]
"""Handler for decrypting structured JSON data.

Note: Must be an async function. Decryption typically involves I/O operations
(calling external KMS services), which should be async.

Inverse of JsonEncryptor. Must be able to decrypt data that
was encrypted by the corresponding encryptor.

Args:
    ctx: Encryption context with model type, field name, and metadata
    data: The encrypted JSON dictionary

Returns:
    Awaitable that resolves to decrypted JSON dictionary
"""

if typing.TYPE_CHECKING:
    from starlette.authentication import BaseUser

ContextHandler = Callable[
    ["BaseUser", EncryptionContext], Awaitable[dict[str, typing.Any]]
]
"""Handler for deriving encryption context from authenticated user info.

Note: Must be an async function as it may involve I/O operations.

The context handler is called once per request in middleware (after auth),
allowing encryption context to be derived from JWT claims, user properties,
or other auth-derived data instead of requiring a separate X-Encryption-Context header.

The return value becomes ctx.metadata for subsequent encrypt/decrypt operations
and is persisted with encrypted data for later decryption.

Note: ctx.model and ctx.field will be None in context handlers since
the handler runs once per request before any specific model/field is known.

Args:
    user: The authenticated user (from Starlette's AuthenticationMiddleware)
    ctx: Current encryption context with metadata from X-Encryption-Context header

Returns:
    Awaitable that resolves to dict that becomes the new ctx.metadata
"""
