"""Encryption and decryption types for LangGraph.

This module defines the core types used for custom at-rest encryption
in LangGraph. It includes context types and typed dictionaries for
encryption operations.
"""

from __future__ import annotations

import typing
from collections.abc import Awaitable, Callable

Metadata = dict[str, typing.Any]
"""Metadata dictionary type for assistant/thread/run/store metadata."""


class EncryptionContext:
    """Context passed to encryption/decryption handlers.

    Contains arbitrary non-secret key-values that will be stored on encrypt.
    These key-values are intended to be sent to an external service that
    manages keys and handles the actual encryption and decryption of data.

    Attributes:
        model: The model type being encrypted (e.g., "assistant", "thread", "run", "checkpoint")
        metadata: Additional context metadata that can be used for encryption decisions
    """

    __slots__ = ("model", "metadata")

    def __init__(
        self,
        model: str | None = None,
        metadata: dict[str, typing.Any] | None = None,
    ):
        self.model = model
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"EncryptionContext(model={self.model!r}, metadata={self.metadata!r})"


BlobEncryptor = Callable[[EncryptionContext, bytes], bytes | Awaitable[bytes]]
"""Handler for encrypting opaque blob data like checkpoints.

Args:
    ctx: Encryption context with model type and metadata
    blob: The raw bytes to encrypt

Returns:
    Encrypted bytes
"""

BlobDecryptor = Callable[[EncryptionContext, bytes], bytes | Awaitable[bytes]]
"""Handler for decrypting opaque blob data like checkpoints.

Args:
    ctx: Encryption context with model type and metadata
    blob: The encrypted bytes to decrypt

Returns:
    Decrypted bytes
"""

MetadataEncryptor = Callable[
    [EncryptionContext, Metadata], Metadata | Awaitable[Metadata]
]
"""Handler for encrypting metadata key/value pairs.

Maps plaintext metadata fields to encrypted fields. A practical approach:
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
    ctx: Encryption context with model type and metadata
    metadata: The plaintext metadata dictionary

Returns:
    Encrypted metadata dictionary
"""

MetadataDecryptor = Callable[
    [EncryptionContext, Metadata], Metadata | Awaitable[Metadata]
]
"""Handler for decrypting metadata key/value pairs.

Inverse of MetadataEncryptor. Must be able to decrypt metadata that
was encrypted by the corresponding encryptor.

Args:
    ctx: Encryption context with model type and metadata
    metadata: The encrypted metadata dictionary

Returns:
    Decrypted metadata dictionary
"""
