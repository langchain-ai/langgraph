"""Custom encryption support for LangGraph.

This module provides a framework for implementing custom at-rest encryption
in LangGraph applications. Similar to the Auth system, it allows developers
to define custom encryption and decryption handlers that are executed
server-side.
"""

from __future__ import annotations

import inspect
import typing

from langgraph_sdk.encryption import types

EH = typing.TypeVar("EH", bound=types.BlobEncryptor | types.MetadataEncryptor)
DH = typing.TypeVar("DH", bound=types.BlobDecryptor | types.MetadataDecryptor)


class _EncryptDecorators:
    """Decorators for encryption handlers.

    Provides @encrypt.blob and @encrypt.metadata decorators for
    registering encryption functions.
    """

    def __init__(self, parent: Encrypt):
        self._parent = parent

    def blob(self, fn: types.BlobEncryptor) -> types.BlobEncryptor:
        """Register a blob encryption handler.

        The handler will be called to encrypt opaque data like checkpoint blobs.

        Example:
            ```python
            @encrypt.blob
            def encrypt_checkpoint(ctx: EncryptionContext, blob: bytes) -> bytes:
                # Encrypt the blob using your encryption service
                return encrypted_blob
            ```

        Args:
            fn: The encryption handler function

        Returns:
            The registered handler function
        """
        self._parent._blob_encryptor = fn
        return fn

    def metadata(self, fn: types.MetadataEncryptor) -> types.MetadataEncryptor:
        """Register a metadata encryption handler.

        The handler will be called to encrypt metadata key/value pairs.

        Example:
            ```python
            @encrypt.metadata
            def encrypt_metadata(ctx: EncryptionContext, metadata: dict) -> dict:
                # Encrypt metadata fields
                return encrypted_metadata
            ```

        Args:
            fn: The encryption handler function

        Returns:
            The registered handler function
        """
        self._parent._metadata_encryptor = fn
        return fn


class _DecryptDecorators:
    """Decorators for decryption handlers.

    Provides @decrypt.blob and @decrypt.metadata decorators for
    registering decryption functions.
    """

    def __init__(self, parent: Encrypt):
        self._parent = parent

    def blob(self, fn: types.BlobDecryptor) -> types.BlobDecryptor:
        """Register a blob decryption handler.

        The handler will be called to decrypt opaque data like checkpoint blobs.

        Example:
            ```python
            @decrypt.blob
            def decrypt_checkpoint(ctx: EncryptionContext, blob: bytes) -> bytes:
                # Decrypt the blob using your encryption service
                return decrypted_blob
            ```

        Args:
            fn: The decryption handler function

        Returns:
            The registered handler function
        """
        self._parent._blob_decryptor = fn
        return fn

    def metadata(self, fn: types.MetadataDecryptor) -> types.MetadataDecryptor:
        """Register a metadata decryption handler.

        The handler will be called to decrypt metadata key/value pairs.

        Example:
            ```python
            @decrypt.metadata
            def decrypt_metadata(ctx: EncryptionContext, metadata: dict) -> dict:
                # Decrypt metadata fields
                return decrypted_metadata
            ```

        Args:
            fn: The decryption handler function

        Returns:
            The registered handler function
        """
        self._parent._metadata_decryptor = fn
        return fn


class Encrypt:
    """Add custom at-rest encryption to your LangGraph application.

    The Encrypt class provides a system for implementing custom encryption
    of data at rest in LangGraph applications. It supports encryption of
    both opaque blobs (like checkpoints) and structured metadata.

    To use, create a separate Python file and add the path to the file to your
    LangGraph API configuration file (`langgraph.json`). Within that file, create
    an instance of the Encrypt class and register encryption and decryption
    handlers as needed.

    Example `langgraph.json` file:

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "agent": "./my_agent/agent.py:graph"
      },
      "env": ".env",
      "encrypt": {
        "path": "./encrypt.py:my_encrypt"
      }
    }
    ```

    Then the LangGraph server will load your encryption file and use it to
    encrypt/decrypt data at rest.

    ???+ example "Basic Usage"

        ```python
        from langgraph_sdk import Encrypt, EncryptionContext

        my_encrypt = Encrypt()

        @my_encrypt.encrypt.blob
        def encrypt_checkpoint(ctx: EncryptionContext, blob: bytes) -> bytes:
            # Call your encryption service
            return encrypted_blob

        @my_encrypt.decrypt.blob
        def decrypt_checkpoint(ctx: EncryptionContext, blob: bytes) -> bytes:
            # Call your decryption service
            return decrypted_blob

        @my_encrypt.encrypt.metadata
        def encrypt_metadata(ctx: EncryptionContext, metadata: dict) -> dict:
            # Practical encryption strategy:
            # - "owner" field: unencrypted (for search/filtering)
            # - "my.customer.org/" prefixed fields: encrypt VALUES only
            # - All other fields: pass through unencrypted
            encrypted = {}
            for key, value in metadata.items():
                if key.startswith("my.customer.org/"):
                    # Encrypt VALUE for sensitive customer data
                    encrypted[key] = encrypt_value(value)
                else:
                    # Pass through (including "owner" for search)
                    encrypted[key] = value
            return encrypted

        @my_encrypt.decrypt.metadata
        def decrypt_metadata(ctx: EncryptionContext, metadata: dict) -> dict:
            # Decrypt VALUES for "my.customer.org/" prefixed fields
            decrypted = {}
            for key, value in metadata.items():
                if key.startswith("my.customer.org/"):
                    decrypted[key] = decrypt_value(value)
                else:
                    decrypted[key] = value
            return decrypted
        ```
    """

    __slots__ = (
        "encrypt",
        "decrypt",
        "_blob_encryptor",
        "_blob_decryptor",
        "_metadata_encryptor",
        "_metadata_decryptor",
    )

    types = types
    """Reference to encryption type definitions.

    Provides access to all type definitions used in the encryption system,
    including EncryptionContext, BlobEncryptor, BlobDecryptor,
    MetadataEncryptor, and MetadataDecryptor.
    """

    def __init__(self) -> None:
        """Initialize the Encrypt instance."""
        self.encrypt = _EncryptDecorators(self)
        self.decrypt = _DecryptDecorators(self)
        self._blob_encryptor: types.BlobEncryptor | None = None
        self._blob_decryptor: types.BlobDecryptor | None = None
        self._metadata_encryptor: types.MetadataEncryptor | None = None
        self._metadata_decryptor: types.MetadataDecryptor | None = None

    def __repr__(self) -> str:
        handlers = []
        if self._blob_encryptor:
            handlers.append("blob_encryptor")
        if self._blob_decryptor:
            handlers.append("blob_decryptor")
        if self._metadata_encryptor:
            handlers.append("metadata_encryptor")
        if self._metadata_decryptor:
            handlers.append("metadata_decryptor")
        return f"Encrypt(handlers=[{', '.join(handlers)}])"
