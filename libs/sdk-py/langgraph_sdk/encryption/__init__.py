"""Custom encryption support for LangGraph.

.. warning::
    This API is in beta and may change in future versions.

This module provides a framework for implementing custom at-rest encryption
in LangGraph applications. Similar to the Auth system, it allows developers
to define custom encryption and decryption handlers that are executed
server-side.
"""

from __future__ import annotations

import functools
import inspect
import typing
import warnings

from langgraph_sdk.encryption import types


class LangGraphBetaWarning(UserWarning):
    """Warning for beta features in LangGraph SDK."""


@functools.lru_cache(maxsize=1)
def _warn_encryption_beta() -> None:
    warnings.warn(
        "The Encryption API is in beta and may change in future versions.",
        LangGraphBetaWarning,
        stacklevel=4,
    )


class DuplicateHandlerError(Exception):
    """Raised when attempting to register a duplicate encryption/decryption handler."""

    pass


def _validate_handler(fn: typing.Callable, handler_type: str) -> None:
    """Validate that a handler function has the correct signature.

    Args:
        fn: The handler function to validate
        handler_type: Description of the handler for error messages

    Raises:
        TypeError: If the handler is not an async function or has wrong parameter count
    """
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"{handler_type} must be an async function, got {type(fn)}")

    sig = inspect.signature(fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(params) != 2:
        raise TypeError(
            f"{handler_type} must accept exactly 2 parameters "
            f"(ctx, data), got {len(params)}"
        )


class _EncryptDecorators:
    """Decorators for encryption handlers.

    Provides @encryption.encrypt.blob and @encryption.encrypt.json decorators for
    registering encryption functions.
    """

    def __init__(self, parent: Encryption):
        self._parent = parent

    def blob(self, fn: types.BlobEncryptor) -> types.BlobEncryptor:
        """Register a blob encryption handler.

        The handler will be called to encrypt opaque data like checkpoint blobs.

        Example:
            ```python
            @encryption.encrypt.blob
            async def encrypt_blob(ctx: EncryptionContext, blob: bytes) -> bytes:
                # Encrypt the blob using your encryption service
                return encrypted_blob
            ```

        Args:
            fn: The encryption handler function

        Returns:
            The registered handler function

        Raises:
            DuplicateHandlerError: If blob encryptor already registered
            TypeError: If handler has invalid signature
        """
        if self._parent._blob_encryptor is not None:
            raise DuplicateHandlerError("Blob encryptor already registered")
        _validate_handler(fn, "Blob encryptor")
        self._parent._blob_encryptor = fn
        return fn

    def json(self, fn: types.JsonEncryptor) -> types.JsonEncryptor:
        """Register the JSON encryption handler.

        Example:
            ```python
            @encryption.encrypt.json
            async def encrypt_json(ctx: EncryptionContext, data: dict) -> dict:
                # Encrypt the data
                return encrypt_data(data)
            ```

        Args:
            fn: The encryption handler function

        Returns:
            The registered handler function

        Raises:
            DuplicateHandlerError: If JSON encryptor already registered
            TypeError: If handler has invalid signature
        """
        if self._parent._json_encryptor is not None:
            raise DuplicateHandlerError("JSON encryptor already registered")
        _validate_handler(fn, "JSON encryptor")
        self._parent._json_encryptor = fn
        return fn


class _DecryptDecorators:
    """Decorators for decryption handlers.

    Provides @encryption.decrypt.blob and @encryption.decrypt.json decorators for
    registering decryption functions.
    """

    def __init__(self, parent: Encryption):
        self._parent = parent

    def blob(self, fn: types.BlobDecryptor) -> types.BlobDecryptor:
        """Register a blob decryption handler.

        The handler will be called to decrypt opaque data like checkpoint blobs.

        Example:
            ```python
            @encryption.decrypt.blob
            async def decrypt_blob(ctx: EncryptionContext, blob: bytes) -> bytes:
                # Decrypt the blob using your encryption service
                return decrypted_blob
            ```

        Args:
            fn: The decryption handler function

        Returns:
            The registered handler function

        Raises:
            DuplicateHandlerError: If blob decryptor already registered
            TypeError: If handler has invalid signature
        """
        if self._parent._blob_decryptor is not None:
            raise DuplicateHandlerError("Blob decryptor already registered")
        _validate_handler(fn, "Blob decryptor")
        self._parent._blob_decryptor = fn
        return fn

    def json(self, fn: types.JsonDecryptor) -> types.JsonDecryptor:
        """Register the JSON decryption handler.

        Example:
            ```python
            @encryption.decrypt.json
            async def decrypt_json(ctx: EncryptionContext, data: dict) -> dict:
                # Decrypt the data
                return decrypt_data(data)
            ```

        Args:
            fn: The decryption handler function

        Returns:
            The registered handler function

        Raises:
            DuplicateHandlerError: If JSON decryptor already registered
            TypeError: If handler has invalid signature
        """
        if self._parent._json_decryptor is not None:
            raise DuplicateHandlerError("JSON decryptor already registered")
        _validate_handler(fn, "JSON decryptor")
        self._parent._json_decryptor = fn
        return fn


class Encryption:
    """Add custom at-rest encryption to your LangGraph application.

    .. warning::
        This API is in beta and may change in future versions.

    The Encryption class provides a system for implementing custom encryption
    of data at rest in LangGraph applications. It supports encryption of
    both opaque blobs (like checkpoints) and structured JSON data (like
    metadata, context, kwargs, values, etc.).

    To use, create a separate Python file and add the path to the file to your
    LangGraph API configuration file (`langgraph.json`). Within that file, create
    an instance of the Encryption class and register encryption and decryption
    handlers as needed.

    Example `langgraph.json` file:

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "agent": "./my_agent/agent.py:graph"
      },
      "env": ".env",
      "encryption": {
        "path": "./encryption.py:my_encryption"
      }
    }
    ```

    Then the LangGraph server will load your encryption file and use it to
    encrypt/decrypt data at rest.

    !!! warning "JSON Encryptors Must Preserve Keys"

        JSON encryptors **must not add or remove keys** from the input dict.
        Only values may be transformed. This constraint is **enforced at runtime
        by the server** and exists because SQL JSONB merge operations (used for
        partial updates) work at the key level.

        **Correct (per-key encryption):**
        ```python
        # Input:  {"secret": "value", "plain": "x"}
        # Output: {"secret": "<encrypted>", "plain": "x"}  ✓ Keys preserved
        ```

        **Incorrect (key consolidation):**
        ```python
        # Input:  {"secret": "value", "plain": "x"}
        # Output: {"__encrypted__": "<blob>", "plain": "x"}  ✗ Key changed
        ```

        If your encryptor needs to store auxiliary data (DEK, IV, etc.), embed it
        within the encrypted value itself, not as separate keys.

    ???+ example "Basic Usage"

        ```python
        from langgraph_sdk import Encryption, EncryptionContext

        my_encryption = Encryption()

        SKIP_FIELDS = {"tenant_id", "owner", "thread_id", "assistant_id"}
        ENCRYPTED_PREFIX = "encrypted:"

        @my_encryption.encrypt.blob
        async def encrypt_blob(ctx: EncryptionContext, blob: bytes) -> bytes:
            return your_encrypt_bytes(blob)

        @my_encryption.decrypt.blob
        async def decrypt_blob(ctx: EncryptionContext, blob: bytes) -> bytes:
            return your_decrypt_bytes(blob)

        @my_encryption.encrypt.json
        async def encrypt_json(ctx: EncryptionContext, data: dict) -> dict:
            result = {}
            for k, v in data.items():
                if k in SKIP_FIELDS or v is None:
                    result[k] = v
                else:
                    result[k] = ENCRYPTED_PREFIX + your_encrypt_string(v)
            return result

        @my_encryption.decrypt.json
        async def decrypt_json(ctx: EncryptionContext, data: dict) -> dict:
            result = {}
            for k, v in data.items():
                if isinstance(v, str) and v.startswith(ENCRYPTED_PREFIX):
                    result[k] = your_decrypt_string(v[len(ENCRYPTED_PREFIX):])
                else:
                    result[k] = v
            return result
        ```

    ???+ example "Field-Specific Logic"

        The `ctx.model` and `ctx.field` attributes tell you which model type and
        specific field is being encrypted, allowing different logic:

        ```python
        @my_encryption.encrypt.json
        async def encrypt_json(ctx: EncryptionContext, data: dict) -> dict:
            if ctx.field == "metadata":
                # Metadata - standard encryption
                return encrypt_standard(data)
            elif ctx.field == "values":
                # Thread values - more sensitive, use stronger encryption
                return encrypt_sensitive(data)
            else:
                return encrypt_standard(data)
        ```

        !!! warning "Model/Field May Differ Between Encrypt and Decrypt"

            Data encrypted with one `(model, field)` pair is **not guaranteed**
            to be decrypted with the same pair. The server performs SQL JSONB
            merges that can move encrypted values between models (e.g., cron
            metadata → run metadata). Your decryption logic must handle data
            regardless of the `ctx.model` or `ctx.field` values at decrypt time.

            **Safe:** Use `ctx.model`/`ctx.field` for logging or metrics only.

            **Safe:** Encrypt different keys based on `ctx.field`, but use a
            single decrypt handler that decrypts any value with the encrypted
            prefix (and passes through plaintext unchanged):

            ```python
            ENCRYPTED_PREFIX = "enc:"

            @my_encryption.encrypt.json
            async def encrypt_json(ctx: EncryptionContext, data: dict) -> dict:
                # Encrypt different keys depending on the field
                if ctx.field == "context":
                    keys_to_encrypt = {"api_key", "secret_token"}
                else:
                    keys_to_encrypt = {"email", "ssn"}
                return {
                    k: ENCRYPTED_PREFIX + encrypt(v) if k in keys_to_encrypt else v
                    for k, v in data.items()
                }

            @my_encryption.decrypt.json
            async def decrypt_json(ctx: EncryptionContext, data: dict) -> dict:
                # Decrypt ANY value with the prefix, regardless of model/field
                return {
                    k: decrypt(v[len(ENCRYPTED_PREFIX):])
                       if isinstance(v, str) and v.startswith(ENCRYPTED_PREFIX)
                       else v
                    for k, v in data.items()
                }
            ```

            **Unsafe:** Using different encryption keys or algorithms based on
            `ctx.model`/`ctx.field` will cause decryption failures.
    """

    __slots__ = (
        "_blob_decryptor",
        "_blob_encryptor",
        "_context_handler",
        "_json_decryptor",
        "_json_encryptor",
        "decrypt",
        "encrypt",
    )

    types = types
    """Reference to encryption type definitions.

    Provides access to all type definitions used in the encryption system,
    including EncryptionContext, BlobEncryptor, BlobDecryptor,
    JsonEncryptor, and JsonDecryptor.
    """

    def __init__(self) -> None:
        """Initialize the Encryption instance."""
        _warn_encryption_beta()
        self.encrypt = _EncryptDecorators(self)
        self.decrypt = _DecryptDecorators(self)
        self._blob_encryptor: types.BlobEncryptor | None = None
        self._blob_decryptor: types.BlobDecryptor | None = None
        self._json_encryptor: types.JsonEncryptor | None = None
        self._json_decryptor: types.JsonDecryptor | None = None
        self._context_handler: types.ContextHandler | None = None

    def context(self, fn: types.ContextHandler) -> types.ContextHandler:
        """Register a context handler to derive encryption context from auth.

        The handler receives the authenticated user and current EncryptionContext,
        and returns a dict that becomes ctx.metadata for encrypt/decrypt handlers.

        This allows encryption context to be derived from JWT claims or other
        auth-derived data instead of requiring a separate X-Encryption-Context header.

        Note: The context handler is called once per request in middleware,
        so ctx.model and ctx.field will be None in the handler.

        Example:
            ```python
            from langgraph_sdk import Encryption, EncryptionContext
            from starlette.authentication import BaseUser

            encryption = Encryption()

            @encryption.context
            async def get_context(user: BaseUser, ctx: EncryptionContext) -> dict:
                # Derive encryption context from authenticated user
                return {
                    **ctx.metadata,  # preserve X-Encryption-Context header if present
                    "tenant_id": user.tenant_id,
                }
            ```

        Args:
            fn: The context handler function

        Returns:
            The registered handler function
        """
        self._context_handler = fn
        return fn

    def get_json_encryptor(
        self,
        _model: str | None = None,  # kept for langgraph-api compat
    ) -> types.JsonEncryptor | None:
        """Get the JSON encryptor.

        Args:
            _model: Ignored. Kept for backwards compatibility with langgraph-api
                which passes model_type to this method.

        Returns:
            The JSON encryptor, or None if not registered.
        """
        return self._json_encryptor

    def get_json_decryptor(
        self,
        _model: str | None = None,  # kept for langgraph-api compat
    ) -> types.JsonDecryptor | None:
        """Get the JSON decryptor.

        Args:
            _model: Ignored. Kept for backwards compatibility with langgraph-api
                which passes model_type to this method.

        Returns:
            The JSON decryptor, or None if not registered.
        """
        return self._json_decryptor

    def __repr__(self) -> str:
        handlers = []
        if self._blob_encryptor:
            handlers.append("blob_encryptor")
        if self._blob_decryptor:
            handlers.append("blob_decryptor")
        if self._json_encryptor:
            handlers.append("json_encryptor")
        if self._json_decryptor:
            handlers.append("json_decryptor")
        if self._context_handler:
            handlers.append("context_handler")
        return f"Encryption(handlers=[{', '.join(handlers)}])"
