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


class _JsonEncryptDecorators:
    """Dynamic decorator factory for JSON encryption handlers.

    Supports both default and model-specific handlers:
    - @encrypt.json - default handler for all models
    - @encrypt.json.thread - handler for thread model
    """

    def __init__(self, parent: Encryption):
        self._parent = parent

    def __call__(self, fn: types.JsonEncryptor) -> types.JsonEncryptor:
        """Register the default JSON encryption handler.

        Args:
            fn: The handler function

        Returns:
            The registered handler function

        Raises:
            DuplicateHandlerError: If handler already registered
            TypeError: If handler has invalid signature
        """
        if self._parent._json_encryptor is not None:
            raise DuplicateHandlerError("Default JSON encryptor already registered")
        _validate_handler(fn, "Default JSON encryptor")
        self._parent._json_encryptor = fn
        return fn

    def __getattr__(
        self, model: str
    ) -> typing.Callable[[types.JsonEncryptor], types.JsonEncryptor]:
        """Dynamic attribute access for model-specific handlers.

        Allows @encryption.encrypt.json.thread, @encryption.encrypt.json.assistant, etc.

        Raises:
            DuplicateHandlerError: If handler already registered for this model
            TypeError: If handler has invalid signature
        """

        def decorator(fn: types.JsonEncryptor) -> types.JsonEncryptor:
            if model in self._parent._json_encryptors:
                raise DuplicateHandlerError(
                    f"JSON encryptor for model '{model}' already registered"
                )
            _validate_handler(fn, f"JSON encryptor for model '{model}'")
            self._parent._json_encryptors[model] = fn
            return fn

        return decorator


class _JsonDecryptDecorators:
    """Dynamic decorator factory for JSON decryption handlers.

    Supports both default and model-specific handlers:
    - @encryption.decrypt.json - default handler for all models
    - @encryption.decrypt.json.thread - handler for thread model
    """

    def __init__(self, parent: Encryption):
        self._parent = parent

    def __call__(self, fn: types.JsonDecryptor) -> types.JsonDecryptor:
        """Register the default JSON decryption handler.

        Args:
            fn: The handler function

        Returns:
            The registered handler function

        Raises:
            DuplicateHandlerError: If handler already registered
            TypeError: If handler has invalid signature
        """
        if self._parent._json_decryptor is not None:
            raise DuplicateHandlerError("Default JSON decryptor already registered")
        _validate_handler(fn, "Default JSON decryptor")
        self._parent._json_decryptor = fn
        return fn

    def __getattr__(
        self, model: str
    ) -> typing.Callable[[types.JsonDecryptor], types.JsonDecryptor]:
        """Dynamic attribute access for model-specific handlers.

        Allows @encryption.decrypt.json.thread, @encryption.decrypt.json.assistant, etc.

        Raises:
            DuplicateHandlerError: If handler already registered for this model
            TypeError: If handler has invalid signature
        """

        def decorator(fn: types.JsonDecryptor) -> types.JsonDecryptor:
            if model in self._parent._json_decryptors:
                raise DuplicateHandlerError(
                    f"JSON decryptor for model '{model}' already registered"
                )
            _validate_handler(fn, f"JSON decryptor for model '{model}'")
            self._parent._json_decryptors[model] = fn
            return fn

        return decorator


class _EncryptDecorators:
    """Decorators for encryption handlers.

    Provides @encryption.encrypt.blob and @encryption.encrypt.json decorators for
    registering encryption functions.
    """

    def __init__(self, parent: Encryption):
        self._parent = parent
        self._json = _JsonEncryptDecorators(parent)

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

    @property
    def json(self) -> _JsonEncryptDecorators:
        """Access JSON encryption decorators.

        Supports model-specific handlers:
        - @encryption.encrypt.json - default handler for all models
        - @encryption.encrypt.json.thread - handler for thread model only
        - @encryption.encrypt.json.assistant - handler for assistant model only

        Example:
            ```python
            @encryption.encrypt.json
            async def default_encrypt(ctx: EncryptionContext, data: dict) -> dict:
                # Default encryption for all models
                return encrypt_data(data)

            @encryption.encrypt.json.thread
            async def encrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
                # Special encryption for thread model only
                return encrypt_thread_data(data)
            ```
        """
        return self._json


class _DecryptDecorators:
    """Decorators for decryption handlers.

    Provides @encryption.decrypt.blob and @encryption.decrypt.json decorators for
    registering decryption functions.
    """

    def __init__(self, parent: Encryption):
        self._parent = parent
        self._json = _JsonDecryptDecorators(parent)

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

    @property
    def json(self) -> _JsonDecryptDecorators:
        """Access JSON decryption decorators.

        Supports model-specific handlers:
        - @encryption.decrypt.json - default handler for all models
        - @encryption.decrypt.json.thread - handler for thread model only
        - @encryption.decrypt.json.assistant - handler for assistant model only

        Example:
            ```python
            @encryption.decrypt.json
            async def default_decrypt(ctx: EncryptionContext, data: dict) -> dict:
                # Default decryption for all models
                return decrypt_data(data)

            @encryption.decrypt.json.thread
            async def decrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
                # Special decryption for thread model only
                return decrypt_thread_data(data)
            ```
        """
        return self._json


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

    ???+ example "Basic Usage"

        ```python
        from langgraph_sdk import Encryption, EncryptionContext

        my_encryption = Encryption()

        @my_encryption.encrypt.blob
        async def encrypt_blob(ctx: EncryptionContext, blob: bytes) -> bytes:
            # Call your encryption service
            return encrypted_blob

        @my_encryption.decrypt.blob
        async def decrypt_blob(ctx: EncryptionContext, blob: bytes) -> bytes:
            # Call your decryption service
            return decrypted_blob

        @my_encryption.encrypt.json
        async def encrypt_json(ctx: EncryptionContext, data: dict) -> dict:
            # Practical encryption strategy:
            # - "owner" field: unencrypted (for search/filtering)
            # - "my.customer.org/" prefixed fields: encrypt VALUES only
            # - All other fields: pass through unencrypted
            encrypted = {}
            for key, value in data.items():
                if key.startswith("my.customer.org/"):
                    # Encrypt VALUE for sensitive customer data
                    encrypted[key] = encrypt_value(value)
                else:
                    # Pass through (including "owner" for search)
                    encrypted[key] = value
            return encrypted

        @my_encryption.decrypt.json
        async def decrypt_json(ctx: EncryptionContext, data: dict) -> dict:
            # Decrypt VALUES for "my.customer.org/" prefixed fields
            decrypted = {}
            for key, value in data.items():
                if key.startswith("my.customer.org/"):
                    decrypted[key] = decrypt_value(value)
                else:
                    decrypted[key] = value
            return decrypted
        ```

    ???+ example "Model-Specific Handlers"

        You can register different encryption handlers for different model types
        (thread, assistant, run, cron, checkpoint, etc.):

        ```python
        from langgraph_sdk import Encryption, EncryptionContext

        my_encryption = Encryption()

        # Default handler for models without specific handlers
        @my_encryption.encrypt.json
        async def default_encrypt(ctx: EncryptionContext, data: dict) -> dict:
            return standard_encrypt(data)

        # Thread-specific handler (uses different KMS key)
        @my_encryption.encrypt.json.thread
        async def encrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
            return encrypt_with_thread_key(data)

        # Assistant-specific handler
        @my_encryption.encrypt.json.assistant
        async def encrypt_assistant(ctx: EncryptionContext, data: dict) -> dict:
            return encrypt_with_assistant_key(data)

        # Same pattern for decryption
        @my_encryption.decrypt.json
        async def default_decrypt(ctx: EncryptionContext, data: dict) -> dict:
            return standard_decrypt(data)

        @my_encryption.decrypt.json.thread
        async def decrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
            return decrypt_with_thread_key(data)
        ```

    ???+ example "Field-Specific Logic"

        The `ctx.field` attribute tells you which specific field is being encrypted,
        allowing different logic within the same model:

        ```python
        @my_encryption.encrypt.json.thread
        async def encrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
            if ctx.field == "metadata":
                # Thread metadata - standard encryption
                return encrypt_standard(data)
            elif ctx.field == "values":
                # Thread values - more sensitive, use stronger encryption
                return encrypt_sensitive(data)
            else:
                return encrypt_standard(data)
        ```
    """

    __slots__ = (
        "_blob_decryptor",
        "_blob_encryptor",
        "_context_handler",
        "_json_decryptor",
        "_json_decryptors",
        "_json_encryptor",
        "_json_encryptors",
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
        self._json_encryptors: dict[str, types.JsonEncryptor] = {}
        self._json_decryptors: dict[str, types.JsonDecryptor] = {}
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
        self, model: str | None = None
    ) -> types.JsonEncryptor | None:
        """Get the JSON encryptor for a specific model.

        Args:
            model: The model type (e.g., "thread", "assistant"). If None, returns default.

        Returns:
            Model-specific encryptor if registered, otherwise default encryptor, or None.
        """
        if model and model in self._json_encryptors:
            return self._json_encryptors[model]
        return self._json_encryptor

    def get_json_decryptor(
        self, model: str | None = None
    ) -> types.JsonDecryptor | None:
        """Get the JSON decryptor for a specific model.

        Args:
            model: The model type (e.g., "thread", "assistant"). If None, returns default.

        Returns:
            Model-specific decryptor if registered, otherwise default decryptor, or None.
        """
        if model and model in self._json_decryptors:
            return self._json_decryptors[model]
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
        if self._json_encryptors:
            handlers.append(f"json_encryptors({list(self._json_encryptors.keys())})")
        if self._json_decryptors:
            handlers.append(f"json_decryptors({list(self._json_decryptors.keys())})")
        if self._context_handler:
            handlers.append("context_handler")
        return f"Encryption(handlers=[{', '.join(handlers)}])"
