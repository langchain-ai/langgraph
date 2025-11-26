"""Custom encryption support for LangGraph.

This module provides a framework for implementing custom at-rest encryption
in LangGraph applications. Similar to the Auth system, it allows developers
to define custom encryption and decryption handlers that are executed
server-side.
"""

from __future__ import annotations

import typing
import warnings

from langgraph_sdk.encryption import types


class _JsonEncryptDecorators:
    """Dynamic decorator factory for JSON encryption handlers.

    Supports both default and model-specific handlers:
    - @encrypt.json - default handler for all models
    - @encrypt.json.thread - handler for thread model
    - @encrypt.json("custom") - handler for custom model name
    """

    def __init__(self, parent: Encrypt):
        self._parent = parent

    def __call__(
        self, fn_or_model: types.JsonEncryptor | str
    ) -> (
        types.JsonEncryptor
        | typing.Callable[[types.JsonEncryptor], types.JsonEncryptor]
    ):
        """Register a JSON encryption handler.

        Can be used as:
        - @encrypt.json - register default handler
        - @encrypt.json("model") - register model-specific handler

        Args:
            fn_or_model: Either the handler function (default) or model name string

        Returns:
            The registered handler function, or a decorator if model name provided
        """
        if isinstance(fn_or_model, str):
            model_name = fn_or_model

            def decorator(fn: types.JsonEncryptor) -> types.JsonEncryptor:
                self._parent._json_encryptors[model_name] = fn
                return fn

            return decorator
        else:
            fn = fn_or_model
            self._parent._json_encryptor = fn
            return fn

    def __getattr__(
        self, model: str
    ) -> typing.Callable[[types.JsonEncryptor], types.JsonEncryptor]:
        """Dynamic attribute access for model-specific handlers.

        Allows @encrypt.json.thread, @encrypt.json.assistant, etc.
        """

        def decorator(fn: types.JsonEncryptor) -> types.JsonEncryptor:
            self._parent._json_encryptors[model] = fn
            return fn

        return decorator


class _JsonDecryptDecorators:
    """Dynamic decorator factory for JSON decryption handlers.

    Supports both default and model-specific handlers:
    - @decrypt.json - default handler for all models
    - @decrypt.json.thread - handler for thread model
    - @decrypt.json("custom") - handler for custom model name
    """

    def __init__(self, parent: Encrypt):
        self._parent = parent

    def __call__(
        self, fn_or_model: types.JsonDecryptor | str
    ) -> (
        types.JsonDecryptor
        | typing.Callable[[types.JsonDecryptor], types.JsonDecryptor]
    ):
        """Register a JSON decryption handler.

        Can be used as:
        - @decrypt.json - register default handler
        - @decrypt.json("model") - register model-specific handler

        Args:
            fn_or_model: Either the handler function (default) or model name string

        Returns:
            The registered handler function, or a decorator if model name provided
        """
        if isinstance(fn_or_model, str):
            model_name = fn_or_model

            def decorator(fn: types.JsonDecryptor) -> types.JsonDecryptor:
                self._parent._json_decryptors[model_name] = fn
                return fn

            return decorator
        else:
            fn = fn_or_model
            self._parent._json_decryptor = fn
            return fn

    def __getattr__(
        self, model: str
    ) -> typing.Callable[[types.JsonDecryptor], types.JsonDecryptor]:
        """Dynamic attribute access for model-specific handlers.

        Allows @decrypt.json.thread, @decrypt.json.assistant, etc.
        """

        def decorator(fn: types.JsonDecryptor) -> types.JsonDecryptor:
            self._parent._json_decryptors[model] = fn
            return fn

        return decorator


class _EncryptDecorators:
    """Decorators for encryption handlers.

    Provides @encrypt.blob and @encrypt.json decorators for
    registering encryption functions.
    """

    def __init__(self, parent: Encrypt):
        self._parent = parent
        self._json = _JsonEncryptDecorators(parent)

    def blob(self, fn: types.BlobEncryptor) -> types.BlobEncryptor:
        """Register a blob encryption handler.

        The handler will be called to encrypt opaque data like checkpoint blobs.

        Example:
            ```python
            @encrypt.blob
            async def encrypt_checkpoint(ctx: EncryptionContext, blob: bytes) -> bytes:
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

    @property
    def json(self) -> _JsonEncryptDecorators:
        """Access JSON encryption decorators.

        Supports model-specific handlers:
        - @encrypt.json - default handler for all models
        - @encrypt.json.thread - handler for thread model only
        - @encrypt.json.assistant - handler for assistant model only
        - @encrypt.json("custom") - handler for custom model name

        Example:
            ```python
            @encrypt.json
            async def default_encrypt(ctx: EncryptionContext, data: dict) -> dict:
                # Default encryption for all models
                return encrypt_data(data)

            @encrypt.json.thread
            async def encrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
                # Special encryption for thread model only
                return encrypt_thread_data(data)
            ```
        """
        return self._json

    @property
    def metadata(self) -> _JsonEncryptDecorators:
        """Deprecated: Use @encrypt.json instead."""
        warnings.warn(
            "@encrypt.metadata is deprecated, use @encrypt.json instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._json


class _DecryptDecorators:
    """Decorators for decryption handlers.

    Provides @decrypt.blob and @decrypt.json decorators for
    registering decryption functions.
    """

    def __init__(self, parent: Encrypt):
        self._parent = parent
        self._json = _JsonDecryptDecorators(parent)

    def blob(self, fn: types.BlobDecryptor) -> types.BlobDecryptor:
        """Register a blob decryption handler.

        The handler will be called to decrypt opaque data like checkpoint blobs.

        Example:
            ```python
            @decrypt.blob
            async def decrypt_checkpoint(ctx: EncryptionContext, blob: bytes) -> bytes:
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

    @property
    def json(self) -> _JsonDecryptDecorators:
        """Access JSON decryption decorators.

        Supports model-specific handlers:
        - @decrypt.json - default handler for all models
        - @decrypt.json.thread - handler for thread model only
        - @decrypt.json.assistant - handler for assistant model only
        - @decrypt.json("custom") - handler for custom model name

        Example:
            ```python
            @decrypt.json
            async def default_decrypt(ctx: EncryptionContext, data: dict) -> dict:
                # Default decryption for all models
                return decrypt_data(data)

            @decrypt.json.thread
            async def decrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
                # Special decryption for thread model only
                return decrypt_thread_data(data)
            ```
        """
        return self._json

    @property
    def metadata(self) -> _JsonDecryptDecorators:
        """Deprecated: Use @decrypt.json instead."""
        warnings.warn(
            "@decrypt.metadata is deprecated, use @decrypt.json instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._json


class Encrypt:
    """Add custom at-rest encryption to your LangGraph application.

    The Encrypt class provides a system for implementing custom encryption
    of data at rest in LangGraph applications. It supports encryption of
    both opaque blobs (like checkpoints) and structured JSON data (like
    metadata, context, kwargs, values, etc.).

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
        async def encrypt_checkpoint(ctx: EncryptionContext, blob: bytes) -> bytes:
            # Call your encryption service
            return encrypted_blob

        @my_encrypt.decrypt.blob
        async def decrypt_checkpoint(ctx: EncryptionContext, blob: bytes) -> bytes:
            # Call your decryption service
            return decrypted_blob

        @my_encrypt.encrypt.json
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

        @my_encrypt.decrypt.json
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
        from langgraph_sdk import Encrypt, EncryptionContext

        my_encrypt = Encrypt()

        # Default handler for models without specific handlers
        @my_encrypt.encrypt.json
        async def default_encrypt(ctx: EncryptionContext, data: dict) -> dict:
            return standard_encrypt(data)

        # Thread-specific handler (uses different KMS key)
        @my_encrypt.encrypt.json.thread
        async def encrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
            return encrypt_with_thread_key(data)

        # Assistant-specific handler
        @my_encrypt.encrypt.json.assistant
        async def encrypt_assistant(ctx: EncryptionContext, data: dict) -> dict:
            return encrypt_with_assistant_key(data)

        # Same pattern for decryption
        @my_encrypt.decrypt.json
        async def default_decrypt(ctx: EncryptionContext, data: dict) -> dict:
            return standard_decrypt(data)

        @my_encrypt.decrypt.json.thread
        async def decrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
            return decrypt_with_thread_key(data)
        ```

    ???+ example "Field-Specific Logic"

        The `ctx.field` attribute tells you which specific field is being encrypted,
        allowing different logic within the same model:

        ```python
        @my_encrypt.encrypt.json.thread
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
        """Initialize the Encrypt instance."""
        self.encrypt = _EncryptDecorators(self)
        self.decrypt = _DecryptDecorators(self)
        self._blob_encryptor: types.BlobEncryptor | None = None
        self._blob_decryptor: types.BlobDecryptor | None = None
        self._json_encryptor: types.JsonEncryptor | None = None
        self._json_decryptor: types.JsonDecryptor | None = None
        self._json_encryptors: dict[str, types.JsonEncryptor] = {}
        self._json_decryptors: dict[str, types.JsonDecryptor] = {}

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

    @property
    def _metadata_encryptor(self) -> types.JsonEncryptor | None:
        """Deprecated: Use get_json_encryptor() instead."""
        return self._json_encryptor

    @_metadata_encryptor.setter
    def _metadata_encryptor(self, value: types.JsonEncryptor | None) -> None:
        """Deprecated: Use @encrypt.json instead."""
        self._json_encryptor = value

    @property
    def _metadata_decryptor(self) -> types.JsonDecryptor | None:
        """Deprecated: Use get_json_decryptor() instead."""
        return self._json_decryptor

    @_metadata_decryptor.setter
    def _metadata_decryptor(self, value: types.JsonDecryptor | None) -> None:
        """Deprecated: Use @decrypt.json instead."""
        self._json_decryptor = value

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
        return f"Encrypt(handlers=[{', '.join(handlers)}])"
