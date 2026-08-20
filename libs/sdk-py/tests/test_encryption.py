from collections.abc import Awaitable, Callable

import pytest

from langgraph_sdk import DecryptResult, EncryptionContext
from langgraph_sdk.encryption import DuplicateHandlerError, Encryption


def test_decrypt_result():
    result = DecryptResult(plaintext=b"plain", replacement=b"rotated")

    assert result.plaintext == b"plain"
    assert result.replacement == b"rotated"
    assert DecryptResult(plaintext={"plain": True}).replacement is None


def test_decrypt_decorators_preserve_return_types():
    encryption = Encryption()

    @encryption.decrypt.blob
    async def blob_dec(_ctx: EncryptionContext, data: bytes) -> bytes:
        return data

    @encryption.decrypt.json
    async def json_dec(
        _ctx: EncryptionContext, data: dict[str, object]
    ) -> dict[str, object]:
        return data

    blob_handler: Callable[[EncryptionContext, bytes], Awaitable[bytes]] = blob_dec
    json_handler: Callable[
        [EncryptionContext, dict[str, object]], Awaitable[dict[str, object]]
    ] = json_dec
    assert blob_handler is blob_dec
    assert json_handler is json_dec


class TestHandlerValidation:
    """Test duplicate handler and signature validation."""

    def test_duplicate_handlers_raise_error(self):
        """Registering the same handler type twice raises DuplicateHandlerError."""
        encryption = Encryption()

        @encryption.encrypt.blob
        async def blob_enc(_ctx, data):
            return data

        @encryption.decrypt.blob
        async def blob_dec(_ctx, data):
            return data

        @encryption.encrypt.json
        async def json_enc(_ctx, data):
            return data

        @encryption.decrypt.json
        async def json_dec(_ctx, data):
            return data

        # All duplicates should raise
        with pytest.raises(DuplicateHandlerError):

            @encryption.encrypt.blob
            async def dup(_ctx, data):
                return data

        with pytest.raises(DuplicateHandlerError):

            @encryption.decrypt.blob
            async def dup(_ctx, data):
                return data

        with pytest.raises(DuplicateHandlerError):

            @encryption.encrypt.json
            async def dup(_ctx, data):
                return data

        with pytest.raises(DuplicateHandlerError):

            @encryption.decrypt.json
            async def dup(_ctx, data):
                return data

    def test_handlers_must_be_async(self):
        """Sync functions raise TypeError."""
        encryption = Encryption()

        with pytest.raises(TypeError, match="must be an async function"):

            @encryption.encrypt.blob
            def sync_handler(_ctx, data):
                return data

    def test_handlers_must_have_two_params(self):
        """Wrong parameter count raises TypeError."""
        encryption = Encryption()

        with pytest.raises(TypeError, match="must accept exactly 2 parameters"):

            @encryption.encrypt.blob  # ty: ignore[invalid-argument-type]
            async def wrong_params(ctx):
                return ctx
