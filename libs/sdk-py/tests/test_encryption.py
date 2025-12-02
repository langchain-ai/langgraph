import pytest

from langgraph_sdk.encryption import DuplicateHandlerError, Encrypt


class TestHandlerValidation:
    """Test duplicate handler and signature validation."""

    def test_duplicate_handlers_raise_error(self):
        """Registering the same handler type twice raises DuplicateHandlerError."""
        encrypt = Encrypt()

        @encrypt.encrypt.blob
        async def blob_enc(ctx, data):
            return data

        @encrypt.decrypt.blob
        async def blob_dec(ctx, data):
            return data

        @encrypt.encrypt.json
        async def json_enc(ctx, data):
            return data

        @encrypt.decrypt.json
        async def json_dec(ctx, data):
            return data

        @encrypt.encrypt.json.thread
        async def thread_enc(ctx, data):
            return data

        @encrypt.decrypt.json("custom")
        async def custom_dec(ctx, data):
            return data

        # All duplicates should raise
        with pytest.raises(DuplicateHandlerError):

            @encrypt.encrypt.blob
            async def dup(ctx, data):
                return data

        with pytest.raises(DuplicateHandlerError):

            @encrypt.decrypt.blob
            async def dup(ctx, data):
                return data

        with pytest.raises(DuplicateHandlerError):

            @encrypt.encrypt.json
            async def dup(ctx, data):
                return data

        with pytest.raises(DuplicateHandlerError):

            @encrypt.decrypt.json
            async def dup(ctx, data):
                return data

        with pytest.raises(DuplicateHandlerError):

            @encrypt.encrypt.json.thread
            async def dup(ctx, data):
                return data

        with pytest.raises(DuplicateHandlerError):

            @encrypt.decrypt.json("custom")
            async def dup(ctx, data):
                return data

    def test_handlers_must_be_async(self):
        """Sync functions raise TypeError."""
        encrypt = Encrypt()

        with pytest.raises(TypeError, match="must be an async function"):

            @encrypt.encrypt.blob
            def sync_handler(ctx, data):
                return data

    def test_handlers_must_have_two_params(self):
        """Wrong parameter count raises TypeError."""
        encrypt = Encrypt()

        with pytest.raises(TypeError, match="must accept exactly 2 parameters"):

            @encrypt.encrypt.blob
            async def wrong_params(ctx):
                return ctx
