"""Tests for EncryptedSerializer with msgpack allowlist functionality.

These tests mirror the msgpack allowlist tests in test_jsonplus.py but run them
through the EncryptedSerializer to ensure the allowlist behavior is preserved
when encryption is enabled.
"""

from __future__ import annotations

import logging
import pathlib
import re
import uuid
from collections import deque
from datetime import date, datetime, time, timezone
from decimal import Decimal
from ipaddress import IPv4Address
from typing import Literal, cast

import ormsgpack
import pytest
from pydantic import BaseModel

from langgraph.checkpoint.base import BaseCheckpointSaver, _with_msgpack_allowlist
from langgraph.checkpoint.serde import _msgpack as _lg_msgpack
from langgraph.checkpoint.serde.base import CipherProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import (
    EXT_METHOD_SINGLE_ARG,
    JsonPlusSerializer,
    _msgpack_enc,
)


class InnerPydantic(BaseModel):
    hello: str


class MyPydantic(BaseModel):
    foo: str
    bar: int
    inner: InnerPydantic


class AnotherPydantic(BaseModel):
    foo: str


class _PassthroughCipher(CipherProtocol):
    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return "passthrough", plaintext

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        assert ciphername == "passthrough"
        return ciphertext


def _make_encrypted_serde(
    allowed_msgpack_modules: (
        _lg_msgpack.AllowedMsgpackModules | Literal[True] | None | object
    ) = _lg_msgpack._SENTINEL,
) -> EncryptedSerializer:
    """Create an EncryptedSerializer with AES encryption for testing."""
    inner = JsonPlusSerializer(
        allowed_msgpack_modules=cast(
            _lg_msgpack.AllowedMsgpackModules | Literal[True] | None,
            allowed_msgpack_modules,
        )
    )
    return EncryptedSerializer.from_pycryptodome_aes(
        serde=inner, key=b"1234567890123456"
    )


def test_msgpack_method_pathlib_blocked_encrypted_strict(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "secret.txt"
    target.write_text("secret")
    payload = ormsgpack.packb(
        ormsgpack.Ext(
            EXT_METHOD_SINGLE_ARG,
            _msgpack_enc(("pathlib", "Path", target, "read_text")),
        ),
        option=ormsgpack.OPT_NON_STR_KEYS,
    )
    serde = EncryptedSerializer(
        _PassthroughCipher(),
        JsonPlusSerializer(allowed_msgpack_modules=None),
    )

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    caplog.clear()
    result = serde.loads_typed(("msgpack+passthrough", payload))

    assert result == target
    assert "blocked deserialization of method call pathlib.path.read_text" in (
        caplog.text.lower()
    )


class TestEncryptedSerializerMsgpackAllowlist:
    """Test msgpack allowlist behavior through EncryptedSerializer."""

    def test_safe_types_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test safe types deserialize without warnings through encryption."""
        serde = _make_encrypted_serde()

        safe_objects = [
            datetime.now(),
            date.today(),
            time(12, 30),
            timezone.utc,
            uuid.uuid4(),
            Decimal("123.45"),
            {1, 2, 3},
            frozenset([1, 2, 3]),
            deque([1, 2, 3]),
            IPv4Address("192.168.1.1"),
            pathlib.Path("/tmp/test"),
        ]

        for obj in safe_objects:
            caplog.clear()
            dumped = serde.dumps_typed(obj)
            # Verify encryption is happening
            assert "+aes" in dumped[0], f"Expected encryption for {type(obj)}"
            result = serde.loads_typed(dumped)
            assert "unregistered type" not in caplog.text.lower(), (
                f"Unexpected warning for {type(obj)}"
            )
            assert result is not None

    def test_pydantic_warns_by_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Pydantic models not in allowlist should log warning but still deserialize."""
        current = _lg_msgpack.STRICT_MSGPACK_ENABLED
        _lg_msgpack.STRICT_MSGPACK_ENABLED = False
        serde = _make_encrypted_serde()

        obj = MyPydantic(foo="test", bar=42, inner=InnerPydantic(hello="world"))

        caplog.clear()
        dumped = serde.dumps_typed(obj)
        assert "+aes" in dumped[0]
        result = serde.loads_typed(dumped)

        assert "unregistered type" in caplog.text.lower()
        assert "allowed_msgpack_modules" in caplog.text
        assert result == obj
        _lg_msgpack.STRICT_MSGPACK_ENABLED = current

    def test_strict_mode_blocks_unregistered(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Strict mode should block unregistered types through encryption."""
        serde = _make_encrypted_serde(allowed_msgpack_modules=None)

        obj = MyPydantic(foo="test", bar=42, inner=InnerPydantic(hello="world"))

        caplog.clear()
        dumped = serde.dumps_typed(obj)
        assert "+aes" in dumped[0]
        result = serde.loads_typed(dumped)

        assert "blocked" in caplog.text.lower()
        expected = obj.model_dump()
        assert result == expected

    def test_allowlist_silences_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Types in allowed_msgpack_modules should deserialize without warnings."""
        serde = _make_encrypted_serde(
            allowed_msgpack_modules=[
                ("tests.test_encrypted", "MyPydantic"),
                ("tests.test_encrypted", "InnerPydantic"),
            ]
        )

        obj = MyPydantic(foo="test", bar=42, inner=InnerPydantic(hello="world"))

        caplog.clear()
        dumped = serde.dumps_typed(obj)
        assert "+aes" in dumped[0]
        result = serde.loads_typed(dumped)

        assert "unregistered type" not in caplog.text.lower()
        assert "blocked" not in caplog.text.lower()
        assert result == obj

    def test_allowlist_blocks_non_listed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Allowlists should block unregistered types even through encryption."""
        serde = _make_encrypted_serde(
            allowed_msgpack_modules=[("tests.test_encrypted", "MyPydantic")]
        )

        obj = AnotherPydantic(foo="nope")

        caplog.clear()
        dumped = serde.dumps_typed(obj)
        assert "+aes" in dumped[0]
        result = serde.loads_typed(dumped)

        assert "blocked" in caplog.text.lower()
        expected = obj.model_dump()
        assert result == expected

    def test_safe_types_value_equality(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify safe types are correctly restored with proper values through encryption."""
        serde = _make_encrypted_serde(allowed_msgpack_modules=None)

        test_cases = [
            datetime(2024, 1, 15, 12, 30, 45, 123456),
            date(2024, 6, 15),
            time(14, 30, 0),
            uuid.UUID("12345678-1234-5678-1234-567812345678"),
            Decimal("123.456789"),
            {1, 2, 3, 4, 5},
            frozenset(["a", "b", "c"]),
            deque([1, 2, 3]),
            IPv4Address("10.0.0.1"),
            pathlib.Path("/some/test/path"),
            re.compile(r"\d+", re.MULTILINE),
        ]

        for obj in test_cases:
            caplog.clear()
            dumped = serde.dumps_typed(obj)
            assert "+aes" in dumped[0], f"Expected encryption for {type(obj)}"
            result = serde.loads_typed(dumped)

            assert "blocked" not in caplog.text.lower(), f"Blocked for {type(obj)}"
            if isinstance(obj, re.Pattern):
                assert result.pattern == obj.pattern
                assert result.flags == obj.flags
            else:
                assert result == obj, (
                    f"Value mismatch for {type(obj)}: {result} != {obj}"
                )

    def test_regex_safe_type(self, caplog: pytest.LogCaptureFixture) -> None:
        """re.compile patterns should deserialize without warnings as a safe type."""
        serde = _make_encrypted_serde(allowed_msgpack_modules=None)
        pattern = re.compile(r"foo.*bar", re.IGNORECASE | re.DOTALL)

        caplog.clear()
        dumped = serde.dumps_typed(pattern)
        assert "+aes" in dumped[0]
        result = serde.loads_typed(dumped)

        assert "blocked" not in caplog.text.lower()
        assert "unregistered" not in caplog.text.lower()
        assert result.pattern == pattern.pattern
        assert result.flags == pattern.flags


class TestWithMsgpackAllowlistEncrypted:
    """Test _with_msgpack_allowlist function with EncryptedSerializer."""

    def test_propagates_allowlist_to_inner_serde(self) -> None:
        """_with_msgpack_allowlist should propagate allowlist to inner JsonPlusSerializer."""
        inner = JsonPlusSerializer(allowed_msgpack_modules=None)
        encrypted = EncryptedSerializer.from_pycryptodome_aes(
            serde=inner, key=b"1234567890123456"
        )

        extra = [("my.module", "MyClass")]
        result = _with_msgpack_allowlist(encrypted, extra)

        # Should return a new EncryptedSerializer
        assert isinstance(result, EncryptedSerializer)
        assert result is not encrypted
        # Inner serde should have the allowlist
        assert isinstance(result.serde, JsonPlusSerializer)
        assert isinstance(result.serde._allowed_msgpack_modules, set)
        assert ("my.module", "MyClass") in result.serde._allowed_msgpack_modules

    def test_preserves_cipher(self) -> None:
        """_with_msgpack_allowlist should preserve the cipher from the original."""
        inner = JsonPlusSerializer(allowed_msgpack_modules=None)
        encrypted = EncryptedSerializer.from_pycryptodome_aes(
            serde=inner, key=b"1234567890123456"
        )

        result = _with_msgpack_allowlist(encrypted, [("my.module", "MyClass")])

        assert isinstance(result, EncryptedSerializer)
        # Should use the same cipher
        assert result.cipher is encrypted.cipher

    def test_returns_same_if_not_jsonplus_inner(self) -> None:
        """_with_msgpack_allowlist should return same serde if inner is not JsonPlusSerializer."""

        class DummyInnerSerde:
            def dumps_typed(self, obj: object) -> tuple[str, bytes]:
                return ("dummy", b"")

            def loads_typed(self, data: tuple[str, bytes]) -> None:
                return None

        from langgraph.checkpoint.serde.base import CipherProtocol

        class DummyCipher(CipherProtocol):
            def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
                return "dummy", plaintext

            def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
                return ciphertext

        encrypted = EncryptedSerializer(DummyCipher(), DummyInnerSerde())
        result = _with_msgpack_allowlist(encrypted, [("my.module", "MyClass")])

        assert result is encrypted

    def test_warns_if_allowlist_unsupported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class DummySerde:
            def dumps_typed(self, obj: object) -> tuple[str, bytes]:
                return ("dummy", b"")

            def loads_typed(self, data: tuple[str, bytes]) -> object:
                return data

        serde = DummySerde()
        caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.base")
        caplog.clear()

        result = _with_msgpack_allowlist(serde, [("my.module", "MyClass")])

        assert result is serde
        assert "does not support msgpack allowlist" in caplog.text.lower()

    def test_noop_allowlist_returns_same_encrypted_instance(self) -> None:
        inner = JsonPlusSerializer(allowed_msgpack_modules=None)
        encrypted = EncryptedSerializer.from_pycryptodome_aes(
            serde=inner, key=b"1234567890123456"
        )

        result = _with_msgpack_allowlist(encrypted, ())

        assert result is encrypted

    def test_functional_roundtrip_with_allowlist(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """End-to-end test: allowlist applied via _with_msgpack_allowlist works."""
        inner = JsonPlusSerializer(allowed_msgpack_modules=None)
        encrypted = EncryptedSerializer.from_pycryptodome_aes(
            serde=inner, key=b"1234567890123456"
        )

        # Apply allowlist for MyPydantic
        updated = _with_msgpack_allowlist(
            encrypted,
            [
                ("tests.test_encrypted", "MyPydantic"),
                ("tests.test_encrypted", "InnerPydantic"),
            ],
        )

        obj = MyPydantic(foo="test", bar=42, inner=InnerPydantic(hello="world"))

        caplog.clear()
        dumped = updated.dumps_typed(obj)
        assert "+aes" in dumped[0]
        result = updated.loads_typed(dumped)

        # Should deserialize without blocking
        assert "blocked" not in caplog.text.lower()
        assert result == obj

    def test_original_still_blocks_after_with_allowlist(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Original serde should still block after _with_msgpack_allowlist creates a new one."""
        inner = JsonPlusSerializer(allowed_msgpack_modules=None)
        encrypted = EncryptedSerializer.from_pycryptodome_aes(
            serde=inner, key=b"1234567890123456"
        )

        # Apply allowlist - this should create a NEW serde
        _with_msgpack_allowlist(
            encrypted,
            [("tests.test_encrypted", "MyPydantic")],
        )

        # Original should still block
        obj = MyPydantic(foo="test", bar=42, inner=InnerPydantic(hello="world"))

        caplog.clear()
        dumped = encrypted.dumps_typed(obj)
        result = encrypted.loads_typed(dumped)

        assert "blocked" in caplog.text.lower()
        assert result == obj.model_dump()


class TestEncryptedSerializerUnencryptedFallback:
    """Test that EncryptedSerializer handles unencrypted data correctly."""

    def test_loads_unencrypted_data(self) -> None:
        """EncryptedSerializer should handle unencrypted data for backwards compat."""
        plain = JsonPlusSerializer(allowed_msgpack_modules=None)
        encrypted = _make_encrypted_serde(allowed_msgpack_modules=None)

        obj = {"key": "value", "number": 42}

        # Serialize with plain serde
        dumped = plain.dumps_typed(obj)
        assert "+aes" not in dumped[0]

        # Should still deserialize with encrypted serde
        result = encrypted.loads_typed(dumped)
        assert result == obj


def test_with_allowlist_uses_copy_protocol() -> None:
    class CopyAwareSaver(BaseCheckpointSaver[str]):
        def __init__(self) -> None:
            super().__init__(serde=JsonPlusSerializer(allowed_msgpack_modules=None))
            self.copy_was_used = False

        def __copy__(self) -> object:
            clone = object.__new__(self.__class__)
            clone.__dict__ = self.__dict__.copy()
            clone.copy_was_used = True
            return clone

    saver = CopyAwareSaver()

    updated = saver.with_allowlist([("tests.test_encrypted", "MyPydantic")])

    assert isinstance(updated, CopyAwareSaver)
    assert updated is not saver
    assert updated.copy_was_used is True
    assert saver.copy_was_used is False
