import os
from typing import Any

from langgraph.checkpoint.serde.base import CipherProtocol, SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class EncryptedSerializer(SerializerProtocol):
    """Serializer that encrypts and decrypts data using an encryption protocol."""

    def __init__(
        self,
        cipher: CipherProtocol,
        serde: SerializerProtocol = JsonPlusSerializer(),
        *,
        allow_plaintext: bool = False,
    ) -> None:
        self.cipher = cipher
        self.serde = serde
        # SECURITY: fail-closed by default. When False (the default), any
        # stored row whose type tag lacks a cipher suffix (i.e. it was never
        # encrypted) is rejected instead of silently passed through to the
        # inner deserializer. This prevents an attacker with store-write
        # access from substituting a forged plaintext row for a genuinely
        # encrypted one. Only set this to True if you have a deliberate,
        # understood need to read pre-existing unencrypted data (e.g. a
        # one-time migration), since it reopens that bypass.
        self.allow_plaintext = allow_plaintext

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize an object to a tuple `(type, bytes)` and encrypt the bytes."""
        # serialize data
        typ, data = self.serde.dumps_typed(obj)
        # encrypt data
        ciphername, ciphertext = self.cipher.encrypt(data)
        # add cipher name to type
        return f"{typ}+{ciphername}", ciphertext

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        enc_cipher, ciphertext = data
        # unencrypted data
        if "+" not in enc_cipher:
            if not self.allow_plaintext:
                # Fail closed: never hand unauthenticated, unencrypted bytes
                # to the inner deserializer unless explicitly allowed. Raise
                # before any further processing of `ciphertext` occurs.
                raise ValueError(
                    "Refusing to load unencrypted checkpoint data: the stored "
                    f"type tag {enc_cipher!r} has no cipher suffix, so this "
                    "row was never encrypted (or was tampered with). Set "
                    "allow_plaintext=True on EncryptedSerializer only if you "
                    "intentionally need to read legacy unencrypted data."
                )
            return self.serde.loads_typed(data)
        # extract cipher name
        typ, ciphername = enc_cipher.split("+", 1)
        # decrypt data
        decrypted_data = self.cipher.decrypt(ciphername, ciphertext)
        # deserialize data
        return self.serde.loads_typed((typ, decrypted_data))

    @classmethod
    def from_pycryptodome_aes(
        cls,
        serde: SerializerProtocol = JsonPlusSerializer(),
        *,
        allow_plaintext: bool = False,
        **kwargs: Any,
    ) -> "EncryptedSerializer":
        """Create an `EncryptedSerializer` using AES encryption."""
        try:
            from Crypto.Cipher import AES
        except ImportError:
            raise ImportError(
                "Pycryptodome is not installed. Please install it with `pip install pycryptodome`."
            ) from None

        # check if AES key is provided
        if "key" in kwargs:
            key: bytes = kwargs.pop("key")
        else:
            key_str = os.getenv("LANGGRAPH_AES_KEY")
            if key_str is None:
                raise ValueError("LANGGRAPH_AES_KEY environment variable is not set.")
            key = key_str.encode()
            if len(key) not in (16, 24, 32):
                raise ValueError("LANGGRAPH_AES_KEY must be 16, 24, or 32 bytes long.")

        # set default mode to EAX if not provided
        if kwargs.get("mode") is None:
            kwargs["mode"] = AES.MODE_EAX

        class PycryptodomeAesCipher(CipherProtocol):
            def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
                cipher = AES.new(key, **kwargs)
                ciphertext, tag = cipher.encrypt_and_digest(plaintext)
                return "aes", cipher.nonce + tag + ciphertext

            def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
                assert ciphername == "aes", f"Unsupported cipher: {ciphername}"
                nonce = ciphertext[:16]
                tag = ciphertext[16:32]
                actual_ciphertext = ciphertext[32:]

                cipher = AES.new(key, **kwargs, nonce=nonce)
                return cipher.decrypt_and_verify(actual_ciphertext, tag)

        return cls(PycryptodomeAesCipher(), serde, allow_plaintext=allow_plaintext)
