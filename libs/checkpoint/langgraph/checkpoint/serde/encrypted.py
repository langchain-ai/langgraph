import os
from typing import Any

from langgraph.checkpoint.serde.base import CipherProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class EncryptedSerializer(JsonPlusSerializer):
    """Serializer that encrypts and decrypts data using an encryption protocol."""

    def __init__(self, cipher: CipherProtocol) -> None:
        super().__init__()
        self.cipher = cipher

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize an object to a tuple (type, bytes) and encrypt the bytes."""
        # serialize data
        typ, data = super().dumps_typed(obj)
        # encrypt data
        encrypted_data = self.cipher.encrypt(data)
        # add cipher name to type
        return f"{typ}+{self.cipher.name}", encrypted_data

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        enc_typ, enc_data = data
        # unencrypted data
        if "+" not in enc_typ:
            return super().loads_typed(data)
        # verify cipher name
        typ, cipher_name = enc_typ.split("+", 1)
        if cipher_name != self.cipher.name:
            raise ValueError(
                f"Cipher mismatch: expected {self.cipher.name}, got {cipher_name}"
            )
        # decrypt data
        decrypted_data = self.cipher.decrypt(enc_data)
        # deserialize data
        return super().loads_typed((typ, decrypted_data))

    @classmethod
    def from_pycryptodome_aes(cls, **kwargs: Any) -> "EncryptedSerializer":
        """Create an EncryptedSerializer using AES encryption."""
        try:
            from Crypto.Cipher import AES  # type: ignore
        except ImportError:
            raise ImportError(
                "Pycryptodome is not installed. Please install it with `pip install pycryptodome`."
            ) from None

        # check if AES key is provided
        if "key" in kwargs:
            key = kwargs.pop("key")
        else:
            key = os.getenvb(b"LANGGRAPH_AES_KEY")
            if key is None:
                raise ValueError("LANGGRAPH_AES_KEY environment variable is not set.")
            if len(key) not in (16, 24, 32):
                raise ValueError("LANGGRAPH_AES_KEY must be 16, 24, or 32 bytes long.")

        # set default mode to EAX if not provided
        if kwargs.get("mode") is None:
            kwargs["mode"] = AES.MODE_EAX

        class PycryptodomeAesCipher(CipherProtocol):
            name: str = "aes"

            def encrypt(self, plaintext: bytes) -> bytes:
                cipher = AES.new(key, **kwargs)
                ciphertext, tag = cipher.encrypt_and_digest(plaintext)
                return cipher.nonce + tag + ciphertext

            def decrypt(self, ciphertext: bytes) -> bytes:
                nonce = ciphertext[:16]
                tag = ciphertext[16:32]
                actual_ciphertext = ciphertext[32:]

                cipher = AES.new(key, **kwargs, nonce=nonce)
                return cipher.decrypt_and_verify(actual_ciphertext, tag)

        return cls(PycryptodomeAesCipher())
