import base64
import os
import zlib
from types import TracebackType
from typing import Optional, Any, Type

import msgpack
import vcr

os.environ.pop("LANGCHAIN_TRACING_V2", None)
custom_vcr = vcr.VCR()


def compress_data(data: Any, compression_level: int = 9) -> str:
    packed = msgpack.packb(data, use_bin_type=True)
    compressed = zlib.compress(packed, level=compression_level)
    return base64.b64encode(compressed).decode("utf-8")


def decompress_data(compressed_string: str) -> Any:
    decoded = base64.b64decode(compressed_string)
    decompressed = zlib.decompress(decoded)
    return msgpack.unpackb(decompressed, raw=False)


class AdvancedCompressedSerializer:
    def serialize(self, cassette_dict: Any) -> str:
        return compress_data(cassette_dict)

    def deserialize(self, cassette_string: str) -> Any:
        return decompress_data(cassette_string)


custom_vcr.register_serializer("advanced_compressed", AdvancedCompressedSerializer())
custom_vcr.serializer = "advanced_compressed"


class HashedCassette:
    def __init__(self, cassette_path: str, hash_value: str) -> None:
        """A context manager for using VCR cassettes with an embedded hash value.

        Args:
            cassette_path (str): The file path of the cassette (independent of hash).
            hash_value (str): The expected hash value (e.g. a uuid string).

        This class provides a context manager for using VCR cassettes with an embedded hash value.
        The hash value is used to ensure that the cassette matches the expected state, and if not,
        the cassette is removed or updated with the new hash value.
        """
        self.cassette_path: str = cassette_path
        self.hash_value: str = hash_value
        self.vcr: vcr.VCR = custom_vcr
        self.cassette_context: Optional[Any] = None
        self.exited: bool = False

    def __enter__(self) -> Any:
        self.exited: bool = False
        # Get the serializer instance from the VCR instance.
        serializer = self.vcr.serializers[self.vcr.serializer]
        # If the cassette file exists, check its embedded hash.
        if os.path.exists(self.cassette_path):
            with open(self.cassette_path, "r") as f:
                content = f.read()
            try:
                cassette_data = serializer.deserialize(content)
            except Exception as e:
                os.remove(self.cassette_path)
            else:
                existing_hash = cassette_data.get("cassette_hash")
                if existing_hash != self.hash_value:
                    os.remove(self.cassette_path)
        # Now enter the VCR cassette context.
        self.cassette_context = custom_vcr.use_cassette(
            self.cassette_path,
            filter_headers=["x-api-key", "authorization"],
            record_mode="once",
            serializer="advanced_compressed",
        )
        return self.cassette_context.__enter__()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_val: Optional[BaseException] = None,
        exc_tb: Optional[TracebackType] = None,
    ) -> Optional[bool]:
        if self.exited:
            return
        self.exited = True
        # Exit the VCR cassette context.
        result = self.cassette_context.__exit__(exc_type, exc_val, exc_tb)
        serializer = self.vcr.serializers[self.vcr.serializer]
        # If a cassette was recorded (or updated), open and update its hash.
        if os.path.exists(self.cassette_path):
            with open(self.cassette_path, "r") as f:
                content = f.read()
            try:
                cassette_data = serializer.deserialize(content)
            except Exception as e:
                return result
            # Update the cassette data with the expected hash.
            if cassette_data.get("cassette_hash") != self.hash_value:
                cassette_data["cassette_hash"] = self.hash_value
                serialized_data = serializer.serialize(cassette_data)
                with open(self.cassette_path, "w") as f:
                    f.write(serialized_data)
        return result
