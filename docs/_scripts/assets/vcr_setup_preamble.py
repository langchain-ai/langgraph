import nest_asyncio

nest_asyncio.apply()

import vcr  # noqa: E402
import msgpack  # noqa: E402
import base64  # noqa: E402
import zlib  # noqa: E402
import os  # noqa: E402

os.environ.pop("LANGCHAIN_TRACING_V2", None)
custom_vcr = vcr.VCR()


def compress_data(data, compression_level=9):
    packed = msgpack.packb(data, use_bin_type=True)
    compressed = zlib.compress(packed, level=compression_level)
    return base64.b64encode(compressed).decode("utf-8")


def decompress_data(compressed_string):
    decoded = base64.b64decode(compressed_string)
    decompressed = zlib.decompress(decoded)
    return msgpack.unpackb(decompressed, raw=False)


class AdvancedCompressedSerializer:
    def serialize(self, cassette_dict):
        return compress_data(cassette_dict)

    def deserialize(self, cassette_string):
        return decompress_data(cassette_string)


custom_vcr.register_serializer("advanced_compressed", AdvancedCompressedSerializer())
custom_vcr.serializer = "advanced_compressed"
