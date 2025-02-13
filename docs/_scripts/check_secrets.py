import base64
import json
import os
import re
import sys
import zlib
from typing import Any

import msgpack


def compress_data(data: Any, compression_level: int = 9) -> str:
    packed = msgpack.packb(data, use_bin_type=True)
    compressed = zlib.compress(packed, level=compression_level)
    return base64.b64encode(compressed).decode("utf-8")


def decompress_data(compressed_string: str) -> Any:
    decoded = base64.b64decode(compressed_string)
    decompressed = zlib.decompress(decoded)
    return msgpack.unpackb(decompressed, raw=False)


if __name__ == "__main__":
    for file in os.listdir("cassettes"):
        if file.endswith(".msgpack.zlib"):
            with open(
                f"cassettes/{file}",
                "r",
            ) as f:
                data = f.read()

        decompressed = decompress_data(data)
        decompressed = json.dumps(decompressed, default=str)
        if "x-api-key" in decompressed.lower():
            print(f"Found secret (x-api-key) in {file}!", file=sys.stderr)
            print(decompressed, file=sys.stderr)
        if "Bearer: " in decompressed.lower():
            print(f"Found potential secret (bearer token) in {file}!", file=sys.stderr)
            print(decompressed, file=sys.stderr)
        if match := re.match(r"sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}", decompressed.lower()):
            print(f"OpenAI user API key found in {file}!", file=sys.stderr)
            print(match.group(0), file=sys.stderr)
        if match := re.match(r"sk-proj-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}", decompressed.lower()):
            print(f"OpenAI user project API key found in {file}!", file=sys.stderr)
            print(match.group(0), file=sys.stderr)
        if match := re.match(r"sk-proj-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}", decompressed.lower()):
            print(f"OpenAI user project API key found in {file}!", file=sys.stderr)
            print(match.group(0), file=sys.stderr)
        for service_id in re.compile(r"^[A-Za-z0-9]+(-*[A-Za-z0-9]+)*$").finditer(decompressed):
            if match := re.match(f"sk-{service_id.group(0)}-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}", decompressed.lower()):
                print(f"OpenAI service key found in {file}!", file=sys.stderr)
                print(match.group(0), file=sys.stderr)
        if match := re.match(r"sk-ant-[A-Za-z0-9_-]{101}", decompressed.lower()):
            print(f"Anthropic API key found in {file}!", file=sys.stderr)
            print(match.group(0), file=sys.stderr)
