"""
:private:
Utilities for langchain-checkpoint-mongod.
"""

from typing import Any, Union

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

serde: SerializerProtocol = JsonPlusSerializer()


def loads_metadata(metadata: dict[str, Any]) -> CheckpointMetadata:
    """Deserialize metadata document

    The CheckpointMetadata class itself cannot be stored directly in MongoDB,
    but as a dictionary it can. For efficient filtering in MongoDB,
    we keep dict keys as strings.

    metadata is stored in MongoDB collection with string keys and
    serde serialized keys.
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = loads_metadata(value)
        return output
    else:
        return serde.loads(metadata)


def dumps_metadata(
    metadata: Union[CheckpointMetadata, Any],
) -> Union[bytes, dict[str, Any]]:
    """Serialize all values in metadata dictionary.

    Keep dict keys as strings for efficient filtering in MongoDB
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = dumps_metadata(value)
        return output
    else:
        return serde.dumps(metadata)
