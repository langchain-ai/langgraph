from typing import Any, Callable, Dict, Union

from langgraph.checkpoint.base import CheckpointMetadata

#


def prepare_metadata(
    metadata: Union[CheckpointMetadata, Any], prepare: Callable
) -> Union[bytes, Dict[str, Any]]:
    """Recursively serialize or deserialize all values in metadata dictionary.

    On dumps, one goes from CheckpointMetadata -> Dict[str, bytes]
    On loads, one goes from Dict[str, Any] -> CheckpointMetadata

    Keep dict keys as strings for efficient filtering in MongoDB
    Args:
        metadata:
        prepare:

    Returns:
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = prepare_metadata(value, prepare)
        return output
    else:
        return prepare(metadata)
