from langgraph_sdk.auth import Auth
from langgraph_sdk.client import get_client, get_sync_client

try:
    from importlib import metadata

    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["Auth", "get_client", "get_sync_client"]
