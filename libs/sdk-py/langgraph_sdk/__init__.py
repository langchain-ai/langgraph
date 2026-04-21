from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph_sdk.auth import Auth
    from langgraph_sdk.client import get_client, get_sync_client
    from langgraph_sdk.encryption import Encryption
    from langgraph_sdk.encryption.types import EncryptionContext

__version__ = "0.3.13"

__all__ = ["Auth", "Encryption", "EncryptionContext", "get_client", "get_sync_client"]

_LAZY: dict[str, str] = {
    "Auth": "langgraph_sdk.auth",
    "get_client": "langgraph_sdk.client",
    "get_sync_client": "langgraph_sdk.client",
    "Encryption": "langgraph_sdk.encryption",
    "EncryptionContext": "langgraph_sdk.encryption.types",
}


def __getattr__(name: str) -> object:
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
