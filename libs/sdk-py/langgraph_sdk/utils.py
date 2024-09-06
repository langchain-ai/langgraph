import os
from typing import Any, Optional

import langgraph_sdk

RESERVED_HEADERS = ("x-api-key",)


def _get_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Get the API key from the environment.
    Precedence:
        1. explicit argument
        2. LANGGRAPH_API_KEY
        3. LANGSMITH_API_KEY
        4. LANGCHAIN_API_KEY
    """
    if api_key:
        return api_key
    for prefix in ["LANGGRAPH", "LANGSMITH", "LANGCHAIN"]:
        if env := os.getenv(f"{prefix}_API_KEY"):
            return env.strip().strip('"').strip("'")
    return None  # type: ignore


def get_headers(
    api_key: Optional[str], custom_headers: Optional[dict[str, str]]
) -> dict[str, str]:
    """Combine api_key and custom user-provided headers."""
    custom_headers = custom_headers or {}
    for header in RESERVED_HEADERS:
        if header in custom_headers:
            raise ValueError(f"Cannot set reserved header '{header}'")

    headers = {
        "User-Agent": f"langgraph-sdk-py/{langgraph_sdk.__version__}",
        **custom_headers,
    }
    api_key = _get_api_key(api_key)
    if api_key:
        headers["x-api-key"] = api_key

    return headers


def orjson_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    elif hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
