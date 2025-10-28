"""Custom JSON serializer for Kusto that returns JSON strings instead of msgpack."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.load import dumps as lc_dumps
from langchain_core.load import loads as lc_loads
from langgraph.checkpoint.serde.base import SerializerProtocol


class JsonStringSerializer(SerializerProtocol):
    """Serializer that returns JSON strings instead of msgpack bytes.
    
    This serializer uses LangChain's JSON serialization which properly handles
    LangChain objects (messages, documents, etc.) and returns JSON strings
    that can be stored directly in Kusto without base64 encoding.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, str]:
        """Serialize an object to a JSON string.
        
        Args:
            obj: The object to serialize.
            
        Returns:
            Tuple of (type_name, json_string) where json_string is a string, not bytes.
        """
        try:
            # Use LangChain's dumps which handles LangChain objects
            json_str = lc_dumps(obj)
            return ("json", json_str)
        except Exception:
            # Fallback to standard JSON for simple types
            json_str = json.dumps(obj, ensure_ascii=False)
            return ("json", json_str)

    def loads_typed(self, data: tuple[str, Any]) -> Any:
        """Deserialize an object from a JSON string.
        
        Args:
            data: Tuple of (type_name, json_string).
            
        Returns:
            The deserialized object.
        """
        type_name, json_data = data
        
        # Only support JSON format
        if type_name != "json":
            raise ValueError(
                f"Unsupported type: {type_name}. "
                f"JsonStringSerializer only supports 'json' type. "
                f"Please clear old data from Kusto or use a different serializer."
            )
        
        # Handle string input (from Kusto)
        if isinstance(json_data, bytes):
            json_str = json_data.decode('utf-8')
        else:
            json_str = json_data
        
        try:
            # Try LangChain's loads first (handles LangChain objects)
            return lc_loads(json_str)
        except Exception:
            # Fallback to standard JSON
            return json.loads(json_str)
