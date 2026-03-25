import os
from collections.abc import Iterable
from typing import cast

STRICT_MSGPACK_ENABLED = os.getenv("LANGGRAPH_STRICT_MSGPACK", "false").lower() in (
    "1",
    "true",
    "yes",
)


_SENTINEL = cast(None, object())

SAFE_MSGPACK_TYPES: frozenset[tuple[str, ...]] = frozenset(
    {
        # datetime types
        ("datetime", "datetime"),
        ("datetime", "date"),
        ("datetime", "time"),
        ("datetime", "timedelta"),
        ("datetime", "timezone"),
        # uuid
        ("uuid", "UUID"),
        # numeric
        ("decimal", "Decimal"),
        # collections
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("collections", "deque"),
        # ip addresses
        ("ipaddress", "IPv4Address"),
        ("ipaddress", "IPv4Interface"),
        ("ipaddress", "IPv4Network"),
        ("ipaddress", "IPv6Address"),
        ("ipaddress", "IPv6Interface"),
        ("ipaddress", "IPv6Network"),
        # pathlib
        ("pathlib", "Path"),
        ("pathlib", "PosixPath"),
        ("pathlib", "WindowsPath"),
        # pathlib in Python 3.13+
        ("pathlib._local", "Path"),
        ("pathlib._local", "PosixPath"),
        ("pathlib._local", "WindowsPath"),
        # zoneinfo
        ("zoneinfo", "ZoneInfo"),
        # regex
        ("re", "compile"),
        # langchain-core messages (safe container types used by graph state)
        ("langchain_core.messages.base", "BaseMessage"),
        ("langchain_core.messages.base", "BaseMessageChunk"),
        ("langchain_core.messages.human", "HumanMessage"),
        ("langchain_core.messages.human", "HumanMessageChunk"),
        ("langchain_core.messages.ai", "AIMessage"),
        ("langchain_core.messages.ai", "AIMessageChunk"),
        ("langchain_core.messages.system", "SystemMessage"),
        ("langchain_core.messages.system", "SystemMessageChunk"),
        ("langchain_core.messages.chat", "ChatMessage"),
        ("langchain_core.messages.chat", "ChatMessageChunk"),
        ("langchain_core.messages.tool", "ToolMessage"),
        ("langchain_core.messages.tool", "ToolMessageChunk"),
        ("langchain_core.messages.function", "FunctionMessage"),
        ("langchain_core.messages.function", "FunctionMessageChunk"),
        ("langchain_core.messages.modifier", "RemoveMessage"),
        # langchain-core document model
        ("langchain_core.documents.base", "Document"),
        # langgraph
        ("langgraph.types", "Send"),
        ("langgraph.types", "Interrupt"),
        ("langgraph.types", "Command"),
        ("langgraph.types", "StateSnapshot"),
        ("langgraph.types", "PregelTask"),
        ("langgraph.types", "Overwrite"),
        ("langgraph.store.base", "Item"),
        ("langgraph.store.base", "GetOp"),
    }
)

# Allowed (module, name, method) triples for EXT_METHOD_SINGLE_ARG.
# Only these specific method invocations are permitted during deserialization.
# This is separate from SAFE_MSGPACK_TYPES which only governs construction.
SAFE_MSGPACK_METHODS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("datetime", "datetime", "fromisoformat"),
    }
)


AllowedMsgpackModules = Iterable[tuple[str, ...] | type]
