"""Low-level A2A protocol client and data types.

Implements the JSON-RPC 2.0 transport for A2A v1.0 specification.
This is an internal module — use `A2ARemoteGraph` from `langgraph.pregel.a2a_remote`.

Architecture:
    _a2a.py (this file)          <- Pure protocol layer, zero LangGraph dependencies
        |
    a2a_remote.py                <- LangGraph adapter layer, bridges two state models
        |
    remote A2A agent (any framework)

Design decisions:
    - dataclass over Pydantic: protocol layer doesn't need validation/serialization magic
    - httpx over requests: single library provides both sync + async clients
    - Only implements client-side subset of a2a.proto, not the full spec
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# JSON-RPC 2.0 version string, required by the A2A protocol
_JSONRPC_VERSION = "2.0"

# Default timeout configuration:
#   - connect/pool: 5s (connection setup should be quick)
#   - read/write: 300s (agent may need significant time to think)
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5, read=300, write=300, pool=5)


# --------------------------------------------------------------------------- #
#  A2A data types (mirrors a2a.proto, client-side subset)
# --------------------------------------------------------------------------- #


class TaskState(str, Enum):
    """A2A task lifecycle states.

    State transitions:
        SUBMITTED -> WORKING -> COMPLETED (normal completion)
                              -> FAILED    (processing error)
                              -> CANCELED  (cancelled by caller)
                              -> REJECTED  (rejected by agent)
                              -> INPUT_REQUIRED (needs user input, resumable)
                              -> AUTH_REQUIRED  (needs auth, resumable)

    Inherits str so enum values serialize directly to JSON, e.g.:
        json.dumps(TaskState.COMPLETED)  # "TASK_STATE_COMPLETED"
    """

    UNSPECIFIED = "TASK_STATE_UNSPECIFIED"
    SUBMITTED = "TASK_STATE_SUBMITTED"
    WORKING = "TASK_STATE_WORKING"
    COMPLETED = "TASK_STATE_COMPLETED"
    FAILED = "TASK_STATE_FAILED"
    CANCELED = "TASK_STATE_CANCELED"
    INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
    REJECTED = "TASK_STATE_REJECTED"
    AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"

    @property
    def is_terminal(self) -> bool:
        """Whether this is a terminal state (no further transitions possible).

        Once terminal, polling can stop.
        """
        return self in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
            TaskState.REJECTED,
        )

    @property
    def is_interrupted(self) -> bool:
        """Whether this is an interrupted state (needs external intervention to resume).

        Maps to LangGraph's GraphInterrupt:
            - INPUT_REQUIRED -> agent is waiting for user input
            - AUTH_REQUIRED  -> agent is waiting for authentication
        """
        return self in (TaskState.INPUT_REQUIRED, TaskState.AUTH_REQUIRED)


class Role(str, Enum):
    """A2A message role.

    Note: A2A only has USER/AGENT roles,
    unlike LangGraph which has human/ai/system/tool.
    """

    UNSPECIFIED = "ROLE_UNSPECIFIED"
    USER = "ROLE_USER"
    AGENT = "ROLE_AGENT"


@dataclass
class Part:
    """A2A message's minimal content unit.

    Similar to LangGraph message's `content` field but more flexible:
        - text:       plain text
        - data:       structured JSON data
        - url:        file/resource link
        - media_type: MIME type (e.g. "image/png")
        - filename:   filename

    A single message can contain multiple Parts for multimodal content.
    """

    text: str | None = None
    data: Any | None = None
    url: str | None = None
    media_type: str | None = None
    filename: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, only including non-None fields.

        A2A protocol convention: empty fields are omitted to reduce payload size.
        Note: text uses `is not None` (empty string "" is valid text),
        while media_type uses truthy check (empty string is meaningless).
        """
        d: dict[str, Any] = {}
        if self.text is not None:
            d["text"] = self.text
        if self.data is not None:
            d["data"] = self.data
        if self.url is not None:
            d["url"] = self.url
        if self.media_type:
            d["media_type"] = self.media_type
        if self.filename:
            d["filename"] = self.filename
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Part:
        """Deserialize from a JSON-RPC response dict."""
        return cls(
            text=d.get("text"),
            data=d.get("data"),
            url=d.get("url"),
            media_type=d.get("media_type"),
            filename=d.get("filename"),
            metadata=d.get("metadata"),
        )


@dataclass
class A2AMessage:
    """A2A message.

    Core field mapping to LangGraph:
        - context_id  <-> thread_id     (conversation ID)
        - task_id     <-> checkpoint_id (task ID)
        - parts       <-> content       (message content)
        - role        <-> type          (USER->human, AGENT->ai)
    """

    message_id: str                          # Unique message identifier
    role: Role                               # USER or AGENT
    parts: list[Part]                        # Message content (can be multiple)
    context_id: str | None = None            # Conversation ID (-> LangGraph thread_id)
    task_id: str | None = None               # Task ID (-> LangGraph checkpoint_id)
    metadata: dict[str, Any] | None = None   # Custom metadata
    extensions: list[str] | None = None      # A2A protocol extension identifiers
    reference_task_ids: list[str] | None = None  # Referenced task IDs

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-RPC params."""
        d: dict[str, Any] = {
            "message_id": self.message_id,
            "role": self.role.value,         # Enum value -> string
            "parts": [p.to_dict() for p in self.parts],
        }
        # Optional fields only included when present
        if self.context_id:
            d["context_id"] = self.context_id
        if self.task_id:
            d["task_id"] = self.task_id
        if self.metadata:
            d["metadata"] = self.metadata
        if self.extensions:
            d["extensions"] = self.extensions
        if self.reference_task_ids:
            d["reference_task_ids"] = self.reference_task_ids
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> A2AMessage:
        """Deserialize from a JSON-RPC response."""
        return cls(
            message_id=d["message_id"],
            role=Role(d["role"]),
            parts=[Part.from_dict(p) for p in d.get("parts", [])],
            context_id=d.get("context_id"),
            task_id=d.get("task_id"),
            metadata=d.get("metadata"),
            extensions=d.get("extensions"),
            reference_task_ids=d.get("reference_task_ids"),
        )


@dataclass
class TaskStatus:
    """A2A task status snapshot.

    Returned by tasks/get or received via SSE events.
    The message field carries context about the state (e.g. failure reason,
    interruption prompt, etc.).
    """

    state: TaskState
    message: A2AMessage | None = None
    timestamp: str | None = None             # ISO 8601 timestamp

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskStatus:
        msg = d.get("message")
        return cls(
            state=TaskState(d["state"]),
            message=A2AMessage.from_dict(msg) if msg else None,
            timestamp=d.get("timestamp"),
        )


@dataclass
class Artifact:
    """A2A task artifact — output produced by the agent.

    Examples of artifacts:
        - Translated text
        - Generated code
        - Analysis report

    A single Task can have multiple Artifacts (multi-step output).
    """

    artifact_id: str
    parts: list[Part]                        # Artifact content
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Artifact:
        return cls(
            artifact_id=d["artifact_id"],
            parts=[Part.from_dict(p) for p in d.get("parts", [])],
            name=d.get("name"),
            description=d.get("description"),
            metadata=d.get("metadata"),
        )


@dataclass
class A2ATask:
    """A2A Task — the core protocol object.

    A single message/send call creates a Task with a full lifecycle:
        SUBMITTED -> WORKING -> COMPLETED/FAILED/INPUT_REQUIRED/...

    Fields:
        - id:         Unique task ID (-> LangGraph checkpoint_id)
        - context_id: Conversation ID (-> LangGraph thread_id)
        - status:     Current state snapshot
        - artifacts:  Output artifacts list
        - history:    Full message history
    """

    id: str
    status: TaskStatus
    context_id: str | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    history: list[A2AMessage] = field(default_factory=list)
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> A2ATask:
        return cls(
            id=d["id"],
            status=TaskStatus.from_dict(d["status"]),
            context_id=d.get("context_id"),
            artifacts=[Artifact.from_dict(a) for a in d.get("artifacts", [])],
            history=[A2AMessage.from_dict(m) for m in d.get("history", [])],
            metadata=d.get("metadata"),
        )


@dataclass
class AgentSkill:
    """A skill declared by the agent in its Agent Card.

    Used for service discovery — callers can select agents
    based on tags/description.
    """

    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    examples: list[str] | None = None        # Example inputs
    input_modes: list[str] | None = None     # Supported input formats
    output_modes: list[str] | None = None    # Supported output formats


@dataclass
class AgentCapabilities:
    """Agent capability declarations.

    A2ARemoteGraph uses these to decide communication strategy:
        - streaming=True  -> use message/stream (SSE)
        - streaming=False -> use message/send + poll tasks/get
    """

    streaming: bool = False
    push_notifications: bool = False
    extended_agent_card: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentCapabilities:
        return cls(
            streaming=d.get("streaming", False),
            push_notifications=d.get("push_notifications", False),
            extended_agent_card=d.get("extended_agent_card", False),
        )


@dataclass
class AgentInterface:
    """Agent communication interface declaration.

    An agent can support multiple protocol bindings (HTTP, gRPC, WebSocket, etc.).
    We only care about protocol_binding == "JSONRPC".
    """

    url: str                                 # JSON-RPC endpoint URL
    protocol_binding: str                    # "JSONRPC" / "HTTP" / ...
    protocol_version: str                    # "1.0"
    tenant: str | None = None                # Multi-tenancy identifier

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentInterface:
        return cls(
            url=d["url"],
            protocol_binding=d["protocol_binding"],
            protocol_version=d["protocol_version"],
            tenant=d.get("tenant"),
        )


@dataclass
class AgentCard:
    """Agent Card — A2A's service discovery mechanism.

    Similar to an OpenAPI spec: describes who an agent is, what it can do,
    and how to call it. Typically hosted at /.well-known/agent-card.json:
        https://agent.example.com/.well-known/agent-card.json

    A2ARemoteGraph.from_agent_card_sync() fetches and parses the Agent Card
    to extract the JSON-RPC endpoint and capability declarations.
    """

    name: str                                # Agent name
    description: str                         # Agent description
    version: str                             # Agent version
    supported_interfaces: list[AgentInterface]  # Communication interfaces
    capabilities: AgentCapabilities          # Capability declarations
    skills: list[AgentSkill]                 # Skill list
    default_input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = field(default_factory=lambda: ["text/plain"])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentCard:
        skills = [
            AgentSkill(
                id=s["id"],
                name=s["name"],
                description=s["description"],
                tags=s.get("tags", []),
                examples=s.get("examples"),
                input_modes=s.get("input_modes"),
                output_modes=s.get("output_modes"),
            )
            for s in d.get("skills", [])
        ]
        return cls(
            name=d["name"],
            description=d["description"],
            version=d["version"],
            supported_interfaces=[
                AgentInterface.from_dict(i)
                for i in d.get("supported_interfaces", [])
            ],
            capabilities=AgentCapabilities.from_dict(d.get("capabilities", {})),
            skills=skills,
            default_input_modes=d.get("default_input_modes", ["text/plain"]),
            default_output_modes=d.get("default_output_modes", ["text/plain"]),
        )

    @property
    def jsonrpc_url(self) -> str | None:
        """Find the JSON-RPC endpoint URL from supported_interfaces.

        Returns None if no JSONRPC interface is declared;
        the caller will fall back to deriving it from the Agent Card URL.
        """
        for iface in self.supported_interfaces:
            if iface.protocol_binding == "JSONRPC":
                return iface.url
        return None


# --------------------------------------------------------------------------- #
#  A2A error
# --------------------------------------------------------------------------- #


class A2AClientError(Exception):
    """Wraps a JSON-RPC 2.0 error response.

    Raised when the remote agent returns {"error": {"code": -32600, "message": "..."}}.
    Error codes follow the JSON-RPC 2.0 spec:
        - -32700: Parse error
        - -32600: Invalid Request
        - -32601: Method not found
        - -32602: Invalid params
        - -32603: Internal error
        - Custom negative codes: Application-level errors
    """

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.data = data
        super().__init__(f"A2A error {code}: {message}")


# --------------------------------------------------------------------------- #
#  Sync JSON-RPC client
# --------------------------------------------------------------------------- #


class SyncA2AClient:
    """Synchronous A2A JSON-RPC 2.0 client.

    Built on httpx.Client, provides:
        - send_message():           message/send (request/response)
        - send_streaming_message(): message/stream (SSE streaming)
        - get_task():               tasks/get (query task state)
        - cancel_task():            tasks/cancel (cancel a task)
        - fetch_agent_card():       GET Agent Card (service discovery)
    """

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
    ):
        self.url = url.rstrip("/")           # Strip trailing slash to prevent //
        # Standard JSON-RPC request headers
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"  # Bearer token auth
        if headers:
            h.update(headers)                # Custom headers can override defaults
        self._http = httpx.Client(headers=h, timeout=timeout or _DEFAULT_TIMEOUT)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make a single JSON-RPC 2.0 call.

        Builds the standard JSON-RPC payload:
            {"jsonrpc": "2.0", "id": "<uuid>", "method": "...", "params": {...}}

        Returns the `result` field from the response.
        Raises A2AClientError if the response contains an `error` field.
        """
        payload = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": str(uuid.uuid4()),         # Request ID for correlating req/resp
            "method": method,
            "params": params,
        }
        resp = self._http.post(self.url, json=payload)
        resp.raise_for_status()              # Check HTTP-level errors (4xx/5xx) first
        body = resp.json()
        if "error" in body:                  # Then check JSON-RPC-level errors
            e = body["error"]
            raise A2AClientError(e["code"], e["message"], e.get("data"))
        return body.get("result", {})

    def send_message(
        self,
        message: A2AMessage,
        *,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call message/send — send a message and wait for a full response.

        Response is typically a Task dict (with id + status),
        but may also be a Message dict (with message_id).
        """
        params: dict[str, Any] = {"message": message.to_dict()}
        if configuration:
            params["configuration"] = configuration
        if metadata:
            params["metadata"] = metadata
        return self._call("message/send", params)

    def send_streaming_message(
        self,
        message: A2AMessage,
        *,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Call message/stream — send a message and receive SSE events.

        SSE event format:
            data: {"id": "task_1", "status": {"state": "TASK_STATE_WORKING"}}
            data: {"task_id": "task_1", "artifact": {"artifact_id": "a1", ...}}
            data: {"id": "task_1", "status": {"state": "TASK_STATE_COMPLETED"}, ...}

        Each `data:` line is parsed into a dict and yielded.
        Uses httpx.stream() context manager to avoid loading the entire response
        into memory.
        """
        params: dict[str, Any] = {"message": message.to_dict()}
        if configuration:
            params["configuration"] = configuration
        if metadata:
            params["metadata"] = metadata
        payload = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": str(uuid.uuid4()),
            "method": "message/stream",
            "params": params,
        }
        with self._http.stream("POST", self.url, json=payload) as resp:
            resp.raise_for_status()
            yield from _iter_sse_lines(resp.iter_lines())

    def get_task(
        self, task_id: str, *, history_length: int | None = None
    ) -> dict[str, Any]:
        """Call tasks/get — query the current state of a task.

        Used for polling: called repeatedly after send_message until the
        state becomes terminal. history_length limits the returned message
        history count.
        """
        params: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params["history_length"] = history_length
        return self._call("tasks/get", params)

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Call tasks/cancel — cancel an in-progress task."""
        return self._call("tasks/cancel", {"id": task_id})

    def fetch_agent_card(self, card_url: str) -> AgentCard:
        """GET request to fetch an Agent Card (plain HTTP GET, not JSON-RPC).

        Agent Cards are typically at /.well-known/agent-card.json
        """
        resp = self._http.get(card_url)
        resp.raise_for_status()
        return AgentCard.from_dict(resp.json())


# --------------------------------------------------------------------------- #
#  Async JSON-RPC client
# --------------------------------------------------------------------------- #


class AsyncA2AClient:
    """Asynchronous A2A JSON-RPC 2.0 client.

    Symmetric with SyncA2AClient, built on httpx.AsyncClient.
    All method signatures and logic are identical, just with async/await.
    """

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
    ):
        self.url = url.rstrip("/")
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        if headers:
            h.update(headers)
        self._http = httpx.AsyncClient(headers=h, timeout=timeout or _DEFAULT_TIMEOUT)

    async def close(self) -> None:
        """Close the underlying async HTTP connection pool."""
        await self._http.aclose()

    async def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Async JSON-RPC 2.0 call; logic mirrors SyncA2AClient._call."""
        payload = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }
        resp = await self._http.post(self.url, json=payload)
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            e = body["error"]
            raise A2AClientError(e["code"], e["message"], e.get("data"))
        return body.get("result", {})

    async def send_message(
        self,
        message: A2AMessage,
        *,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async message/send."""
        params: dict[str, Any] = {"message": message.to_dict()}
        if configuration:
            params["configuration"] = configuration
        if metadata:
            params["metadata"] = metadata
        return await self._call("message/send", params)

    async def send_streaming_message(
        self,
        message: A2AMessage,
        *,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async message/stream (SSE)."""
        params: dict[str, Any] = {"message": message.to_dict()}
        if configuration:
            params["configuration"] = configuration
        if metadata:
            params["metadata"] = metadata
        payload = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": str(uuid.uuid4()),
            "method": "message/stream",
            "params": params,
        }
        async with self._http.stream("POST", self.url, json=payload) as resp:
            resp.raise_for_status()
            async for event in _aiter_sse_lines(resp.aiter_lines()):
                yield event

    async def get_task(
        self, task_id: str, *, history_length: int | None = None
    ) -> dict[str, Any]:
        """Async tasks/get."""
        params: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params["history_length"] = history_length
        return await self._call("tasks/get", params)

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Async tasks/cancel."""
        return await self._call("tasks/cancel", {"id": task_id})

    async def fetch_agent_card(self, card_url: str) -> AgentCard:
        """Async Agent Card fetch."""
        resp = await self._http.get(card_url)
        resp.raise_for_status()
        return AgentCard.from_dict(resp.json())


# --------------------------------------------------------------------------- #
#  SSE helpers
# --------------------------------------------------------------------------- #


def _iter_sse_lines(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse an SSE (Server-Sent Events) stream.

    SSE format: each event line starts with "data:", followed by a JSON string:
        data: {"id": "task_1", "status": {"state": "TASK_STATE_WORKING"}}
        (blank line)
        data: {"task_id": "task_1", "artifact": {...}}

    Malformed lines are logged at debug level and skipped without breaking
    the stream.
    """
    for line in lines:
        if line.startswith("data:"):
            data_str = line[5:].strip()      # Strip "data:" prefix and whitespace
            if data_str:
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed SSE data: %s", data_str)


async def _aiter_sse_lines(
    lines: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    """Async SSE parser; logic mirrors _iter_sse_lines."""
    async for line in lines:
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str:
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed SSE data: %s", data_str)
