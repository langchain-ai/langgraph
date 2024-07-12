from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    NamedTuple,
    Optional,
    TypedDict,
    Union,
    overload,
)

import httpx
import httpx_sse
import orjson
from httpx._types import QueryParamTypes

import langgraph_sdk
from langgraph_sdk.schema import (
    Assistant,
    Config,
    Cron,
    GraphSchema,
    Metadata,
    MultitaskStrategy,
    OnConflictBehavior,
    Run,
    StreamMode,
    Thread,
    ThreadState,
    ThreadStatus,
)

logger = logging.getLogger(__name__)


class RunCreate(TypedDict):
    """Payload for creating a background run."""

    thread_id: Optional[str]
    assistant_id: str
    input: Optional[dict]
    metadata: Optional[dict]
    config: Optional[Config]
    checkpoint_id: Optional[str]
    interrupt_before: Optional[list[str]]
    interrupt_after: Optional[list[str]]
    webhook: Optional[str]
    multitask_strategy: Optional[MultitaskStrategy]


def get_client(
    *, url: str = "http://localhost:8123", api_key: Optional[str] = None
) -> LangGraphClient:
    """Get a LangGraphClient instance.

    Args:
        url (str, optional): The URL of the LangGraph API. Defaults to "http://localhost:8123".
        api_key (str, optional): The API key. If not provided, it will be read from the environment.
            Precedence:
                1. explicit argument
                2. LANGGRAPH_API_KEY
                3. LANGSMITH_API_KEY
                4. LANGCHAIN_API_KEY
    """
    headers = {
        "User-Agent": f"langgraph-sdk-py/{langgraph_sdk.__version__}",
    }
    api_key = _get_api_key(api_key)
    if api_key:
        headers["x-api-key"] = api_key
    client = httpx.AsyncClient(
        base_url=url,
        transport=httpx.AsyncHTTPTransport(retries=5),
        timeout=httpx.Timeout(connect=5, read=60, write=60, pool=5),
        headers=headers,
    )
    return LangGraphClient(client)


class StreamPart(NamedTuple):
    event: str
    data: dict


class LangGraphClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.http = HttpClient(client)
        self.assistants = AssistantsClient(self.http)
        self.threads = ThreadsClient(self.http)
        self.runs = RunsClient(self.http)
        self.crons = CronClient(self.http)


class HttpClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def get(self, path: str, *, params: Optional[QueryParamTypes] = None) -> Any:
        """Make a GET request."""
        r = await self.client.get(path, params=params)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (await r.aread()).decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from langgraph-api: {body}", exc_info=e)
            raise e
        return await decode_json(r)

    async def post(self, path: str, *, json: Optional[dict]) -> Any:
        """Make a POST request."""
        if json is not None:
            headers, content = await encode_json(json)
        else:
            headers, content = {}, b""
        r = await self.client.post(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (await r.aread()).decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from langgraph-api: {body}", exc_info=e)
            raise e
        return await decode_json(r)

    async def put(self, path: str, *, json: dict) -> Any:
        """Make a PUT request."""
        headers, content = await encode_json(json)
        r = await self.client.put(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (await r.aread()).decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from langgraph-api: {body}", exc_info=e)
            raise e
        return await decode_json(r)

    async def patch(self, path: str, *, json: dict) -> Any:
        """Make a PATCH request."""
        headers, content = await encode_json(json)
        r = await self.client.patch(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (await r.aread()).decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from langgraph-api: {body}", exc_info=e)
            raise e
        return await decode_json(r)

    async def delete(self, path: str) -> None:
        """Make a DELETE request."""
        r = await self.client.delete(path)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (await r.aread()).decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from langgraph-api: {body}", exc_info=e)
            raise e

    async def stream(
        self, path: str, method: str, *, json: Optional[dict] = None
    ) -> AsyncIterator[StreamPart]:
        """Stream the results of a request using SSE."""
        headers, content = await encode_json(json)
        async with httpx_sse.aconnect_sse(
            self.client, method, path, headers=headers, content=content
        ) as sse:
            try:
                sse.response.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = (await sse.response.aread()).decode()
                if sys.version_info >= (3, 11):
                    e.add_note(body)
                else:
                    logger.error(f"Error from langgraph-api: {body}", exc_info=e)
                raise e
            async for event in sse.aiter_sse():
                yield StreamPart(
                    event.event, orjson.loads(event.data) if event.data else None
                )


def _orjson_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    elif hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


async def encode_json(json: Any) -> tuple[dict[str, str], bytes]:
    body = await asyncio.get_running_loop().run_in_executor(
        None,
        orjson.dumps,
        json,
        _orjson_default,
        orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, body


async def decode_json(r: httpx.Response) -> Any:
    body = await r.aread()
    return (
        await asyncio.get_running_loop().run_in_executor(None, orjson.loads, body)
        if body
        else None
    )


class AssistantsClient:
    def __init__(self, http: HttpClient) -> None:
        self.http = http

    async def get(self, assistant_id: str) -> Assistant:
        """Get an assistant by ID."""
        return await self.http.get(f"/assistants/{assistant_id}")

    async def get_graph(self, assistant_id: str) -> dict[str, list[dict[str, Any]]]:
        """Get the graph of an assistant by ID."""
        return await self.http.get(f"/assistants/{assistant_id}/graph")

    async def get_schemas(self, assistant_id: str) -> GraphSchema:
        """Get the schemas of an assistant by ID."""
        return await self.http.get(f"/assistants/{assistant_id}/schemas")

    async def create(
        self,
        graph_id: Optional[str],
        config: Optional[Config] = None,
        *,
        metadata: Metadata = None,
        assistant_id: Optional[str] = None,
    ) -> Assistant:
        """Create a new assistant."""
        payload: Dict[str, Any] = {
            "graph_id": graph_id,
        }
        if config:
            payload["config"] = config
        if metadata:
            payload["metadata"] = metadata
        if assistant_id:
            payload["assistant_id"] = assistant_id
        return await self.http.post("/assistants", json=payload)

    async def update(
        self,
        assistant_id: str,
        *,
        graph_id: Optional[str] = None,
        config: Optional[Config] = None,
        metadata: Metadata = None,
    ) -> Assistant:
        """Update an assistant."""
        payload: Dict[str, Any] = {}
        if graph_id:
            payload["graph_id"] = graph_id
        if config:
            payload["config"] = config
        if metadata:
            payload["metadata"] = metadata
        return await self.http.patch(
            f"/assistants/{assistant_id}",
            json=payload,
        )

    async def delete(
        self,
        assistant_id: str,
    ) -> None:
        """Delete an assistant."""
        await self.http.delete(f"/assistants/{assistant_id}")

    async def search(
        self,
        *,
        metadata: Metadata = None,
        graph_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Assistant]:
        """Search for assistants.

        Args:
            metadata (dict, optional): Metadata to filter by. Defaults to None.
            graph_id (str, optional): The ID of the graph to filter by. Defaults to None.
                The graph ID is normally set in your langgraph.json configuration.
            limit (int, optional): The maximum number of results to return. Defaults to 10.
            offset (int, optional): The number of results to skip. Defaults to 0.

        Returns:
            list[Assistant]: A list of assistants.
        """
        payload: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if metadata:
            payload["metadata"] = metadata
        if graph_id:
            payload["graph_id"] = graph_id
        return await self.http.post(
            "/assistants/search",
            json=payload,
        )


class ThreadsClient:
    def __init__(self, http: HttpClient) -> None:
        self.http = http

    async def get(self, thread_id: str) -> Thread:
        """Get a thread by ID."""
        return await self.http.get(f"/threads/{thread_id}")

    async def create(
        self,
        *,
        metadata: Metadata = None,
        thread_id: Optional[str] = None,
        if_exists: Optional[OnConflictBehavior] = None,
    ) -> Thread:
        """Create a new thread."""
        payload: Dict[str, Any] = {}
        if thread_id:
            payload["thread_id"] = thread_id
        if metadata:
            payload["metadata"] = metadata
        if if_exists:
            payload["if_exists"] = if_exists
        return await self.http.post("/threads", json=payload)

    async def update(self, thread_id: str, *, metadata: dict[str, Any]) -> Thread:
        """Update a thread."""
        return await self.http.patch(
            f"/threads/{thread_id}", json={"metadata": metadata}
        )

    async def delete(self, thread_id: str) -> None:
        """Delete a thread."""
        await self.http.delete(f"/threads/{thread_id}")

    async def search(
        self,
        *,
        metadata: Metadata = None,
        status: Optional[ThreadStatus] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Thread]:
        """Search for threads."""
        payload: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if metadata:
            payload["metadata"] = metadata
        if status:
            payload["status"] = status
        return await self.http.post(
            "/threads/search",
            json=payload,
        )

    async def get_state(
        self, thread_id: str, checkpoint_id: Optional[str] = None
    ) -> ThreadState:
        """Get the state of a thread."""
        if checkpoint_id:
            return await self.http.get(f"/threads/{thread_id}/state/{checkpoint_id}")
        else:
            return await self.http.get(f"/threads/{thread_id}/state")

    async def update_state(
        self,
        thread_id: str,
        values: dict,
        *,
        as_node: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> None:
        """Update the state of a thread."""
        payload: Dict[str, Any] = {
            "values": values,
        }
        if checkpoint_id:
            payload["checkpoint_id"] = checkpoint_id
        if as_node:
            payload["as_node"] = as_node
        return await self.http.post(f"/threads/{thread_id}/state", json=payload)

    async def patch_state(
        self,
        thread_id: Union[str, Config],
        metadata: dict,
    ) -> None:
        """Patch the state of a thread."""
        if isinstance(thread_id, dict):
            thread_id_: str = thread_id["configurable"]["thread_id"]
        else:
            thread_id_ = thread_id
        return await self.http.patch(
            f"/threads/{thread_id_}/state",
            json={"metadata": metadata},
        )

    async def get_history(
        self,
        thread_id: str,
        limit: int = 10,
        before: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> list[ThreadState]:
        """Get the history of a thread."""
        payload: Dict[str, Any] = {
            "limit": limit,
        }
        if before:
            payload["before"] = before
        if metadata:
            payload["metadata"] = metadata
        return await self.http.post(f"/threads/{thread_id}/history", json=payload)


class RunsClient:
    def __init__(self, http: HttpClient) -> None:
        self.http = http

    @overload
    def stream(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        stream_mode: Union[StreamMode, list[StreamMode]] = "values",
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        feedback_keys: Optional[list[str]] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
    ) -> AsyncIterator[StreamPart]:
        ...

    @overload
    def stream(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        stream_mode: Union[StreamMode, list[StreamMode]] = "values",
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        feedback_keys: Optional[list[str]] = None,
    ) -> AsyncIterator[StreamPart]:
        ...

    def stream(
        self,
        thread_id: Optional[str],
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        stream_mode: Union[StreamMode, list[StreamMode]] = "values",
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        feedback_keys: Optional[list[str]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
    ) -> AsyncIterator[StreamPart]:
        """Create a run and stream the results."""
        payload = {
            "input": input,
            "config": config,
            "metadata": metadata,
            "stream_mode": stream_mode,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "feedback_keys": feedback_keys,
            "webhook": webhook,
            "checkpoint_id": checkpoint_id,
            "multitask_strategy": multitask_strategy,
        }
        endpoint = (
            f"/threads/{thread_id}/runs/stream"
            if thread_id is not None
            else "/runs/stream"
        )
        return self.http.stream(
            endpoint, "POST", json={k: v for k, v in payload.items() if v is not None}
        )

    @overload
    async def create(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        webhook: Optional[str] = None,
    ) -> Run:
        ...

    @overload
    async def create(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
    ) -> Run:
        ...

    async def create(
        self,
        thread_id: Optional[str],
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
    ) -> Run:
        """Create a background run."""
        payload = {
            "input": input,
            "config": config,
            "metadata": metadata,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
            "checkpoint_id": checkpoint_id,
            "multitask_strategy": multitask_strategy,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if thread_id:
            return await self.http.post(f"/threads/{thread_id}/runs", json=payload)
        else:
            return await self.http.post("/runs", json=payload)

    async def create_batch(self, payloads: list[RunCreate]) -> list[Run]:
        """Create a batch of background runs."""

        def filter_payload(payload: RunCreate):
            return {k: v for k, v in payload.items() if v is not None}

        payloads = [filter_payload(payload) for payload in payloads]
        return await self.http.post("/runs/batch", json=payloads)

    @overload
    async def wait(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
    ) -> Union[list[dict], dict[str, Any]]:
        ...

    @overload
    async def wait(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
    ) -> Union[list[dict], dict[str, Any]]:
        ...

    async def wait(
        self,
        thread_id: Optional[str],
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
    ) -> Union[list[dict], dict[str, Any]]:
        """Create a run, wait for and return the final state."""
        payload = {
            "input": input,
            "config": config,
            "metadata": metadata,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
            "checkpoint_id": checkpoint_id,
            "multitask_strategy": multitask_strategy,
        }
        endpoint = (
            f"/threads/{thread_id}/runs/wait" if thread_id is not None else "/runs/wait"
        )
        return await self.http.post(
            endpoint, json={k: v for k, v in payload.items() if v is not None}
        )

    async def list(
        self, thread_id: str, *, limit: int = 10, offset: int = 0
    ) -> List[Run]:
        """List runs."""
        return await self.http.get(
            f"/threads/{thread_id}/runs?limit={limit}&offset={offset}"
        )

    async def get(self, thread_id: str, run_id: str) -> Run:
        """Get a run."""
        return await self.http.get(f"/threads/{thread_id}/runs/{run_id}")

    async def cancel(self, thread_id: str, run_id: str, *, wait: bool = False) -> None:
        """Get a run."""
        return await self.http.post(
            f"/threads/{thread_id}/runs/{run_id}/cancel?wait={1 if wait else 0}",
            json=None,
        )

    async def join(self, thread_id: str, run_id: str) -> None:
        """Block until a run is done."""
        return await self.http.get(f"/threads/{thread_id}/runs/{run_id}/join")

    async def delete(self, thread_id: str, run_id: str) -> None:
        """Delete a run."""
        await self.http.delete(f"/threads/{thread_id}/runs/{run_id}")


class CronClient:
    def __init__(self, http_client: HttpClient) -> None:
        self.http = http_client

    async def create_for_thread(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        schedule: str,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[str] = None,
    ) -> Run:
        """Create a background run."""
        payload = {
            "schedule": schedule,
            "input": input,
            "config": config,
            "metadata": metadata,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
        }
        if multitask_strategy:
            payload["multitask_strategy"] = multitask_strategy
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self.http.post(f"/threads/{thread_id}/runs/crons", json=payload)

    async def create(
        self,
        assistant_id: str,
        *,
        schedule: str,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[str] = None,
    ) -> Run:
        """Create a background run."""
        payload = {
            "schedule": schedule,
            "input": input,
            "config": config,
            "metadata": metadata,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
        }
        if multitask_strategy:
            payload["multitask_strategy"] = multitask_strategy
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self.http.post("/runs/crons", json=payload)

    async def delete(self, cron_id: str) -> None:
        """Delete a cron."""
        await self.http.delete(f"/runs/crons/{cron_id}")

    async def search(
        self,
        *,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Cron]:
        """Get a list of cron jobs."""
        payload = {
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "limit": limit,
            "offset": offset,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self.http.post("/runs/crons/search", json=payload)


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
