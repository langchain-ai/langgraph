"""The LangGraph client implementations connect to the LangGraph API.

This module provides both asynchronous (LangGraphClient) and synchronous (SyncLanggraphClient)
clients to interacting with the LangGraph API's core resources such as
Assistants, Threads, Runs, and Cron jobs, as well as its persistent
document Store.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    overload,
)

import httpx
import orjson
from httpx._types import QueryParamTypes

import langgraph_sdk
from langgraph_sdk.schema import (
    All,
    Assistant,
    AssistantVersion,
    CancelAction,
    Checkpoint,
    Command,
    Config,
    Cron,
    DisconnectMode,
    GraphSchema,
    IfNotExists,
    Item,
    Json,
    ListNamespaceResponse,
    MultitaskStrategy,
    OnCompletionBehavior,
    OnConflictBehavior,
    Run,
    RunCreate,
    RunStatus,
    SearchItemsResponse,
    StreamMode,
    StreamPart,
    Subgraphs,
    Thread,
    ThreadState,
    ThreadStatus,
    ThreadUpdateStateResponse,
)
from langgraph_sdk.sse import SSEDecoder, aiter_lines_raw, iter_lines_raw

logger = logging.getLogger(__name__)


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


def get_client(
    *,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> LangGraphClient:
    """Get a LangGraphClient instance.

    Args:
        url: The URL of the LangGraph API.
        api_key: The API key. If not provided, it will be read from the environment.
            Precedence:
                1. explicit argument
                2. LANGGRAPH_API_KEY
                3. LANGSMITH_API_KEY
                4. LANGCHAIN_API_KEY
        headers: Optional custom headers

    Returns:
        LangGraphClient: The top-level client for accessing AssistantsClient,
        ThreadsClient, RunsClient, and CronClient.

    Example:

        from langgraph_sdk import get_client

        # get top-level LangGraphClient
        client = get_client(url="http://localhost:8123")

        # example usage: client.<model>.<method_name>()
        assistants = await client.assistants.get(assistant_id="some_uuid")
    """
    transport: Optional[httpx.AsyncBaseTransport] = None
    if url is None:
        try:
            from langgraph_api.server import app  # type: ignore

            url = "http://api"
            transport = httpx.ASGITransport(app, root_path="/noauth")
        except Exception:
            url = "http://localhost:8123"

    if transport is None:
        transport = httpx.AsyncHTTPTransport(retries=5)

    client = httpx.AsyncClient(
        base_url=url,
        transport=transport,
        timeout=httpx.Timeout(connect=5, read=300, write=300, pool=5),
        headers=get_headers(api_key, headers),
    )
    return LangGraphClient(client)


class LangGraphClient:
    """Top-level client for LangGraph API.

    Attributes:
        assistants: Manages versioned configuration for your graphs.
        threads: Handles (potentially) multi-turn interactions, such as conversational threads.
        runs: Controls individual invocations of the graph.
        crons: Manages scheduled operations.
        store: Interfaces with persistent, shared data storage.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.http = HttpClient(client)
        self.assistants = AssistantsClient(self.http)
        self.threads = ThreadsClient(self.http)
        self.runs = RunsClient(self.http)
        self.crons = CronClient(self.http)
        self.store = StoreClient(self.http)


class HttpClient:
    """Handle async requests to the LangGraph API.

    Adds additional error messaging & content handling above the
    provided httpx client.

    Attributes:
        client (httpx.AsyncClient): Underlying HTTPX async client.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def get(self, path: str, *, params: Optional[QueryParamTypes] = None) -> Any:
        """Send a GET request."""
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
        return await adecode_json(r)

    async def post(self, path: str, *, json: Optional[dict]) -> Any:
        """Send a POST request."""
        if json is not None:
            headers, content = await aencode_json(json)
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
        return await adecode_json(r)

    async def put(self, path: str, *, json: dict) -> Any:
        """Send a PUT request."""
        headers, content = await aencode_json(json)
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
        return await adecode_json(r)

    async def patch(self, path: str, *, json: dict) -> Any:
        """Send a PATCH request."""
        headers, content = await aencode_json(json)
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
        return await adecode_json(r)

    async def delete(self, path: str, *, json: Optional[Any] = None) -> None:
        """Send a DELETE request."""
        r = await self.client.request("DELETE", path, json=json)
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
        self,
        path: str,
        method: str,
        *,
        json: Optional[dict] = None,
        params: Optional[QueryParamTypes] = None,
    ) -> AsyncIterator[StreamPart]:
        """Stream results using SSE."""
        headers, content = await aencode_json(json)
        headers["Accept"] = "text/event-stream"
        headers["Cache-Control"] = "no-store"

        async with self.client.stream(
            method, path, headers=headers, content=content, params=params
        ) as res:
            # check status
            try:
                res.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = (await res.aread()).decode()
                if sys.version_info >= (3, 11):
                    e.add_note(body)
                else:
                    logger.error(f"Error from langgraph-api: {body}", exc_info=e)
                raise e
            # check content type
            content_type = res.headers.get("content-type", "").partition(";")[0]
            if "text/event-stream" not in content_type:
                raise httpx.TransportError(
                    "Expected response header Content-Type to contain 'text/event-stream', "
                    f"got {content_type!r}"
                )
            # parse SSE
            decoder = SSEDecoder()
            async for line in aiter_lines_raw(res):
                sse = decoder.decode(line=line.rstrip(b"\n"))
                if sse is not None:
                    yield sse


async def aencode_json(json: Any) -> tuple[dict[str, str], bytes]:
    if json is None:
        return {}, None
    body = await asyncio.get_running_loop().run_in_executor(
        None,
        orjson.dumps,
        json,
        orjson_default,
        orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, body


async def adecode_json(r: httpx.Response) -> Any:
    body = await r.aread()
    return (
        await asyncio.get_running_loop().run_in_executor(None, orjson.loads, body)
        if body
        else None
    )


class AssistantsClient:
    """Client for managing assistants in LangGraph.

    This class provides methods to interact with assistants,
    which are versioned configurations of your graph.

    Example:

        client = get_client()
        assistant = await client.assistants.get("assistant_id_123")
    """

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    async def get(self, assistant_id: str) -> Assistant:
        """Get an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get.

        Returns:
            Assistant: Assistant Object.

        Example Usage:

            assistant = await client.assistants.get(
                assistant_id="my_assistant_id"
            )
            print(assistant)

            ----------------------------------------------------

            {
                'assistant_id': 'my_assistant_id',
                'graph_id': 'agent',
                'created_at': '2024-06-25T17:10:33.109781+00:00',
                'updated_at': '2024-06-25T17:10:33.109781+00:00',
                'config': {},
                'metadata': {'created_by': 'system'}
            }

        """  # noqa: E501
        return await self.http.get(f"/assistants/{assistant_id}")

    async def get_graph(
        self, assistant_id: str, *, xray: Union[int, bool] = False
    ) -> dict[str, list[dict[str, Any]]]:
        """Get the graph of an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get the graph of.
            xray: Include graph representation of subgraphs. If an integer value is provided, only subgraphs with a depth less than or equal to the value will be included.

        Returns:
            Graph: The graph information for the assistant in JSON format.

        Example Usage:

            graph_info = await client.assistants.get_graph(
                assistant_id="my_assistant_id"
            )
            print(graph_info)

            --------------------------------------------------------------------------------------------------------------------------

            {
                'nodes':
                    [
                        {'id': '__start__', 'type': 'schema', 'data': '__start__'},
                        {'id': '__end__', 'type': 'schema', 'data': '__end__'},
                        {'id': 'agent','type': 'runnable','data': {'id': ['langgraph', 'utils', 'RunnableCallable'],'name': 'agent'}},
                    ],
                'edges':
                    [
                        {'source': '__start__', 'target': 'agent'},
                        {'source': 'agent','target': '__end__'}
                    ]
            }


        """  # noqa: E501
        return await self.http.get(
            f"/assistants/{assistant_id}/graph", params={"xray": xray}
        )

    async def get_schemas(self, assistant_id: str) -> GraphSchema:
        """Get the schemas of an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get the schema of.

        Returns:
            GraphSchema: The graph schema for the assistant.

        Example Usage:

            schema = await client.assistants.get_schemas(
                assistant_id="my_assistant_id"
            )
            print(schema)

            ----------------------------------------------------------------------------------------------------------------------------

            {
                'graph_id': 'agent',
                'state_schema':
                    {
                        'title': 'LangGraphInput',
                        '$ref': '#/definitions/AgentState',
                        'definitions':
                            {
                                'BaseMessage':
                                    {
                                        'title': 'BaseMessage',
                                        'description': 'Base abstract Message class. Messages are the inputs and outputs of ChatModels.',
                                        'type': 'object',
                                        'properties':
                                            {
                                             'content':
                                                {
                                                    'title': 'Content',
                                                    'anyOf': [
                                                        {'type': 'string'},
                                                        {'type': 'array','items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}
                                                    ]
                                                },
                                            'additional_kwargs':
                                                {
                                                    'title': 'Additional Kwargs',
                                                    'type': 'object'
                                                },
                                            'response_metadata':
                                                {
                                                    'title': 'Response Metadata',
                                                    'type': 'object'
                                                },
                                            'type':
                                                {
                                                    'title': 'Type',
                                                    'type': 'string'
                                                },
                                            'name':
                                                {
                                                    'title': 'Name',
                                                    'type': 'string'
                                                },
                                            'id':
                                                {
                                                    'title': 'Id',
                                                    'type': 'string'
                                                }
                                            },
                                        'required': ['content', 'type']
                                    },
                                'AgentState':
                                    {
                                        'title': 'AgentState',
                                        'type': 'object',
                                        'properties':
                                            {
                                                'messages':
                                                    {
                                                        'title': 'Messages',
                                                        'type': 'array',
                                                        'items': {'$ref': '#/definitions/BaseMessage'}
                                                    }
                                            },
                                        'required': ['messages']
                                    }
                            }
                    },
                'config_schema':
                    {
                        'title': 'Configurable',
                        'type': 'object',
                        'properties':
                            {
                                'model_name':
                                    {
                                        'title': 'Model Name',
                                        'enum': ['anthropic', 'openai'],
                                        'type': 'string'
                                    }
                            }
                    }
            }

        """  # noqa: E501
        return await self.http.get(f"/assistants/{assistant_id}/schemas")

    async def get_subgraphs(
        self, assistant_id: str, namespace: Optional[str] = None, recurse: bool = False
    ) -> Subgraphs:
        """Get the schemas of an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get the schema of.

        Returns:
            Subgraphs: The graph schema for the assistant.

        """  # noqa: E501
        if namespace is not None:
            return await self.http.get(
                f"/assistants/{assistant_id}/subgraphs/{namespace}",
                params={"recurse": recurse},
            )
        else:
            return await self.http.get(
                f"/assistants/{assistant_id}/subgraphs",
                params={"recurse": recurse},
            )

    async def create(
        self,
        graph_id: Optional[str],
        config: Optional[Config] = None,
        *,
        metadata: Json = None,
        assistant_id: Optional[str] = None,
        if_exists: Optional[OnConflictBehavior] = None,
        name: Optional[str] = None,
    ) -> Assistant:
        """Create a new assistant.

        Useful when graph is configurable and you want to create different assistants based on different configurations.

        Args:
            graph_id: The ID of the graph the assistant should use. The graph ID is normally set in your langgraph.json configuration.
            config: Configuration to use for the graph.
            metadata: Metadata to add to assistant.
            assistant_id: Assistant ID to use, will default to a random UUID if not provided.
            if_exists: How to handle duplicate creation. Defaults to 'raise' under the hood.
                Must be either 'raise' (raise error if duplicate), or 'do_nothing' (return existing assistant).
            name: The name of the assistant. Defaults to 'Untitled' under the hood.

        Returns:
            Assistant: The created assistant.

        Example Usage:

            assistant = await client.assistants.create(
                graph_id="agent",
                config={"configurable": {"model_name": "openai"}},
                metadata={"number":1},
                assistant_id="my-assistant-id",
                if_exists="do_nothing",
                name="my_name"
            )
        """  # noqa: E501
        payload: Dict[str, Any] = {
            "graph_id": graph_id,
        }
        if config:
            payload["config"] = config
        if metadata:
            payload["metadata"] = metadata
        if assistant_id:
            payload["assistant_id"] = assistant_id
        if if_exists:
            payload["if_exists"] = if_exists
        if name:
            payload["name"] = name
        return await self.http.post("/assistants", json=payload)

    async def update(
        self,
        assistant_id: str,
        *,
        graph_id: Optional[str] = None,
        config: Optional[Config] = None,
        metadata: Json = None,
        name: Optional[str] = None,
    ) -> Assistant:
        """Update an assistant.

        Use this to point to a different graph, update the configuration, or change the metadata of an assistant.

        Args:
            assistant_id: Assistant to update.
            graph_id: The ID of the graph the assistant should use.
                The graph ID is normally set in your langgraph.json configuration. If None, assistant will keep pointing to same graph.
            config: Configuration to use for the graph.
            metadata: Metadata to merge with existing assistant metadata.
            name: The new name for the assistant.

        Returns:
            Assistant: The updated assistant.

        Example Usage:

            assistant = await client.assistants.update(
                assistant_id='e280dad7-8618-443f-87f1-8e41841c180f',
                graph_id="other-graph",
                config={"configurable": {"model_name": "anthropic"}},
                metadata={"number":2}
            )

        """  # noqa: E501
        payload: Dict[str, Any] = {}
        if graph_id:
            payload["graph_id"] = graph_id
        if config:
            payload["config"] = config
        if metadata:
            payload["metadata"] = metadata
        if name:
            payload["name"] = name
        return await self.http.patch(
            f"/assistants/{assistant_id}",
            json=payload,
        )

    async def delete(
        self,
        assistant_id: str,
    ) -> None:
        """Delete an assistant.

        Args:
            assistant_id: The assistant ID to delete.

        Returns:
            None

        Example Usage:

            await client.assistants.delete(
                assistant_id="my_assistant_id"
            )

        """  # noqa: E501
        await self.http.delete(f"/assistants/{assistant_id}")

    async def search(
        self,
        *,
        metadata: Json = None,
        graph_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Assistant]:
        """Search for assistants.

        Args:
            metadata: Metadata to filter by. Exact match filter for each KV pair.
            graph_id: The ID of the graph to filter by.
                The graph ID is normally set in your langgraph.json configuration.
            limit: The maximum number of results to return.
            offset: The number of results to skip.

        Returns:
            list[Assistant]: A list of assistants.

        Example Usage:

            assistants = await client.assistants.search(
                metadata = {"name":"my_name"},
                graph_id="my_graph_id",
                limit=5,
                offset=5
            )
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

    async def get_versions(
        self,
        assistant_id: str,
        metadata: Json = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[AssistantVersion]:
        """List all versions of an assistant.

        Args:
            assistant_id: The assistant ID to get versions for.
            metadata: Metadata to filter versions by. Exact match filter for each KV pair.
            limit: The maximum number of versions to return.
            offset: The number of versions to skip.

        Returns:
            list[Assistant]: A list of assistants.

        Example Usage:

            assistant_versions = await client.assistants.get_versions(
                assistant_id="my_assistant_id"
            )

        """  # noqa: E501

        payload: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if metadata:
            payload["metadata"] = metadata
        return await self.http.post(
            f"/assistants/{assistant_id}/versions", json=payload
        )

    async def set_latest(self, assistant_id: str, version: int) -> Assistant:
        """Change the version of an assistant.

        Args:
            assistant_id: The assistant ID to delete.
            version: The version to change to.

        Returns:
            Assistant: Assistant Object.

        Example Usage:

            new_version_assistant = await client.assistants.set_latest(
                assistant_id="my_assistant_id",
                version=3
            )

        """  # noqa: E501

        payload: Dict[str, Any] = {"version": version}

        return await self.http.post(f"/assistants/{assistant_id}/latest", json=payload)


class ThreadsClient:
    """Client for managing threads in LangGraph.

    A thread maintains the state of a graph across multiple interactions/invocations (aka runs).
    It accumulates and persists the graph's state, allowing for continuity between separate
    invocations of the graph.

    Example:

        client = get_client()
        new_thread = await client.threads.create(metadata={"user_id": "123"})
    """

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    async def get(self, thread_id: str) -> Thread:
        """Get a thread by ID.

        Args:
            thread_id: The ID of the thread to get.

        Returns:
            Thread: Thread object.

        Example Usage:

            thread = await client.threads.get(
                thread_id="my_thread_id"
            )
            print(thread)

            -----------------------------------------------------

            {
                'thread_id': 'my_thread_id',
                'created_at': '2024-07-18T18:35:15.540834+00:00',
                'updated_at': '2024-07-18T18:35:15.540834+00:00',
                'metadata': {'graph_id': 'agent'}
            }

        """  # noqa: E501

        return await self.http.get(f"/threads/{thread_id}")

    async def create(
        self,
        *,
        metadata: Json = None,
        thread_id: Optional[str] = None,
        if_exists: Optional[OnConflictBehavior] = None,
    ) -> Thread:
        """Create a new thread.

        Args:
            metadata: Metadata to add to thread.
            thread_id: ID of thread.
                If None, ID will be a randomly generated UUID.
            if_exists: How to handle duplicate creation. Defaults to 'raise' under the hood.
                Must be either 'raise' (raise error if duplicate), or 'do_nothing' (return existing thread).

        Returns:
            Thread: The created thread.

        Example Usage:

            thread = await client.threads.create(
                metadata={"number":1},
                thread_id="my-thread-id",
                if_exists="raise"
            )
        """  # noqa: E501
        payload: Dict[str, Any] = {}
        if thread_id:
            payload["thread_id"] = thread_id
        if metadata:
            payload["metadata"] = metadata
        if if_exists:
            payload["if_exists"] = if_exists
        return await self.http.post("/threads", json=payload)

    async def update(self, thread_id: str, *, metadata: dict[str, Any]) -> Thread:
        """Update a thread.

        Args:
            thread_id: ID of thread to update.
            metadata: Metadata to merge with existing thread metadata.

        Returns:
            Thread: The created thread.

        Example Usage:

            thread = await client.threads.update(
                thread_id="my-thread-id",
                metadata={"number":1},
            )
        """  # noqa: E501
        return await self.http.patch(
            f"/threads/{thread_id}", json={"metadata": metadata}
        )

    async def delete(self, thread_id: str) -> None:
        """Delete a thread.

        Args:
            thread_id: The ID of the thread to delete.

        Returns:
            None

        Example Usage:

            await client.threads.delete(
                thread_id="my_thread_id"
            )

        """  # noqa: E501
        await self.http.delete(f"/threads/{thread_id}")

    async def search(
        self,
        *,
        metadata: Json = None,
        values: Json = None,
        status: Optional[ThreadStatus] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Thread]:
        """Search for threads.

        Args:
            metadata: Thread metadata to filter on.
            values: State values to filter on.
            status: Thread status to filter on.
                Must be one of 'idle', 'busy', 'interrupted' or 'error'.
            limit: Limit on number of threads to return.
            offset: Offset in threads table to start search from.

        Returns:
            list[Thread]: List of the threads matching the search parameters.

        Example Usage:

            threads = await client.threads.search(
                metadata={"number":1},
                status="interrupted",
                limit=15,
                offset=5
            )

        """  # noqa: E501
        payload: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if metadata:
            payload["metadata"] = metadata
        if values:
            payload["values"] = values
        if status:
            payload["status"] = status
        return await self.http.post(
            "/threads/search",
            json=payload,
        )

    async def copy(self, thread_id: str) -> None:
        """Copy a thread.

        Args:
            thread_id: The ID of the thread to copy.

        Returns:
            None

        Example Usage:

            await client.threads.copy(
                thread_id="my_thread_id"
            )

        """  # noqa: E501
        return await self.http.post(f"/threads/{thread_id}/copy", json=None)

    async def get_state(
        self,
        thread_id: str,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,  # deprecated
        *,
        subgraphs: bool = False,
    ) -> ThreadState:
        """Get the state of a thread.

        Args:
            thread_id: The ID of the thread to get the state of.
            checkpoint: The checkpoint to get the state of.
            subgraphs: Include subgraphs states.

        Returns:
            ThreadState: the thread of the state.

        Example Usage:

            thread_state = await client.threads.get_state(
                thread_id="my_thread_id",
                checkpoint_id="my_checkpoint_id"
            )
            print(thread_state)

            ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

            {
                'values': {
                    'messages': [
                        {
                            'content': 'how are you?',
                            'additional_kwargs': {},
                            'response_metadata': {},
                            'type': 'human',
                            'name': None,
                            'id': 'fe0a5778-cfe9-42ee-b807-0adaa1873c10',
                            'example': False
                        },
                        {
                            'content': "I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, honest, and harmless.",
                            'additional_kwargs': {},
                            'response_metadata': {},
                            'type': 'ai',
                            'name': None,
                            'id': 'run-159b782c-b679-4830-83c6-cef87798fe8b',
                            'example': False,
                            'tool_calls': [],
                            'invalid_tool_calls': [],
                            'usage_metadata': None
                        }
                    ]
                },
                'next': [],
                'checkpoint':
                    {
                        'thread_id': 'e2496803-ecd5-4e0c-a779-3226296181c2',
                        'checkpoint_ns': '',
                        'checkpoint_id': '1ef4a9b8-e6fb-67b1-8001-abd5184439d1'
                    }
                'metadata':
                    {
                        'step': 1,
                        'run_id': '1ef4a9b8-d7da-679a-a45a-872054341df2',
                        'source': 'loop',
                        'writes':
                            {
                                'agent':
                                    {
                                        'messages': [
                                            {
                                                'id': 'run-159b782c-b679-4830-83c6-cef87798fe8b',
                                                'name': None,
                                                'type': 'ai',
                                                'content': "I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, honest, and harmless.",
                                                'example': False,
                                                'tool_calls': [],
                                                'usage_metadata': None,
                                                'additional_kwargs': {},
                                                'response_metadata': {},
                                                'invalid_tool_calls': []
                                            }
                                        ]
                                    }
                            },
                'user_id': None,
                'graph_id': 'agent',
                'thread_id': 'e2496803-ecd5-4e0c-a779-3226296181c2',
                'created_by': 'system',
                'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'},
                'created_at': '2024-07-25T15:35:44.184703+00:00',
                'parent_config':
                    {
                        'thread_id': 'e2496803-ecd5-4e0c-a779-3226296181c2',
                        'checkpoint_ns': '',
                        'checkpoint_id': '1ef4a9b8-d80d-6fa7-8000-9300467fad0f'
                    }
            }

        """  # noqa: E501
        if checkpoint:
            return await self.http.post(
                f"/threads/{thread_id}/state/checkpoint",
                json={"checkpoint": checkpoint, "subgraphs": subgraphs},
            )
        elif checkpoint_id:
            return await self.http.get(
                f"/threads/{thread_id}/state/{checkpoint_id}",
                params={"subgraphs": subgraphs},
            )
        else:
            return await self.http.get(
                f"/threads/{thread_id}/state",
                params={"subgraphs": subgraphs},
            )

    async def update_state(
        self,
        thread_id: str,
        values: Optional[Union[dict, Sequence[dict]]],
        *,
        as_node: Optional[str] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,  # deprecated
    ) -> ThreadUpdateStateResponse:
        """Update the state of a thread.

        Args:
            thread_id: The ID of the thread to update.
            values: The values to update the state with.
            as_node: Update the state as if this node had just executed.
            checkpoint: The checkpoint to update the state of.

        Returns:
            ThreadUpdateStateResponse: Response after updating a thread's state.

        Example Usage:

            response = await client.threads.update_state(
                thread_id="my_thread_id",
                values={"messages":[{"role": "user", "content": "hello!"}]},
                as_node="my_node",
            )
            print(response)

            ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

            {
                'checkpoint': {
                    'thread_id': 'e2496803-ecd5-4e0c-a779-3226296181c2',
                    'checkpoint_ns': '',
                    'checkpoint_id': '1ef4a9b8-e6fb-67b1-8001-abd5184439d1',
                    'checkpoint_map': {}
                }
            }

        """  # noqa: E501
        payload: Dict[str, Any] = {
            "values": values,
        }
        if checkpoint_id:
            payload["checkpoint_id"] = checkpoint_id
        if checkpoint:
            payload["checkpoint"] = checkpoint
        if as_node:
            payload["as_node"] = as_node
        return await self.http.post(f"/threads/{thread_id}/state", json=payload)

    async def get_history(
        self,
        thread_id: str,
        *,
        limit: int = 10,
        before: Optional[str | Checkpoint] = None,
        metadata: Optional[dict] = None,
        checkpoint: Optional[Checkpoint] = None,
    ) -> list[ThreadState]:
        """Get the state history of a thread.

        Args:
            thread_id: The ID of the thread to get the state history for.
            checkpoint: Return states for this subgraph. If empty defaults to root.
            limit: The maximum number of states to return.
            before: Return states before this checkpoint.
            metadata: Filter states by metadata key-value pairs.

        Returns:
            list[ThreadState]: the state history of the thread.

        Example Usage:

            thread_state = await client.threads.get_history(
                thread_id="my_thread_id",
                limit=5,
            )

        """  # noqa: E501
        payload: Dict[str, Any] = {
            "limit": limit,
        }
        if before:
            payload["before"] = before
        if metadata:
            payload["metadata"] = metadata
        if checkpoint:
            payload["checkpoint"] = checkpoint
        return await self.http.post(f"/threads/{thread_id}/history", json=payload)


class RunsClient:
    """Client for managing runs in LangGraph.

    A run is a single assistant invocation with optional input, config, and metadata.
    This client manages runs, which can be stateful (on threads) or stateless.

    Example:

        client = get_client()
        run = await client.runs.create(assistant_id="asst_123", thread_id="thread_456", input={"query": "Hello"})
    """

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    @overload
    def stream(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        feedback_keys: Optional[Sequence[str]] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> AsyncIterator[StreamPart]: ...

    @overload
    def stream(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        feedback_keys: Optional[Sequence[str]] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        if_not_exists: Optional[IfNotExists] = None,
        webhook: Optional[str] = None,
        after_seconds: Optional[int] = None,
    ) -> AsyncIterator[StreamPart]: ...

    def stream(
        self,
        thread_id: Optional[str],
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        feedback_keys: Optional[Sequence[str]] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> AsyncIterator[StreamPart]:
        """Create a run and stream the results.

        Args:
            thread_id: the thread ID to assign to the thread.
                If None will create a stateless run.
            assistant_id: The assistant ID or graph name to stream from.
                If using graph name, will default to first assistant created from that graph.
            input: The input to the graph.
            command: A command to execute. Cannot be combined with input.
            stream_mode: The stream mode(s) to use.
            stream_subgraphs: Whether to stream output from subgraphs.
            metadata: Metadata to assign to the run.
            config: The configuration for the assistant.
            checkpoint: The checkpoint to resume from.
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            feedback_keys: Feedback keys to assign to run.
            on_disconnect: The disconnect mode to use.
                Must be one of 'cancel' or 'continue'.
            on_completion: Whether to delete or keep the thread created for a stateless run.
                Must be one of 'delete' or 'keep'.
            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            if_not_exists: How to handle missing thread. Defaults to 'reject'.
                Must be either 'reject' (raise error if missing), or 'create' (create new thread).
            after_seconds: The number of seconds to wait before starting the run.
                Use to schedule future runs.

        Returns:
            AsyncIterator[StreamPart]: Asynchronous iterator of stream results.

        Example Usage:

            async for chunk in client.runs.stream(
                thread_id=None,
                assistant_id="agent",
                input={"messages": [{"role": "user", "content": "how are you?"}]},
                stream_mode=["values","debug"],
                metadata={"name":"my_run"},
                config={"configurable": {"model_name": "anthropic"}},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                feedback_keys=["my_feedback_key_1","my_feedback_key_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            ):
                print(chunk)

            ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            StreamPart(event='metadata', data={'run_id': '1ef4a9b8-d7da-679a-a45a-872054341df2'})
            StreamPart(event='values', data={'messages': [{'content': 'how are you?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'fe0a5778-cfe9-42ee-b807-0adaa1873c10', 'example': False}]})
            StreamPart(event='values', data={'messages': [{'content': 'how are you?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'fe0a5778-cfe9-42ee-b807-0adaa1873c10', 'example': False}, {'content': "I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, honest, and harmless.", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-159b782c-b679-4830-83c6-cef87798fe8b', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]})
            StreamPart(event='end', data=None)

        """  # noqa: E501
        payload = {
            "input": input,
            "command": {k: v for k, v in command.items() if v is not None}
            if command
            else None,
            "config": config,
            "metadata": metadata,
            "stream_mode": stream_mode,
            "stream_subgraphs": stream_subgraphs,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "feedback_keys": feedback_keys,
            "webhook": webhook,
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "multitask_strategy": multitask_strategy,
            "if_not_exists": if_not_exists,
            "on_disconnect": on_disconnect,
            "on_completion": on_completion,
            "after_seconds": after_seconds,
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
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> Run: ...

    @overload
    async def create(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> Run: ...

    async def create(
        self,
        thread_id: Optional[str],
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        after_seconds: Optional[int] = None,
    ) -> Run:
        """Create a background run.

        Args:
            thread_id: the thread ID to assign to the thread.
                If None will create a stateless run.
            assistant_id: The assistant ID or graph name to stream from.
                If using graph name, will default to first assistant created from that graph.
            input: The input to the graph.
            command: A command to execute. Cannot be combined with input.
            stream_mode: The stream mode(s) to use.
            stream_subgraphs: Whether to stream output from subgraphs.
            metadata: Metadata to assign to the run.
            config: The configuration for the assistant.
            checkpoint: The checkpoint to resume from.
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            on_completion: Whether to delete or keep the thread created for a stateless run.
                Must be one of 'delete' or 'keep'.
            if_not_exists: How to handle missing thread. Defaults to 'reject'.
                Must be either 'reject' (raise error if missing), or 'create' (create new thread).
            after_seconds: The number of seconds to wait before starting the run.
                Use to schedule future runs.

        Returns:
            Run: The created background run.

        Example Usage:

            background_run = await client.runs.create(
                thread_id="my_thread_id",
                assistant_id="my_assistant_id",
                input={"messages": [{"role": "user", "content": "hello!"}]},
                metadata={"name":"my_run"},
                config={"configurable": {"model_name": "openai"}},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            )
            print(background_run)

            --------------------------------------------------------------------------------

            {
                'run_id': 'my_run_id',
                'thread_id': 'my_thread_id',
                'assistant_id': 'my_assistant_id',
                'created_at': '2024-07-25T15:35:42.598503+00:00',
                'updated_at': '2024-07-25T15:35:42.598503+00:00',
                'metadata': {},
                'status': 'pending',
                'kwargs':
                    {
                        'input':
                            {
                                'messages': [
                                    {
                                        'role': 'user',
                                        'content': 'how are you?'
                                    }
                                ]
                            },
                        'config':
                            {
                                'metadata':
                                    {
                                        'created_by': 'system'
                                    },
                                'configurable':
                                    {
                                        'run_id': 'my_run_id',
                                        'user_id': None,
                                        'graph_id': 'agent',
                                        'thread_id': 'my_thread_id',
                                        'checkpoint_id': None,
                                        'model_name': "openai",
                                        'assistant_id': 'my_assistant_id'
                                    }
                            },
                        'webhook': "https://my.fake.webhook.com",
                        'temporary': False,
                        'stream_mode': ['values'],
                        'feedback_keys': None,
                        'interrupt_after': ["node_to_stop_after_1","node_to_stop_after_2"],
                        'interrupt_before': ["node_to_stop_before_1","node_to_stop_before_2"]
                    },
                'multitask_strategy': 'interrupt'
            }

        """  # noqa: E501
        payload = {
            "input": input,
            "command": {k: v for k, v in command.items() if v is not None}
            if command
            else None,
            "stream_mode": stream_mode,
            "stream_subgraphs": stream_subgraphs,
            "config": config,
            "metadata": metadata,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "multitask_strategy": multitask_strategy,
            "if_not_exists": if_not_exists,
            "on_completion": on_completion,
            "after_seconds": after_seconds,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if thread_id:
            return await self.http.post(f"/threads/{thread_id}/runs", json=payload)
        else:
            return await self.http.post("/runs", json=payload)

    async def create_batch(self, payloads: list[RunCreate]) -> list[Run]:
        """Create a batch of stateless background runs."""

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
        command: Optional[Command] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
        raise_error: bool = True,
    ) -> Union[list[dict], dict[str, Any]]: ...

    @overload
    async def wait(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
        raise_error: bool = True,
    ) -> Union[list[dict], dict[str, Any]]: ...

    async def wait(
        self,
        thread_id: Optional[str],
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
        raise_error: bool = True,
    ) -> Union[list[dict], dict[str, Any]]:
        """Create a run, wait until it finishes and return the final state.

        Args:
            thread_id: the thread ID to create the run on.
                If None will create a stateless run.
            assistant_id: The assistant ID or graph name to run.
                If using graph name, will default to first assistant created from that graph.
            input: The input to the graph.
            command: A command to execute. Cannot be combined with input.
            metadata: Metadata to assign to the run.
            config: The configuration for the assistant.
            checkpoint: The checkpoint to resume from.
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            webhook: Webhook to call after LangGraph API call is done.
            on_disconnect: The disconnect mode to use.
                Must be one of 'cancel' or 'continue'.
            on_completion: Whether to delete or keep the thread created for a stateless run.
                Must be one of 'delete' or 'keep'.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            if_not_exists: How to handle missing thread. Defaults to 'reject'.
                Must be either 'reject' (raise error if missing), or 'create' (create new thread).
            after_seconds: The number of seconds to wait before starting the run.
                Use to schedule future runs.

        Returns:
            Union[list[dict], dict[str, Any]]: The output of the run.

        Example Usage:

            final_state_of_run = await client.runs.wait(
                thread_id=None,
                assistant_id="agent",
                input={"messages": [{"role": "user", "content": "how are you?"}]},
                metadata={"name":"my_run"},
                config={"configurable": {"model_name": "anthropic"}},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            )
            print(final_state_of_run)

            -------------------------------------------------------------------------------------------------------------------------------------------

            {
                'messages': [
                    {
                        'content': 'how are you?',
                        'additional_kwargs': {},
                        'response_metadata': {},
                        'type': 'human',
                        'name': None,
                        'id': 'f51a862c-62fe-4866-863b-b0863e8ad78a',
                        'example': False
                    },
                    {
                        'content': "I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, honest, and harmless.",
                        'additional_kwargs': {},
                        'response_metadata': {},
                        'type': 'ai',
                        'name': None,
                        'id': 'run-bf1cd3c6-768f-4c16-b62d-ba6f17ad8b36',
                        'example': False,
                        'tool_calls': [],
                        'invalid_tool_calls': [],
                        'usage_metadata': None
                    }
                ]
            }

        """  # noqa: E501
        payload = {
            "input": input,
            "command": {k: v for k, v in command.items() if v is not None}
            if command
            else None,
            "config": config,
            "metadata": metadata,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "multitask_strategy": multitask_strategy,
            "if_not_exists": if_not_exists,
            "on_disconnect": on_disconnect,
            "on_completion": on_completion,
            "after_seconds": after_seconds,
        }
        endpoint = (
            f"/threads/{thread_id}/runs/wait" if thread_id is not None else "/runs/wait"
        )
        response = await self.http.post(
            endpoint, json={k: v for k, v in payload.items() if v is not None}
        )
        if (
            raise_error
            and isinstance(response, dict)
            and "__error__" in response
            and isinstance(response["__error__"], dict)
        ):
            raise Exception(
                f"{response['__error__'].get('error')}: {response['__error__'].get('message')}"
            )
        return response

    async def list(
        self,
        thread_id: str,
        *,
        limit: int = 10,
        offset: int = 0,
        status: Optional[RunStatus] = None,
    ) -> List[Run]:
        """List runs.

        Args:
            thread_id: The thread ID to list runs for.
            limit: The maximum number of results to return.
            offset: The number of results to skip.
            status: The status of the run to filter by.

        Returns:
            List[Run]: The runs for the thread.

        Example Usage:

            await client.runs.delete(
                thread_id="thread_id_to_delete",
                limit=5,
                offset=5,
            )

        """  # noqa: E501
        params = {
            "limit": limit,
            "offset": offset,
        }
        if status is not None:
            params["status"] = status
        return await self.http.get(f"/threads/{thread_id}/runs", params=params)

    async def get(self, thread_id: str, run_id: str) -> Run:
        """Get a run.

        Args:
            thread_id: The thread ID to get.
            run_id: The run ID to get.

        Returns:
            Run: Run object.

        Example Usage:

            run = await client.runs.get(
                thread_id="thread_id_to_delete",
                run_id="run_id_to_delete",
            )

        """  # noqa: E501

        return await self.http.get(f"/threads/{thread_id}/runs/{run_id}")

    async def cancel(
        self,
        thread_id: str,
        run_id: str,
        *,
        wait: bool = False,
        action: CancelAction = "interrupt",
    ) -> None:
        """Get a run.

        Args:
            thread_id: The thread ID to cancel.
            run_id: The run ID to cancel.
            wait: Whether to wait until run has completed.
            action: Action to take when cancelling the run. Possible values
                are `interrupt` or `rollback`. Default is `interrupt`.

        Returns:
            None

        Example Usage:

            await client.runs.cancel(
                thread_id="thread_id_to_cancel",
                run_id="run_id_to_cancel",
                wait=True,
                action="interrupt"
            )

        """  # noqa: E501
        return await self.http.post(
            f"/threads/{thread_id}/runs/{run_id}/cancel?wait={1 if wait else 0}&action={action}",
            json=None,
        )

    async def join(self, thread_id: str, run_id: str) -> dict:
        """Block until a run is done. Returns the final state of the thread.

        Args:
            thread_id: The thread ID to join.
            run_id: The run ID to join.

        Returns:
            None

        Example Usage:

            result =await client.runs.join(
                thread_id="thread_id_to_join",
                run_id="run_id_to_join"
            )

        """  # noqa: E501
        return await self.http.get(f"/threads/{thread_id}/runs/{run_id}/join")

    def join_stream(
        self, thread_id: str, run_id: str, *, cancel_on_disconnect: bool = False
    ) -> AsyncIterator[StreamPart]:
        """Stream output from a run in real-time, until the run is done.
        Output is not buffered, so any output produced before this call will
        not be received here.

        Args:
            thread_id: The thread ID to join.
            run_id: The run ID to join.
            cancel_on_disconnect: Whether to cancel the run when the stream is disconnected.

        Returns:
            None

        Example Usage:

            await client.runs.join_stream(
                thread_id="thread_id_to_join",
                run_id="run_id_to_join"
            )

        """  # noqa: E501
        return self.http.stream(
            f"/threads/{thread_id}/runs/{run_id}/stream",
            "GET",
            params={"cancel_on_disconnect": cancel_on_disconnect},
        )

    async def delete(self, thread_id: str, run_id: str) -> None:
        """Delete a run.

        Args:
            thread_id: The thread ID to delete.
            run_id: The run ID to delete.

        Returns:
            None

        Example Usage:

            await client.runs.delete(
                thread_id="thread_id_to_delete",
                run_id="run_id_to_delete"
            )

        """  # noqa: E501
        await self.http.delete(f"/threads/{thread_id}/runs/{run_id}")


class CronClient:
    """Client for managing recurrent runs (cron jobs) in LangGraph.

    A run is a single invocation of an assistant with optional input and config.
    This client allows scheduling recurring runs to occur automatically.

    Example:

        client = get_client()
        cron_job = await client.crons.create_for_thread(
            thread_id="thread_123",
            assistant_id="asst_456",
            schedule="0 9 * * *",
            input={"message": "Daily update"}
        )
    """

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
        interrupt_before: Optional[Union[All, list[str]]] = None,
        interrupt_after: Optional[Union[All, list[str]]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[str] = None,
    ) -> Run:
        """Create a cron job for a thread.

        Args:
            thread_id: the thread ID to run the cron job on.
            assistant_id: The assistant ID or graph name to use for the cron job.
                If using graph name, will default to first assistant created from that graph.
            schedule: The cron schedule to execute this job on.
            input: The input to the graph.
            metadata: Metadata to assign to the cron job runs.
            config: The configuration for the assistant.
            interrupt_before: Nodes to interrupt immediately before they get executed.

            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.

            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.

        Returns:
            Run: The cron run.

        Example Usage:

            cron_run = await client.crons.create_for_thread(
                thread_id="my-thread-id",
                assistant_id="agent",
                schedule="27 15 * * *",
                input={"messages": [{"role": "user", "content": "hello!"}]},
                metadata={"name":"my_run"},
                config={"configurable": {"model_name": "openai"}},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            )

        """  # noqa: E501
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
        interrupt_before: Optional[Union[All, list[str]]] = None,
        interrupt_after: Optional[Union[All, list[str]]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[str] = None,
    ) -> Run:
        """Create a cron run.

        Args:
            assistant_id: The assistant ID or graph name to use for the cron job.
                If using graph name, will default to first assistant created from that graph.
            schedule: The cron schedule to execute this job on.
            input: The input to the graph.
            metadata: Metadata to assign to the cron job runs.
            config: The configuration for the assistant.
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.

        Returns:
            Run: The cron run.

        Example Usage:

            cron_run = client.crons.create(
                assistant_id="agent",
                schedule="27 15 * * *",
                input={"messages": [{"role": "user", "content": "hello!"}]},
                metadata={"name":"my_run"},
                config={"configurable": {"model_name": "openai"}},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            )

        """  # noqa: E501
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
        """Delete a cron.

        Args:
            cron_id: The cron ID to delete.

        Returns:
            None

        Example Usage:

            await client.crons.delete(
                cron_id="cron_to_delete"
            )

        """  # noqa: E501
        await self.http.delete(f"/runs/crons/{cron_id}")

    async def search(
        self,
        *,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Cron]:
        """Get a list of cron jobs.

        Args:
            assistant_id: The assistant ID or graph name to search for.
            thread_id: the thread ID to search for.
            limit: The maximum number of results to return.
            offset: The number of results to skip.

        Returns:
            list[Cron]: The list of cron jobs returned by the search,

        Example Usage:

            cron_jobs = await client.crons.search(
                assistant_id="my_assistant_id",
                thread_id="my_thread_id",
                limit=5,
                offset=5,
            )
            print(cron_jobs)

            ----------------------------------------------------------

            [
                {
                    'cron_id': '1ef3cefa-4c09-6926-96d0-3dc97fd5e39b',
                    'assistant_id': 'my_assistant_id',
                    'thread_id': 'my_thread_id',
                    'user_id': None,
                    'payload':
                        {
                            'input': {'start_time': ''},
                            'schedule': '4 * * * *',
                            'assistant_id': 'my_assistant_id'
                        },
                    'schedule': '4 * * * *',
                    'next_run_date': '2024-07-25T17:04:00+00:00',
                    'end_time': None,
                    'created_at': '2024-07-08T06:02:23.073257+00:00',
                    'updated_at': '2024-07-08T06:02:23.073257+00:00'
                }
            ]

        """  # noqa: E501
        payload = {
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "limit": limit,
            "offset": offset,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self.http.post("/runs/crons/search", json=payload)


class StoreClient:
    """Client for interacting with the graph's shared storage.

    The Store provides a key-value storage system for persisting data across graph executions,
    allowing for stateful operations and data sharing across threads.

    Example:

        client = get_client()
        await client.store.put_item(["users", "user123"], "mem-123451342", {"name": "Alice", "score": 100})
    """

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    async def put_item(
        self,
        namespace: Sequence[str],
        /,
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
    ) -> None:
        """Store or update an item.

        Args:
            namespace: A list of strings representing the namespace path.
            key: The unique identifier for the item within the namespace.
            value: A dictionary containing the item's data.
            index: Controls search indexing - None (use defaults), False (disable), or list of field paths to index.

        Returns:
            None

        Example Usage:

            await client.store.put_item(
                ["documents", "user123"],
                key="item456",
                value={"title": "My Document", "content": "Hello World"}
            )
        """
        for label in namespace:
            if "." in label:
                raise ValueError(
                    f"Invalid namespace label '{label}'. Namespace labels cannot contain periods ('.')."
                )
        payload = {"namespace": namespace, "key": key, "value": value, "index": index}
        await self.http.put("/store/items", json=payload)

    async def get_item(self, namespace: Sequence[str], /, key: str) -> Item:
        """Retrieve a single item.

        Args:
            key: The unique identifier for the item.
            namespace: Optional list of strings representing the namespace path.

        Returns:
            Item: The retrieved item.

        Example Usage:

            item = await client.store.get_item(
                ["documents", "user123"],
                key="item456",
            )
            print(item)

            ----------------------------------------------------------------

            {
                'namespace': ['documents', 'user123'],
                'key': 'item456',
                'value': {'title': 'My Document', 'content': 'Hello World'},
                'created_at': '2024-07-30T12:00:00Z',
                'updated_at': '2024-07-30T12:00:00Z'
            }
        """
        for label in namespace:
            if "." in label:
                raise ValueError(
                    f"Invalid namespace label '{label}'. Namespace labels cannot contain periods ('.')."
                )
        return await self.http.get(
            "/store/items", params={"namespace": ".".join(namespace), "key": key}
        )

    async def delete_item(self, namespace: Sequence[str], /, key: str) -> None:
        """Delete an item.

        Args:
            key: The unique identifier for the item.
            namespace: Optional list of strings representing the namespace path.

        Returns:
            None

        Example Usage:

            await client.store.delete_item(
                ["documents", "user123"],
                key="item456",
            )
        """
        await self.http.delete(
            "/store/items", json={"namespace": namespace, "key": key}
        )

    async def search_items(
        self,
        namespace_prefix: Sequence[str],
        /,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        query: Optional[str] = None,
    ) -> SearchItemsResponse:
        """Search for items within a namespace prefix.

        Args:
            namespace_prefix: List of strings representing the namespace prefix.
            filter: Optional dictionary of key-value pairs to filter results.
            limit: Maximum number of items to return (default is 10).
            offset: Number of items to skip before returning results (default is 0).
            query: Optional query for natural language search.

        Returns:
            List[Item]: A list of items matching the search criteria.

        Example Usage:

            items = await client.store.search_items(
                ["documents"],
                filter={"author": "John Doe"},
                limit=5,
                offset=0
            )
            print(items)

            ----------------------------------------------------------------

            {
                "items": [
                    {
                        "namespace": ["documents", "user123"],
                        "key": "item789",
                        "value": {
                            "title": "Another Document",
                            "author": "John Doe"
                        },
                        "created_at": "2024-07-30T12:00:00Z",
                        "updated_at": "2024-07-30T12:00:00Z"
                    },
                    # ... additional items ...
                ]
            }
        """
        payload = {
            "namespace_prefix": namespace_prefix,
            "filter": filter,
            "limit": limit,
            "offset": offset,
            "query": query,
        }

        return await self.http.post("/store/items/search", json=_provided_vals(payload))

    async def list_namespaces(
        self,
        prefix: Optional[List[str]] = None,
        suffix: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ListNamespaceResponse:
        """List namespaces with optional match conditions.

        Args:
            prefix: Optional list of strings representing the prefix to filter namespaces.
            suffix: Optional list of strings representing the suffix to filter namespaces.
            max_depth: Optional integer specifying the maximum depth of namespaces to return.
            limit: Maximum number of namespaces to return (default is 100).
            offset: Number of namespaces to skip before returning results (default is 0).

        Returns:
            List[List[str]]: A list of namespaces matching the criteria.

        Example Usage:

            namespaces = await client.store.list_namespaces(
                prefix=["documents"],
                max_depth=3,
                limit=10,
                offset=0
            )
            print(namespaces)

            ----------------------------------------------------------------

            [
                ["documents", "user123", "reports"],
                ["documents", "user456", "invoices"],
                ...
            ]
        """
        payload = {
            "prefix": prefix,
            "suffix": suffix,
            "max_depth": max_depth,
            "limit": limit,
            "offset": offset,
        }
        return await self.http.post("/store/namespaces", json=_provided_vals(payload))


def get_sync_client(
    *,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> SyncLangGraphClient:
    """Get a synchronous LangGraphClient instance.

    Args:
        url: The URL of the LangGraph API.
        api_key: The API key. If not provided, it will be read from the environment.
            Precedence:
                1. explicit argument
                2. LANGGRAPH_API_KEY
                3. LANGSMITH_API_KEY
                4. LANGCHAIN_API_KEY
        headers: Optional custom headers
    Returns:
        SyncLangGraphClient: The top-level synchronous client for accessing AssistantsClient,
        ThreadsClient, RunsClient, and CronClient.

    Example:

        from langgraph_sdk import get_sync_client

        # get top-level synchronous LangGraphClient
        client = get_sync_client(url="http://localhost:8123")

        # example usage: client.<model>.<method_name>()
        assistant = client.assistants.get(assistant_id="some_uuid")
    """

    if url is None:
        url = "http://localhost:8123"

    transport = httpx.HTTPTransport(retries=5)
    client = httpx.Client(
        base_url=url,
        transport=transport,
        timeout=httpx.Timeout(connect=5, read=300, write=300, pool=5),
        headers=get_headers(api_key, headers),
    )
    return SyncLangGraphClient(client)


class SyncLangGraphClient:
    """Synchronous client for interacting with the LangGraph API.

    This class provides synchronous access to LangGraph API endpoints for managing
    assistants, threads, runs, cron jobs, and data storage.

    Example:

        client = get_sync_client()
        assistant = client.assistants.get("asst_123")
    """

    def __init__(self, client: httpx.Client) -> None:
        self.http = SyncHttpClient(client)
        self.assistants = SyncAssistantsClient(self.http)
        self.threads = SyncThreadsClient(self.http)
        self.runs = SyncRunsClient(self.http)
        self.crons = SyncCronClient(self.http)
        self.store = SyncStoreClient(self.http)


class SyncHttpClient:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def get(self, path: str, *, params: Optional[QueryParamTypes] = None) -> Any:
        """Send a GET request."""
        r = self.client.get(path, params=params)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = r.read().decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from langgraph-api: {body}", exc_info=e)
            raise e
        return decode_json(r)

    def post(self, path: str, *, json: Optional[dict]) -> Any:
        """Send a POST request."""
        if json is not None:
            headers, content = encode_json(json)
        else:
            headers, content = {}, b""
        r = self.client.post(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = r.read().decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from langgraph-api: {body}", exc_info=e)
            raise e
        return decode_json(r)

    def put(self, path: str, *, json: dict) -> Any:
        """Send a PUT request."""
        headers, content = encode_json(json)
        r = self.client.put(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = r.read().decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from langgraph-api: {body}", exc_info=e)
            raise e
        return decode_json(r)

    def patch(self, path: str, *, json: dict) -> Any:
        """Send a PATCH request."""
        headers, content = encode_json(json)
        r = self.client.patch(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = r.read().decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from langgraph-api: {body}", exc_info=e)
            raise e
        return decode_json(r)

    def delete(self, path: str, *, json: Optional[Any] = None) -> None:
        """Send a DELETE request."""
        r = self.client.request("DELETE", path, json=json)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = r.read().decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from langgraph-api: {body}", exc_info=e)
            raise e

    def stream(
        self,
        path: str,
        method: str,
        *,
        json: Optional[dict] = None,
        params: Optional[QueryParamTypes] = None,
    ) -> Iterator[StreamPart]:
        """Stream the results of a request using SSE."""
        headers, content = encode_json(json)
        with self.client.stream(
            method, path, headers=headers, content=content, params=params
        ) as res:
            # check status
            try:
                res.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = (res.read()).decode()
                if sys.version_info >= (3, 11):
                    e.add_note(body)
                else:
                    logger.error(f"Error from langgraph-api: {body}", exc_info=e)
                raise e
            # check content type
            content_type = res.headers.get("content-type", "").partition(";")[0]
            if "text/event-stream" not in content_type:
                raise httpx.TransportError(
                    "Expected response header Content-Type to contain 'text/event-stream', "
                    f"got {content_type!r}"
                )
            # parse SSE
            decoder = SSEDecoder()
            for line in iter_lines_raw(res):
                sse = decoder.decode(line.rstrip(b"\n"))
                if sse is not None:
                    yield sse


def encode_json(json: Any) -> tuple[dict[str, str], bytes]:
    body = orjson.dumps(
        json,
        orjson_default,
        orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, body


def decode_json(r: httpx.Response) -> Any:
    body = r.read()
    return orjson.loads(body if body else None)


class SyncAssistantsClient:
    """Client for managing assistants in LangGraph synchronously.

    This class provides methods to interact with assistants, which are versioned configurations of your graph.

    Example:

        client = get_client()
        assistant = client.assistants.get("assistant_id_123")
    """

    def __init__(self, http: SyncHttpClient) -> None:
        self.http = http

    def get(self, assistant_id: str) -> Assistant:
        """Get an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get.

        Returns:
            Assistant: Assistant Object.

        Example Usage:

            assistant = client.assistants.get(
                assistant_id="my_assistant_id"
            )
            print(assistant)

            ----------------------------------------------------

            {
                'assistant_id': 'my_assistant_id',
                'graph_id': 'agent',
                'created_at': '2024-06-25T17:10:33.109781+00:00',
                'updated_at': '2024-06-25T17:10:33.109781+00:00',
                'config': {},
                'metadata': {'created_by': 'system'}
            }

        """  # noqa: E501
        return self.http.get(f"/assistants/{assistant_id}")

    def get_graph(
        self, assistant_id: str, *, xray: Union[int, bool] = False
    ) -> dict[str, list[dict[str, Any]]]:
        """Get the graph of an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get the graph of.
            xray: Include graph representation of subgraphs. If an integer value is provided, only subgraphs with a depth less than or equal to the value will be included.

        Returns:
            Graph: The graph information for the assistant in JSON format.

        Example Usage:

            graph_info = client.assistants.get_graph(
                assistant_id="my_assistant_id"
            )
            print(graph_info)

            --------------------------------------------------------------------------------------------------------------------------

            {
                'nodes':
                    [
                        {'id': '__start__', 'type': 'schema', 'data': '__start__'},
                        {'id': '__end__', 'type': 'schema', 'data': '__end__'},
                        {'id': 'agent','type': 'runnable','data': {'id': ['langgraph', 'utils', 'RunnableCallable'],'name': 'agent'}},
                    ],
                'edges':
                    [
                        {'source': '__start__', 'target': 'agent'},
                        {'source': 'agent','target': '__end__'}
                    ]
            }


        """  # noqa: E501
        return self.http.get(f"/assistants/{assistant_id}/graph", params={"xray": xray})

    def get_schemas(self, assistant_id: str) -> GraphSchema:
        """Get the schemas of an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get the schema of.

        Returns:
            GraphSchema: The graph schema for the assistant.

        Example Usage:

            schema = client.assistants.get_schemas(
                assistant_id="my_assistant_id"
            )
            print(schema)

            ----------------------------------------------------------------------------------------------------------------------------

            {
                'graph_id': 'agent',
                'state_schema':
                    {
                        'title': 'LangGraphInput',
                        '$ref': '#/definitions/AgentState',
                        'definitions':
                            {
                                'BaseMessage':
                                    {
                                        'title': 'BaseMessage',
                                        'description': 'Base abstract Message class. Messages are the inputs and outputs of ChatModels.',
                                        'type': 'object',
                                        'properties':
                                            {
                                             'content':
                                                {
                                                    'title': 'Content',
                                                    'anyOf': [
                                                        {'type': 'string'},
                                                        {'type': 'array','items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}
                                                    ]
                                                },
                                            'additional_kwargs':
                                                {
                                                    'title': 'Additional Kwargs',
                                                    'type': 'object'
                                                },
                                            'response_metadata':
                                                {
                                                    'title': 'Response Metadata',
                                                    'type': 'object'
                                                },
                                            'type':
                                                {
                                                    'title': 'Type',
                                                    'type': 'string'
                                                },
                                            'name':
                                                {
                                                    'title': 'Name',
                                                    'type': 'string'
                                                },
                                            'id':
                                                {
                                                    'title': 'Id',
                                                    'type': 'string'
                                                }
                                            },
                                        'required': ['content', 'type']
                                    },
                                'AgentState':
                                    {
                                        'title': 'AgentState',
                                        'type': 'object',
                                        'properties':
                                            {
                                                'messages':
                                                    {
                                                        'title': 'Messages',
                                                        'type': 'array',
                                                        'items': {'$ref': '#/definitions/BaseMessage'}
                                                    }
                                            },
                                        'required': ['messages']
                                    }
                            }
                    },
                'config_schema':
                    {
                        'title': 'Configurable',
                        'type': 'object',
                        'properties':
                            {
                                'model_name':
                                    {
                                        'title': 'Model Name',
                                        'enum': ['anthropic', 'openai'],
                                        'type': 'string'
                                    }
                            }
                    }
            }

        """  # noqa: E501
        return self.http.get(f"/assistants/{assistant_id}/schemas")

    def get_subgraphs(
        self, assistant_id: str, namespace: Optional[str] = None, recurse: bool = False
    ) -> Subgraphs:
        """Get the schemas of an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get the schema of.

        Returns:
            Subgraphs: The graph schema for the assistant.

        """  # noqa: E501
        if namespace is not None:
            return self.http.get(
                f"/assistants/{assistant_id}/subgraphs/{namespace}",
                params={"recurse": recurse},
            )
        else:
            return self.http.get(
                f"/assistants/{assistant_id}/subgraphs",
                params={"recurse": recurse},
            )

    def create(
        self,
        graph_id: Optional[str],
        config: Optional[Config] = None,
        *,
        metadata: Json = None,
        assistant_id: Optional[str] = None,
        if_exists: Optional[OnConflictBehavior] = None,
        name: Optional[str] = None,
    ) -> Assistant:
        """Create a new assistant.

        Useful when graph is configurable and you want to create different assistants based on different configurations.

        Args:
            graph_id: The ID of the graph the assistant should use. The graph ID is normally set in your langgraph.json configuration.
            config: Configuration to use for the graph.
            metadata: Metadata to add to assistant.
            assistant_id: Assistant ID to use, will default to a random UUID if not provided.
            if_exists: How to handle duplicate creation. Defaults to 'raise' under the hood.
                Must be either 'raise' (raise error if duplicate), or 'do_nothing' (return existing assistant).
            name: The name of the assistant. Defaults to 'Untitled' under the hood.

        Returns:
            Assistant: The created assistant.

        Example Usage:

            assistant = client.assistants.create(
                graph_id="agent",
                config={"configurable": {"model_name": "openai"}},
                metadata={"number":1},
                assistant_id="my-assistant-id",
                if_exists="do_nothing",
                name="my_name"
            )
        """  # noqa: E501
        payload: Dict[str, Any] = {
            "graph_id": graph_id,
        }
        if config:
            payload["config"] = config
        if metadata:
            payload["metadata"] = metadata
        if assistant_id:
            payload["assistant_id"] = assistant_id
        if if_exists:
            payload["if_exists"] = if_exists
        if name:
            payload["name"] = name
        return self.http.post("/assistants", json=payload)

    def update(
        self,
        assistant_id: str,
        *,
        graph_id: Optional[str] = None,
        config: Optional[Config] = None,
        metadata: Json = None,
        name: Optional[str] = None,
    ) -> Assistant:
        """Update an assistant.

        Use this to point to a different graph, update the configuration, or change the metadata of an assistant.

        Args:
            assistant_id: Assistant to update.
            graph_id: The ID of the graph the assistant should use.
                The graph ID is normally set in your langgraph.json configuration. If None, assistant will keep pointing to same graph.
            config: Configuration to use for the graph.
            metadata: Metadata to merge with existing assistant metadata.
            name: The new name for the assistant.

        Returns:
            Assistant: The updated assistant.

        Example Usage:

            assistant = client.assistants.update(
                assistant_id='e280dad7-8618-443f-87f1-8e41841c180f',
                graph_id="other-graph",
                config={"configurable": {"model_name": "anthropic"}},
                metadata={"number":2}
            )

        """  # noqa: E501
        payload: Dict[str, Any] = {}
        if graph_id:
            payload["graph_id"] = graph_id
        if config:
            payload["config"] = config
        if metadata:
            payload["metadata"] = metadata
        if name:
            payload["name"] = name
        return self.http.patch(
            f"/assistants/{assistant_id}",
            json=payload,
        )

    def delete(
        self,
        assistant_id: str,
    ) -> None:
        """Delete an assistant.

        Args:
            assistant_id: The assistant ID to delete.

        Returns:
            None

        Example Usage:

            client.assistants.delete(
                assistant_id="my_assistant_id"
            )

        """  # noqa: E501
        self.http.delete(f"/assistants/{assistant_id}")

    def search(
        self,
        *,
        metadata: Json = None,
        graph_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Assistant]:
        """Search for assistants.

        Args:
            metadata: Metadata to filter by. Exact match filter for each KV pair.
            graph_id: The ID of the graph to filter by.
                The graph ID is normally set in your langgraph.json configuration.
            limit: The maximum number of results to return.
            offset: The number of results to skip.

        Returns:
            list[Assistant]: A list of assistants.

        Example Usage:

            assistants = client.assistants.search(
                metadata = {"name":"my_name"},
                graph_id="my_graph_id",
                limit=5,
                offset=5
            )
        """
        payload: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if metadata:
            payload["metadata"] = metadata
        if graph_id:
            payload["graph_id"] = graph_id
        return self.http.post(
            "/assistants/search",
            json=payload,
        )

    def get_versions(
        self,
        assistant_id: str,
        metadata: Json = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[AssistantVersion]:
        """List all versions of an assistant.

        Args:
            assistant_id: The assistant ID to get versions for.
            metadata: Metadata to filter versions by. Exact match filter for each KV pair.
            limit: The maximum number of versions to return.
            offset: The number of versions to skip.

        Returns:
            list[Assistant]: A list of assistants.

        Example Usage:

            assistant_versions = await client.assistants.get_versions(
                assistant_id="my_assistant_id"
            )

        """  # noqa: E501

        payload: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if metadata:
            payload["metadata"] = metadata
        return self.http.post(f"/assistants/{assistant_id}/versions", json=payload)

    def set_latest(self, assistant_id: str, version: int) -> Assistant:
        """Change the version of an assistant.

        Args:
            assistant_id: The assistant ID to delete.
            version: The version to change to.

        Returns:
            Assistant: Assistant Object.

        Example Usage:

            new_version_assistant = await client.assistants.set_latest(
                assistant_id="my_assistant_id",
                version=3
            )

        """  # noqa: E501

        payload: Dict[str, Any] = {"version": version}

        return self.http.post(f"/assistants/{assistant_id}/latest", json=payload)


class SyncThreadsClient:
    """Synchronous client for managing threads in LangGraph.

    This class provides methods to create, retrieve, and manage threads,
    which represent conversations or stateful interactions.

    Example:

        client = get_sync_client()
        thread = client.threads.create(metadata={"user_id": "123"})
    """

    def __init__(self, http: SyncHttpClient) -> None:
        self.http = http

    def get(self, thread_id: str) -> Thread:
        """Get a thread by ID.

        Args:
            thread_id: The ID of the thread to get.

        Returns:
            Thread: Thread object.

        Example Usage:

            thread = client.threads.get(
                thread_id="my_thread_id"
            )
            print(thread)

            -----------------------------------------------------

            {
                'thread_id': 'my_thread_id',
                'created_at': '2024-07-18T18:35:15.540834+00:00',
                'updated_at': '2024-07-18T18:35:15.540834+00:00',
                'metadata': {'graph_id': 'agent'}
            }

        """  # noqa: E501

        return self.http.get(f"/threads/{thread_id}")

    def create(
        self,
        *,
        metadata: Json = None,
        thread_id: Optional[str] = None,
        if_exists: Optional[OnConflictBehavior] = None,
    ) -> Thread:
        """Create a new thread.

        Args:
            metadata: Metadata to add to thread.
            thread_id: ID of thread.
                If None, ID will be a randomly generated UUID.
            if_exists: How to handle duplicate creation. Defaults to 'raise' under the hood.
                Must be either 'raise' (raise error if duplicate), or 'do_nothing' (return existing thread).

        Returns:
            Thread: The created thread.

        Example Usage:

            thread = client.threads.create(
                metadata={"number":1},
                thread_id="my-thread-id",
                if_exists="raise"
            )
        """  # noqa: E501
        payload: Dict[str, Any] = {}
        if thread_id:
            payload["thread_id"] = thread_id
        if metadata:
            payload["metadata"] = metadata
        if if_exists:
            payload["if_exists"] = if_exists
        return self.http.post("/threads", json=payload)

    def update(self, thread_id: str, *, metadata: dict[str, Any]) -> Thread:
        """Update a thread.

        Args:
            thread_id: ID of thread to update.
            metadata: Metadata to merge with existing thread metadata.

        Returns:
            Thread: The created thread.

        Example Usage:

            thread = client.threads.update(
                thread_id="my-thread-id",
                metadata={"number":1},
            )
        """  # noqa: E501
        return self.http.patch(f"/threads/{thread_id}", json={"metadata": metadata})

    def delete(self, thread_id: str) -> None:
        """Delete a thread.

        Args:
            thread_id: The ID of the thread to delete.

        Returns:
            None

        Example Usage:

            client.threads.delete(
                thread_id="my_thread_id"
            )

        """  # noqa: E501
        self.http.delete(f"/threads/{thread_id}")

    def search(
        self,
        *,
        metadata: Json = None,
        values: Json = None,
        status: Optional[ThreadStatus] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Thread]:
        """Search for threads.

        Args:
            metadata: Thread metadata to filter on.
            values: State values to filter on.
            status: Thread status to filter on.
                Must be one of 'idle', 'busy', 'interrupted' or 'error'.
            limit: Limit on number of threads to return.
            offset: Offset in threads table to start search from.

        Returns:
            list[Thread]: List of the threads matching the search parameters.

        Example Usage:

            threads = client.threads.search(
                metadata={"number":1},
                status="interrupted",
                limit=15,
                offset=5
            )

        """  # noqa: E501
        payload: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if metadata:
            payload["metadata"] = metadata
        if values:
            payload["values"] = values
        if status:
            payload["status"] = status
        return self.http.post(
            "/threads/search",
            json=payload,
        )

    def copy(self, thread_id: str) -> None:
        """Copy a thread.

        Args:
            thread_id: The ID of the thread to copy.

        Returns:
            None

        Example Usage:

            client.threads.copy(
                thread_id="my_thread_id"
            )

        """  # noqa: E501
        return self.http.post(f"/threads/{thread_id}/copy", json=None)

    def get_state(
        self,
        thread_id: str,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,  # deprecated
        *,
        subgraphs: bool = False,
    ) -> ThreadState:
        """Get the state of a thread.

        Args:
            thread_id: The ID of the thread to get the state of.
            checkpoint: The checkpoint to get the state of.
            subgraphs: Include subgraphs states.

        Returns:
            ThreadState: the thread of the state.

        Example Usage:

            thread_state = client.threads.get_state(
                thread_id="my_thread_id",
                checkpoint_id="my_checkpoint_id"
            )
            print(thread_state)

            ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

            {
                'values': {
                    'messages': [
                        {
                            'content': 'how are you?',
                            'additional_kwargs': {},
                            'response_metadata': {},
                            'type': 'human',
                            'name': None,
                            'id': 'fe0a5778-cfe9-42ee-b807-0adaa1873c10',
                            'example': False
                        },
                        {
                            'content': "I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, honest, and harmless.",
                            'additional_kwargs': {},
                            'response_metadata': {},
                            'type': 'ai',
                            'name': None,
                            'id': 'run-159b782c-b679-4830-83c6-cef87798fe8b',
                            'example': False,
                            'tool_calls': [],
                            'invalid_tool_calls': [],
                            'usage_metadata': None
                        }
                    ]
                },
                'next': [],
                'checkpoint':
                    {
                        'thread_id': 'e2496803-ecd5-4e0c-a779-3226296181c2',
                        'checkpoint_ns': '',
                        'checkpoint_id': '1ef4a9b8-e6fb-67b1-8001-abd5184439d1'
                    }
                'metadata':
                    {
                        'step': 1,
                        'run_id': '1ef4a9b8-d7da-679a-a45a-872054341df2',
                        'source': 'loop',
                        'writes':
                            {
                                'agent':
                                    {
                                        'messages': [
                                            {
                                                'id': 'run-159b782c-b679-4830-83c6-cef87798fe8b',
                                                'name': None,
                                                'type': 'ai',
                                                'content': "I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, honest, and harmless.",
                                                'example': False,
                                                'tool_calls': [],
                                                'usage_metadata': None,
                                                'additional_kwargs': {},
                                                'response_metadata': {},
                                                'invalid_tool_calls': []
                                            }
                                        ]
                                    }
                            },
                'user_id': None,
                'graph_id': 'agent',
                'thread_id': 'e2496803-ecd5-4e0c-a779-3226296181c2',
                'created_by': 'system',
                'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'},
                'created_at': '2024-07-25T15:35:44.184703+00:00',
                'parent_config':
                    {
                        'thread_id': 'e2496803-ecd5-4e0c-a779-3226296181c2',
                        'checkpoint_ns': '',
                        'checkpoint_id': '1ef4a9b8-d80d-6fa7-8000-9300467fad0f'
                    }
            }

        """  # noqa: E501
        if checkpoint:
            return self.http.post(
                f"/threads/{thread_id}/state/checkpoint",
                json={"checkpoint": checkpoint, "subgraphs": subgraphs},
            )
        elif checkpoint_id:
            return self.http.get(
                f"/threads/{thread_id}/state/{checkpoint_id}",
                params={"subgraphs": subgraphs},
            )
        else:
            return self.http.get(
                f"/threads/{thread_id}/state",
                params={"subgraphs": subgraphs},
            )

    def update_state(
        self,
        thread_id: str,
        values: Optional[Union[dict, Sequence[dict]]],
        *,
        as_node: Optional[str] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,  # deprecated
    ) -> ThreadUpdateStateResponse:
        """Update the state of a thread.

        Args:
            thread_id: The ID of the thread to update.
            values: The values to update the state with.
            as_node: Update the state as if this node had just executed.
            checkpoint: The checkpoint to update the state of.

        Returns:
            ThreadUpdateStateResponse: Response after updating a thread's state.

        Example Usage:

            response = client.threads.update_state(
                thread_id="my_thread_id",
                values={"messages":[{"role": "user", "content": "hello!"}]},
                as_node="my_node",
            )
            print(response)

            ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

            {
                'checkpoint': {
                    'thread_id': 'e2496803-ecd5-4e0c-a779-3226296181c2',
                    'checkpoint_ns': '',
                    'checkpoint_id': '1ef4a9b8-e6fb-67b1-8001-abd5184439d1',
                    'checkpoint_map': {}
                }
            }

        """  # noqa: E501
        payload: Dict[str, Any] = {
            "values": values,
        }
        if checkpoint_id:
            payload["checkpoint_id"] = checkpoint_id
        if checkpoint:
            payload["checkpoint"] = checkpoint
        if as_node:
            payload["as_node"] = as_node
        return self.http.post(f"/threads/{thread_id}/state", json=payload)

    def get_history(
        self,
        thread_id: str,
        *,
        limit: int = 10,
        before: Optional[str | Checkpoint] = None,
        metadata: Optional[dict] = None,
        checkpoint: Optional[Checkpoint] = None,
    ) -> list[ThreadState]:
        """Get the state history of a thread.

        Args:
            thread_id: The ID of the thread to get the state history for.
            checkpoint: Return states for this subgraph. If empty defaults to root.
            limit: The maximum number of states to return.
            before: Return states before this checkpoint.
            metadata: Filter states by metadata key-value pairs.

        Returns:
            list[ThreadState]: the state history of the thread.

        Example Usage:

            thread_state = client.threads.get_history(
                thread_id="my_thread_id",
                limit=5,
                before="my_timestamp",
                metadata={"name":"my_name"}
            )

        """  # noqa: E501
        payload: Dict[str, Any] = {
            "limit": limit,
        }
        if before:
            payload["before"] = before
        if metadata:
            payload["metadata"] = metadata
        if checkpoint:
            payload["checkpoint"] = checkpoint
        return self.http.post(f"/threads/{thread_id}/history", json=payload)


class SyncRunsClient:
    """Synchronous client for managing runs in LangGraph.

    This class provides methods to create, retrieve, and manage runs, which represent
    individual executions of graphs.

    Example:

        client = get_sync_client()
        run = client.runs.create(thread_id="thread_123", assistant_id="asst_456")
    """

    def __init__(self, http: SyncHttpClient) -> None:
        self.http = http

    @overload
    def stream(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        feedback_keys: Optional[Sequence[str]] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> Iterator[StreamPart]: ...

    @overload
    def stream(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        feedback_keys: Optional[Sequence[str]] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        if_not_exists: Optional[IfNotExists] = None,
        webhook: Optional[str] = None,
        after_seconds: Optional[int] = None,
    ) -> Iterator[StreamPart]: ...

    def stream(
        self,
        thread_id: Optional[str],
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        feedback_keys: Optional[Sequence[str]] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> Iterator[StreamPart]:
        """Create a run and stream the results.

        Args:
            thread_id: the thread ID to assign to the thread.
                If None will create a stateless run.
            assistant_id: The assistant ID or graph name to stream from.
                If using graph name, will default to first assistant created from that graph.
            input: The input to the graph.
            command: The command to execute.
            stream_mode: The stream mode(s) to use.
            stream_subgraphs: Whether to stream output from subgraphs.
            metadata: Metadata to assign to the run.
            config: The configuration for the assistant.
            checkpoint: The checkpoint to resume from.
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            feedback_keys: Feedback keys to assign to run.
            on_disconnect: The disconnect mode to use.
                Must be one of 'cancel' or 'continue'.
            on_completion: Whether to delete or keep the thread created for a stateless run.
                Must be one of 'delete' or 'keep'.
            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            if_not_exists: How to handle missing thread. Defaults to 'reject'.
                Must be either 'reject' (raise error if missing), or 'create' (create new thread).
            after_seconds: The number of seconds to wait before starting the run.
                Use to schedule future runs.

        Returns:
            Iterator[StreamPart]: Iterator of stream results.

        Example Usage:

            async for chunk in client.runs.stream(
                thread_id=None,
                assistant_id="agent",
                input={"messages": [{"role": "user", "content": "how are you?"}]},
                stream_mode=["values","debug"],
                metadata={"name":"my_run"},
                config={"configurable": {"model_name": "anthropic"}},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                feedback_keys=["my_feedback_key_1","my_feedback_key_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            ):
                print(chunk)

            ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            StreamPart(event='metadata', data={'run_id': '1ef4a9b8-d7da-679a-a45a-872054341df2'})
            StreamPart(event='values', data={'messages': [{'content': 'how are you?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'fe0a5778-cfe9-42ee-b807-0adaa1873c10', 'example': False}]})
            StreamPart(event='values', data={'messages': [{'content': 'how are you?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'fe0a5778-cfe9-42ee-b807-0adaa1873c10', 'example': False}, {'content': "I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, honest, and harmless.", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-159b782c-b679-4830-83c6-cef87798fe8b', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]})
            StreamPart(event='end', data=None)

        """  # noqa: E501
        payload = {
            "input": input,
            "command": {k: v for k, v in command.items() if v is not None}
            if command
            else None,
            "config": config,
            "metadata": metadata,
            "stream_mode": stream_mode,
            "stream_subgraphs": stream_subgraphs,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "feedback_keys": feedback_keys,
            "webhook": webhook,
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "multitask_strategy": multitask_strategy,
            "if_not_exists": if_not_exists,
            "on_disconnect": on_disconnect,
            "on_completion": on_completion,
            "after_seconds": after_seconds,
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
    def create(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> Run: ...

    @overload
    def create(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> Run: ...

    def create(
        self,
        thread_id: Optional[str],
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        stream_subgraphs: bool = False,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> Run:
        """Create a background run.

        Args:
            thread_id: the thread ID to assign to the thread.
                If None will create a stateless run.
            assistant_id: The assistant ID or graph name to stream from.
                If using graph name, will default to first assistant created from that graph.
            input: The input to the graph.
            command: The command to execute.
            stream_mode: The stream mode(s) to use.
            stream_subgraphs: Whether to stream output from subgraphs.
            metadata: Metadata to assign to the run.
            config: The configuration for the assistant.
            checkpoint: The checkpoint to resume from.
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            on_completion: Whether to delete or keep the thread created for a stateless run.
                Must be one of 'delete' or 'keep'.
            if_not_exists: How to handle missing thread. Defaults to 'reject'.
                Must be either 'reject' (raise error if missing), or 'create' (create new thread).
            after_seconds: The number of seconds to wait before starting the run.
                Use to schedule future runs.

        Returns:
            Run: The created background run.

        Example Usage:

            background_run = client.runs.create(
                thread_id="my_thread_id",
                assistant_id="my_assistant_id",
                input={"messages": [{"role": "user", "content": "hello!"}]},
                metadata={"name":"my_run"},
                config={"configurable": {"model_name": "openai"}},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            )
            print(background_run)

            --------------------------------------------------------------------------------

            {
                'run_id': 'my_run_id',
                'thread_id': 'my_thread_id',
                'assistant_id': 'my_assistant_id',
                'created_at': '2024-07-25T15:35:42.598503+00:00',
                'updated_at': '2024-07-25T15:35:42.598503+00:00',
                'metadata': {},
                'status': 'pending',
                'kwargs':
                    {
                        'input':
                            {
                                'messages': [
                                    {
                                        'role': 'user',
                                        'content': 'how are you?'
                                    }
                                ]
                            },
                        'config':
                            {
                                'metadata':
                                    {
                                        'created_by': 'system'
                                    },
                                'configurable':
                                    {
                                        'run_id': 'my_run_id',
                                        'user_id': None,
                                        'graph_id': 'agent',
                                        'thread_id': 'my_thread_id',
                                        'checkpoint_id': None,
                                        'model_name': "openai",
                                        'assistant_id': 'my_assistant_id'
                                    }
                            },
                        'webhook': "https://my.fake.webhook.com",
                        'temporary': False,
                        'stream_mode': ['values'],
                        'feedback_keys': None,
                        'interrupt_after': ["node_to_stop_after_1","node_to_stop_after_2"],
                        'interrupt_before': ["node_to_stop_before_1","node_to_stop_before_2"]
                    },
                'multitask_strategy': 'interrupt'
            }

        """  # noqa: E501
        payload = {
            "input": input,
            "command": {k: v for k, v in command.items() if v is not None}
            if command
            else None,
            "stream_mode": stream_mode,
            "stream_subgraphs": stream_subgraphs,
            "config": config,
            "metadata": metadata,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "multitask_strategy": multitask_strategy,
            "if_not_exists": if_not_exists,
            "on_completion": on_completion,
            "after_seconds": after_seconds,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if thread_id:
            return self.http.post(f"/threads/{thread_id}/runs", json=payload)
        else:
            return self.http.post("/runs", json=payload)

    def create_batch(self, payloads: list[RunCreate]) -> list[Run]:
        """Create a batch of stateless background runs."""

        def filter_payload(payload: RunCreate):
            return {k: v for k, v in payload.items() if v is not None}

        payloads = [filter_payload(payload) for payload in payloads]
        return self.http.post("/runs/batch", json=payloads)

    @overload
    def wait(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> Union[list[dict], dict[str, Any]]: ...

    @overload
    def wait(
        self,
        thread_id: None,
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> Union[list[dict], dict[str, Any]]: ...

    def wait(
        self,
        thread_id: Optional[str],
        assistant_id: str,
        *,
        input: Optional[dict] = None,
        command: Optional[Command] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        checkpoint: Optional[Checkpoint] = None,
        checkpoint_id: Optional[str] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        webhook: Optional[str] = None,
        on_disconnect: Optional[DisconnectMode] = None,
        on_completion: Optional[OnCompletionBehavior] = None,
        multitask_strategy: Optional[MultitaskStrategy] = None,
        if_not_exists: Optional[IfNotExists] = None,
        after_seconds: Optional[int] = None,
    ) -> Union[list[dict], dict[str, Any]]:
        """Create a run, wait until it finishes and return the final state.

        Args:
            thread_id: the thread ID to create the run on.
                If None will create a stateless run.
            assistant_id: The assistant ID or graph name to run.
                If using graph name, will default to first assistant created from that graph.
            input: The input to the graph.
            command: The command to execute.
            metadata: Metadata to assign to the run.
            config: The configuration for the assistant.
            checkpoint: The checkpoint to resume from.
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            webhook: Webhook to call after LangGraph API call is done.
            on_disconnect: The disconnect mode to use.
                Must be one of 'cancel' or 'continue'.
            on_completion: Whether to delete or keep the thread created for a stateless run.
                Must be one of 'delete' or 'keep'.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.
            if_not_exists: How to handle missing thread. Defaults to 'reject'.
                Must be either 'reject' (raise error if missing), or 'create' (create new thread).
            after_seconds: The number of seconds to wait before starting the run.
                Use to schedule future runs.

        Returns:
            Union[list[dict], dict[str, Any]]: The output of the run.

        Example Usage:

            final_state_of_run = client.runs.wait(
                thread_id=None,
                assistant_id="agent",
                input={"messages": [{"role": "user", "content": "how are you?"}]},
                metadata={"name":"my_run"},
                config={"configurable": {"model_name": "anthropic"}},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            )
            print(final_state_of_run)

            -------------------------------------------------------------------------------------------------------------------------------------------

            {
                'messages': [
                    {
                        'content': 'how are you?',
                        'additional_kwargs': {},
                        'response_metadata': {},
                        'type': 'human',
                        'name': None,
                        'id': 'f51a862c-62fe-4866-863b-b0863e8ad78a',
                        'example': False
                    },
                    {
                        'content': "I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, honest, and harmless.",
                        'additional_kwargs': {},
                        'response_metadata': {},
                        'type': 'ai',
                        'name': None,
                        'id': 'run-bf1cd3c6-768f-4c16-b62d-ba6f17ad8b36',
                        'example': False,
                        'tool_calls': [],
                        'invalid_tool_calls': [],
                        'usage_metadata': None
                    }
                ]
            }

        """  # noqa: E501
        payload = {
            "input": input,
            "command": {k: v for k, v in command.items() if v is not None}
            if command
            else None,
            "config": config,
            "metadata": metadata,
            "assistant_id": assistant_id,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "webhook": webhook,
            "checkpoint": checkpoint,
            "checkpoint_id": checkpoint_id,
            "multitask_strategy": multitask_strategy,
            "if_not_exists": if_not_exists,
            "on_disconnect": on_disconnect,
            "on_completion": on_completion,
            "after_seconds": after_seconds,
        }
        endpoint = (
            f"/threads/{thread_id}/runs/wait" if thread_id is not None else "/runs/wait"
        )
        return self.http.post(
            endpoint, json={k: v for k, v in payload.items() if v is not None}
        )

    def list(self, thread_id: str, *, limit: int = 10, offset: int = 0) -> List[Run]:
        """List runs.

        Args:
            thread_id: The thread ID to list runs for.
            limit: The maximum number of results to return.
            offset: The number of results to skip.

        Returns:
            List[Run]: The runs for the thread.

        Example Usage:

            client.runs.delete(
                thread_id="thread_id_to_delete",
                limit=5,
                offset=5,
            )

        """  # noqa: E501
        return self.http.get(f"/threads/{thread_id}/runs?limit={limit}&offset={offset}")

    def get(self, thread_id: str, run_id: str) -> Run:
        """Get a run.

        Args:
            thread_id: The thread ID to get.
            run_id: The run ID to get.

        Returns:
            Run: Run object.

        Example Usage:

            run = client.runs.get(
                thread_id="thread_id_to_delete",
                run_id="run_id_to_delete",
            )

        """  # noqa: E501

        return self.http.get(f"/threads/{thread_id}/runs/{run_id}")

    def cancel(
        self,
        thread_id: str,
        run_id: str,
        *,
        wait: bool = False,
        action: CancelAction = "interrupt",
    ) -> None:
        """Get a run.

        Args:
            thread_id: The thread ID to cancel.
            run_id: The run ID to cancel.
            wait: Whether to wait until run has completed.
            action: Action to take when cancelling the run. Possible values
                are `interrupt` or `rollback`. Default is `interrupt`.

        Returns:
            None

        Example Usage:

            client.runs.cancel(
                thread_id="thread_id_to_cancel",
                run_id="run_id_to_cancel",
                wait=True,
                action="interrupt"
            )

        """  # noqa: E501
        return self.http.post(
            f"/threads/{thread_id}/runs/{run_id}/cancel?wait={1 if wait else 0}&action={action}",
            json=None,
        )

    def join(self, thread_id: str, run_id: str) -> dict:
        """Block until a run is done. Returns the final state of the thread.

        Args:
            thread_id: The thread ID to join.
            run_id: The run ID to join.

        Returns:
            None

        Example Usage:

            client.runs.join(
                thread_id="thread_id_to_join",
                run_id="run_id_to_join"
            )

        """  # noqa: E501
        return self.http.get(f"/threads/{thread_id}/runs/{run_id}/join")

    def join_stream(self, thread_id: str, run_id: str) -> Iterator[StreamPart]:
        """Stream output from a run in real-time, until the run is done.
        Output is not buffered, so any output produced before this call will
        not be received here.

        Args:
            thread_id: The thread ID to join.
            run_id: The run ID to join.

        Returns:
            None

        Example Usage:

            client.runs.join_stream(
                thread_id="thread_id_to_join",
                run_id="run_id_to_join"
            )

        """  # noqa: E501
        return self.http.stream(f"/threads/{thread_id}/runs/{run_id}/stream", "GET")

    def delete(self, thread_id: str, run_id: str) -> None:
        """Delete a run.

        Args:
            thread_id: The thread ID to delete.
            run_id: The run ID to delete.

        Returns:
            None

        Example Usage:

            client.runs.delete(
                thread_id="thread_id_to_delete",
                run_id="run_id_to_delete"
            )

        """  # noqa: E501
        self.http.delete(f"/threads/{thread_id}/runs/{run_id}")


class SyncCronClient:
    """Synchronous client for managing cron jobs in LangGraph.

    This class provides methods to create and manage scheduled tasks (cron jobs) for automated graph executions.

    Example:

        client = get_sync_client()
        cron_job = client.crons.create_for_thread(thread_id="thread_123", assistant_id="asst_456", schedule="0 * * * *")
    """

    def __init__(self, http_client: SyncHttpClient) -> None:
        self.http = http_client

    def create_for_thread(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        schedule: str,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[Union[All, list[str]]] = None,
        interrupt_after: Optional[Union[All, list[str]]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[str] = None,
    ) -> Run:
        """Create a cron job for a thread.

        Args:
            thread_id: the thread ID to run the cron job on.
            assistant_id: The assistant ID or graph name to use for the cron job.
                If using graph name, will default to first assistant created from that graph.
            schedule: The cron schedule to execute this job on.
            input: The input to the graph.
            metadata: Metadata to assign to the cron job runs.
            config: The configuration for the assistant.
            interrupt_before: Nodes to interrupt immediately before they get executed.

            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.

            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.

        Returns:
            Run: The cron run.

        Example Usage:

            cron_run = client.crons.create_for_thread(
                thread_id="my-thread-id",
                assistant_id="agent",
                schedule="27 15 * * *",
                input={"messages": [{"role": "user", "content": "hello!"}]},
                metadata={"name":"my_run"},
                config={"configurable": {"model_name": "openai"}},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            )

        """  # noqa: E501
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
        return self.http.post(f"/threads/{thread_id}/runs/crons", json=payload)

    def create(
        self,
        assistant_id: str,
        *,
        schedule: str,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
        config: Optional[Config] = None,
        interrupt_before: Optional[Union[All, list[str]]] = None,
        interrupt_after: Optional[Union[All, list[str]]] = None,
        webhook: Optional[str] = None,
        multitask_strategy: Optional[str] = None,
    ) -> Run:
        """Create a cron run.

        Args:
            assistant_id: The assistant ID or graph name to use for the cron job.
                If using graph name, will default to first assistant created from that graph.
            schedule: The cron schedule to execute this job on.
            input: The input to the graph.
            metadata: Metadata to assign to the cron job runs.
            config: The configuration for the assistant.
            interrupt_before: Nodes to interrupt immediately before they get executed.
            interrupt_after: Nodes to Nodes to interrupt immediately after they get executed.
            webhook: Webhook to call after LangGraph API call is done.
            multitask_strategy: Multitask strategy to use.
                Must be one of 'reject', 'interrupt', 'rollback', or 'enqueue'.

        Returns:
            Run: The cron run.

        Example Usage:

            cron_run = client.crons.create(
                assistant_id="agent",
                schedule="27 15 * * *",
                input={"messages": [{"role": "user", "content": "hello!"}]},
                metadata={"name":"my_run"},
                config={"configurable": {"model_name": "openai"}},
                interrupt_before=["node_to_stop_before_1","node_to_stop_before_2"],
                interrupt_after=["node_to_stop_after_1","node_to_stop_after_2"],
                webhook="https://my.fake.webhook.com",
                multitask_strategy="interrupt"
            )

        """  # noqa: E501
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
        return self.http.post("/runs/crons", json=payload)

    def delete(self, cron_id: str) -> None:
        """Delete a cron.

        Args:
            cron_id: The cron ID to delete.

        Returns:
            None

        Example Usage:

            client.crons.delete(
                cron_id="cron_to_delete"
            )

        """  # noqa: E501
        self.http.delete(f"/runs/crons/{cron_id}")

    def search(
        self,
        *,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Cron]:
        """Get a list of cron jobs.

        Args:
            assistant_id: The assistant ID or graph name to search for.
            thread_id: the thread ID to search for.
            limit: The maximum number of results to return.
            offset: The number of results to skip.

        Returns:
            list[Cron]: The list of cron jobs returned by the search,

        Example Usage:

            cron_jobs = client.crons.search(
                assistant_id="my_assistant_id",
                thread_id="my_thread_id",
                limit=5,
                offset=5,
            )
            print(cron_jobs)

            ----------------------------------------------------------

            [
                {
                    'cron_id': '1ef3cefa-4c09-6926-96d0-3dc97fd5e39b',
                    'assistant_id': 'my_assistant_id',
                    'thread_id': 'my_thread_id',
                    'user_id': None,
                    'payload':
                        {
                            'input': {'start_time': ''},
                            'schedule': '4 * * * *',
                            'assistant_id': 'my_assistant_id'
                        },
                    'schedule': '4 * * * *',
                    'next_run_date': '2024-07-25T17:04:00+00:00',
                    'end_time': None,
                    'created_at': '2024-07-08T06:02:23.073257+00:00',
                    'updated_at': '2024-07-08T06:02:23.073257+00:00'
                }
            ]

        """  # noqa: E501
        payload = {
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "limit": limit,
            "offset": offset,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return self.http.post("/runs/crons/search", json=payload)


class SyncStoreClient:
    """A client for synchronous operations on a key-value store.

    Provides methods to interact with a remote key-value store, allowing
    storage and retrieval of items within namespaced hierarchies.

    Example:

        client = get_sync_client()
        client.store.put_item(["users", "profiles"], "user123", {"name": "Alice", "age": 30})
    """

    def __init__(self, http: SyncHttpClient) -> None:
        self.http = http

    def put_item(
        self,
        namespace: Sequence[str],
        /,
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
    ) -> None:
        """Store or update an item.

        Args:
            namespace: A list of strings representing the namespace path.
            key: The unique identifier for the item within the namespace.
            value: A dictionary containing the item's data.
            index: Controls search indexing - None (use defaults), False (disable), or list of field paths to index.

        Returns:
            None

        Example Usage:

            client.store.put_item(
                ["documents", "user123"],
                key="item456",
                value={"title": "My Document", "content": "Hello World"}
            )
        """
        for label in namespace:
            if "." in label:
                raise ValueError(
                    f"Invalid namespace label '{label}'. Namespace labels cannot contain periods ('.')."
                )
        payload = {
            "namespace": namespace,
            "key": key,
            "value": value,
            "index": index,
        }
        self.http.put("/store/items", json=payload)

    def get_item(self, namespace: Sequence[str], /, key: str) -> Item:
        """Retrieve a single item.

        Args:
            key: The unique identifier for the item.
            namespace: Optional list of strings representing the namespace path.

        Returns:
            Item: The retrieved item.

        Example Usage:

            item = client.store.get_item(
                ["documents", "user123"],
                key="item456",
            )
            print(item)

            ----------------------------------------------------------------

            {
                'namespace': ['documents', 'user123'],
                'key': 'item456',
                'value': {'title': 'My Document', 'content': 'Hello World'},
                'created_at': '2024-07-30T12:00:00Z',
                'updated_at': '2024-07-30T12:00:00Z'
            }
        """
        for label in namespace:
            if "." in label:
                raise ValueError(
                    f"Invalid namespace label '{label}'. Namespace labels cannot contain periods ('.')."
                )

        return self.http.get(
            "/store/items", params={"key": key, "namespace": ".".join(namespace)}
        )

    def delete_item(self, namespace: Sequence[str], /, key: str) -> None:
        """Delete an item.

        Args:
            key: The unique identifier for the item.
            namespace: Optional list of strings representing the namespace path.

        Returns:
            None

        Example Usage:

            client.store.delete_item(
                ["documents", "user123"],
                key="item456",
            )
        """
        self.http.delete("/store/items", json={"key": key, "namespace": namespace})

    def search_items(
        self,
        namespace_prefix: Sequence[str],
        /,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        query: Optional[str] = None,
    ) -> SearchItemsResponse:
        """Search for items within a namespace prefix.

        Args:
            namespace_prefix: List of strings representing the namespace prefix.
            filter: Optional dictionary of key-value pairs to filter results.
            limit: Maximum number of items to return (default is 10).
            offset: Number of items to skip before returning results (default is 0).
            query: Optional query for natural language search.

        Returns:
            List[Item]: A list of items matching the search criteria.

        Example Usage:

            items = client.store.search_items(
                ["documents"],
                filter={"author": "John Doe"},
                limit=5,
                offset=0
            )
            print(items)

            ----------------------------------------------------------------

            {
                "items": [
                    {
                        "namespace": ["documents", "user123"],
                        "key": "item789",
                        "value": {
                            "title": "Another Document",
                            "author": "John Doe"
                        },
                        "created_at": "2024-07-30T12:00:00Z",
                        "updated_at": "2024-07-30T12:00:00Z"
                    },
                    # ... additional items ...
                ]
            }
        """
        payload = {
            "namespace_prefix": namespace_prefix,
            "filter": filter,
            "limit": limit,
            "offset": offset,
            "query": query,
        }
        return self.http.post("/store/items/search", json=_provided_vals(payload))

    def list_namespaces(
        self,
        prefix: Optional[List[str]] = None,
        suffix: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ListNamespaceResponse:
        """List namespaces with optional match conditions.

        Args:
            prefix: Optional list of strings representing the prefix to filter namespaces.
            suffix: Optional list of strings representing the suffix to filter namespaces.
            max_depth: Optional integer specifying the maximum depth of namespaces to return.
            limit: Maximum number of namespaces to return (default is 100).
            offset: Number of namespaces to skip before returning results (default is 0).

        Returns:
            List[List[str]]: A list of namespaces matching the criteria.

        Example Usage:

            namespaces = client.store.list_namespaces(
                prefix=["documents"],
                max_depth=3,
                limit=10,
                offset=0
            )
            print(namespaces)

            ----------------------------------------------------------------

            [
                ["documents", "user123", "reports"],
                ["documents", "user456", "invoices"],
                ...
            ]
        """
        payload = {
            "prefix": prefix,
            "suffix": suffix,
            "max_depth": max_depth,
            "limit": limit,
            "offset": offset,
        }
        return self.http.post("/store/namespaces", json=_provided_vals(payload))


def _provided_vals(d: dict):
    return {k: v for k, v in d.items() if v is not None}
