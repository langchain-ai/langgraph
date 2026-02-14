"""Synchronous client for managing threads in LangGraph."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk.schema import (
    Checkpoint,
    Json,
    OnConflictBehavior,
    PruneStrategy,
    QueryParamTypes,
    SortOrder,
    StreamPart,
    Thread,
    ThreadSelectField,
    ThreadSortBy,
    ThreadState,
    ThreadStatus,
    ThreadStreamMode,
    ThreadUpdateStateResponse,
)


class SyncThreadsClient:
    """Synchronous client for managing threads in LangGraph.

    This class provides methods to create, retrieve, and manage threads,
    which represent conversations or stateful interactions.

    ???+ example "Example"

        ```python
        client = get_sync_client(url="http://localhost:2024")
        thread = client.threads.create(metadata={"user_id": "123"})
        ```
    """

    def __init__(self, http: SyncHttpClient) -> None:
        self.http = http

    def get(
        self,
        thread_id: str,
        *,
        include: Sequence[str] | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Thread:
        """Get a thread by ID.

        Args:
            thread_id: The ID of the thread to get.
            include: Additional fields to include in the response.
                Supported values: `"ttl"`.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `Thread` object.

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            thread = client.threads.get(
                thread_id="my_thread_id"
            )
            print(thread)
            ```
            ```shell
            -----------------------------------------------------

            {
                'thread_id': 'my_thread_id',
                'created_at': '2024-07-18T18:35:15.540834+00:00',
                'updated_at': '2024-07-18T18:35:15.540834+00:00',
                'metadata': {'graph_id': 'agent'}
            }
            ```

        """
        query_params: dict[str, Any] = {}
        if include:
            query_params["include"] = ",".join(include)
        if params:
            query_params.update(params)
        return self.http.get(
            f"/threads/{thread_id}",
            headers=headers,
            params=query_params or None,
        )

    def create(
        self,
        *,
        metadata: Json = None,
        thread_id: str | None = None,
        if_exists: OnConflictBehavior | None = None,
        supersteps: Sequence[dict[str, Sequence[dict[str, Any]]]] | None = None,
        graph_id: str | None = None,
        ttl: int | Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Thread:
        """Create a new thread.

        Args:
            metadata: Metadata to add to thread.
            thread_id: ID of thread.
                If `None`, ID will be a randomly generated UUID.
            if_exists: How to handle duplicate creation. Defaults to 'raise' under the hood.
                Must be either 'raise' (raise error if duplicate), or 'do_nothing' (return existing thread).
            supersteps: Apply a list of supersteps when creating a thread, each containing a sequence of updates.
                Each update has `values` or `command` and `as_node`. Used for copying a thread between deployments.
            graph_id: Optional graph ID to associate with the thread.
            ttl: Optional time-to-live in minutes for the thread. You can pass an
                integer (minutes) or a mapping with keys `ttl` and optional
                `strategy` (defaults to "delete").
            headers: Optional custom headers to include with the request.

        Returns:
            The created `Thread`.

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            thread = client.threads.create(
                metadata={"number":1},
                thread_id="my-thread-id",
                if_exists="raise"
            )
            ```
            )
        """
        payload: dict[str, Any] = {}
        if thread_id:
            payload["thread_id"] = thread_id
        if metadata or graph_id:
            payload["metadata"] = {
                **(metadata or {}),
                **({"graph_id": graph_id} if graph_id else {}),
            }
        if if_exists:
            payload["if_exists"] = if_exists
        if supersteps:
            payload["supersteps"] = [
                {
                    "updates": [
                        {
                            "values": u["values"],
                            "command": u.get("command"),
                            "as_node": u["as_node"],
                        }
                        for u in s["updates"]
                    ]
                }
                for s in supersteps
            ]
        if ttl is not None:
            if isinstance(ttl, (int, float)):
                payload["ttl"] = {"ttl": ttl, "strategy": "delete"}
            else:
                payload["ttl"] = ttl

        return self.http.post("/threads", json=payload, headers=headers, params=params)

    def update(
        self,
        thread_id: str,
        *,
        metadata: Mapping[str, Any],
        ttl: int | Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Thread:
        """Update a thread.

        Args:
            thread_id: ID of thread to update.
            metadata: Metadata to merge with existing thread metadata.
            ttl: Optional time-to-live in minutes for the thread. You can pass an
                integer (minutes) or a mapping with keys `ttl` and optional
                `strategy` (defaults to "delete").
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            The created `Thread`.

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            thread = client.threads.update(
                thread_id="my-thread-id",
                metadata={"number":1},
                ttl=43_200,
            )
            ```
        """
        payload: dict[str, Any] = {"metadata": metadata}
        if ttl is not None:
            if isinstance(ttl, (int, float)):
                payload["ttl"] = {"ttl": ttl, "strategy": "delete"}
            else:
                payload["ttl"] = ttl
        return self.http.patch(
            f"/threads/{thread_id}",
            json=payload,
            headers=headers,
            params=params,
        )

    def delete(
        self,
        thread_id: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> None:
        """Delete a thread.

        Args:
            thread_id: The ID of the thread to delete.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client.threads.delete(
                thread_id="my_thread_id"
            )
            ```

        """
        self.http.delete(f"/threads/{thread_id}", headers=headers, params=params)

    def search(
        self,
        *,
        metadata: Json = None,
        values: Json = None,
        ids: Sequence[str] | None = None,
        status: ThreadStatus | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: ThreadSortBy | None = None,
        sort_order: SortOrder | None = None,
        select: list[ThreadSelectField] | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> list[Thread]:
        """Search for threads.

        Args:
            metadata: Thread metadata to filter on.
            values: State values to filter on.
            ids: List of thread IDs to filter by.
            status: Thread status to filter on.
                Must be one of 'idle', 'busy', 'interrupted' or 'error'.
            limit: Limit on number of threads to return.
            offset: Offset in threads table to start search from.
            headers: Optional custom headers to include with the request.

        Returns:
            List of the threads matching the search parameters.

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            threads = client.threads.search(
                metadata={"number":1},
                status="interrupted",
                limit=15,
                offset=5
            )
            ```
        """
        payload: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if metadata:
            payload["metadata"] = metadata
        if values:
            payload["values"] = values
        if ids:
            payload["ids"] = ids
        if status:
            payload["status"] = status
        if sort_by:
            payload["sort_by"] = sort_by
        if sort_order:
            payload["sort_order"] = sort_order
        if select:
            payload["select"] = select
        return self.http.post(
            "/threads/search", json=payload, headers=headers, params=params
        )

    def count(
        self,
        *,
        metadata: Json = None,
        values: Json = None,
        status: ThreadStatus | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> int:
        """Count threads matching filters.

        Args:
            metadata: Thread metadata to filter on.
            values: State values to filter on.
            status: Thread status to filter on.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            int: Number of threads matching the criteria.
        """
        payload: dict[str, Any] = {}
        if metadata:
            payload["metadata"] = metadata
        if values:
            payload["values"] = values
        if status:
            payload["status"] = status
        return self.http.post(
            "/threads/count", json=payload, headers=headers, params=params
        )

    def copy(
        self,
        thread_id: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> None:
        """Copy a thread.

        Args:
            thread_id: The ID of the thread to copy.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            client.threads.copy(
                thread_id="my_thread_id"
            )
            ```

        """
        return self.http.post(
            f"/threads/{thread_id}/copy", json=None, headers=headers, params=params
        )

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: PruneStrategy = "delete",
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> dict[str, Any]:
        """Prune threads by ID.

        Args:
            thread_ids: List of thread IDs to prune.
            strategy: The prune strategy. `"delete"` removes threads entirely.
                `"keep_latest"` prunes old checkpoints but keeps threads and their
                latest state. Defaults to `"delete"`.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            A dict containing `pruned_count` (number of threads pruned).

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            result = client.threads.prune(
                thread_ids=["thread_1", "thread_2"],
            )
            print(result)  # {'pruned_count': 2}
            ```

        """
        payload: dict[str, Any] = {
            "thread_ids": thread_ids,
        }
        if strategy != "delete":
            payload["strategy"] = strategy
        return self.http.post(
            "/threads/prune", json=payload, headers=headers, params=params
        )

    def get_state(
        self,
        thread_id: str,
        checkpoint: Checkpoint | None = None,
        checkpoint_id: str | None = None,  # deprecated
        *,
        subgraphs: bool = False,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> ThreadState:
        """Get the state of a thread.

        Args:
            thread_id: The ID of the thread to get the state of.
            checkpoint: The checkpoint to get the state of.
            subgraphs: Include subgraphs states.
            headers: Optional custom headers to include with the request.

        Returns:
            The thread of the state.

        ???+ example "Example Usage"

            ```python
            client = get_sync_client(url="http://localhost:2024")
            thread_state = client.threads.get_state(
                thread_id="my_thread_id",
                checkpoint_id="my_checkpoint_id"
            )
            print(thread_state)
            ```

            ```shell
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
            ```

        """
        if checkpoint:
            return self.http.post(
                f"/threads/{thread_id}/state/checkpoint",
                json={"checkpoint": checkpoint, "subgraphs": subgraphs},
                headers=headers,
                params=params,
            )
        elif checkpoint_id:
            get_params = {"subgraphs": subgraphs}
            if params:
                get_params = {**get_params, **params}
            return self.http.get(
                f"/threads/{thread_id}/state/{checkpoint_id}",
                params=get_params,
                headers=headers,
            )
        else:
            get_params = {"subgraphs": subgraphs}
            if params:
                get_params = {**get_params, **params}
            return self.http.get(
                f"/threads/{thread_id}/state",
                params=get_params,
                headers=headers,
            )

    def update_state(
        self,
        thread_id: str,
        values: dict[str, Any] | Sequence[dict] | None,
        *,
        as_node: str | None = None,
        checkpoint: Checkpoint | None = None,
        checkpoint_id: str | None = None,  # deprecated
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> ThreadUpdateStateResponse:
        """Update the state of a thread.

        Args:
            thread_id: The ID of the thread to update.
            values: The values to update the state with.
            as_node: Update the state as if this node had just executed.
            checkpoint: The checkpoint to update the state of.
            headers: Optional custom headers to include with the request.

        Returns:
            Response after updating a thread's state.

        ???+ example "Example Usage"

            ```python

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
            ```

        """
        payload: dict[str, Any] = {
            "values": values,
        }
        if checkpoint_id:
            payload["checkpoint_id"] = checkpoint_id
        if checkpoint:
            payload["checkpoint"] = checkpoint
        if as_node:
            payload["as_node"] = as_node
        return self.http.post(
            f"/threads/{thread_id}/state", json=payload, headers=headers, params=params
        )

    def get_history(
        self,
        thread_id: str,
        *,
        limit: int = 10,
        before: str | Checkpoint | None = None,
        metadata: Mapping[str, Any] | None = None,
        checkpoint: Checkpoint | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> list[ThreadState]:
        """Get the state history of a thread.

        Args:
            thread_id: The ID of the thread to get the state history for.
            checkpoint: Return states for this subgraph. If empty defaults to root.
            limit: The maximum number of states to return.
            before: Return states before this checkpoint.
            metadata: Filter states by metadata key-value pairs.
            headers: Optional custom headers to include with the request.

        Returns:
            The state history of the `Thread`.

        ???+ example "Example Usage"

            ```python

            thread_state = client.threads.get_history(
                thread_id="my_thread_id",
                limit=5,
                before="my_timestamp",
                metadata={"name":"my_name"}
            )
            ```

        """
        payload: dict[str, Any] = {
            "limit": limit,
        }
        if before:
            payload["before"] = before
        if metadata:
            payload["metadata"] = metadata
        if checkpoint:
            payload["checkpoint"] = checkpoint
        return self.http.post(
            f"/threads/{thread_id}/history",
            json=payload,
            headers=headers,
            params=params,
        )

    def join_stream(
        self,
        thread_id: str,
        *,
        stream_mode: ThreadStreamMode | Sequence[ThreadStreamMode] = "run_modes",
        last_event_id: str | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Iterator[StreamPart]:
        """Get a stream of events for a thread.

        Args:
            thread_id: The ID of the thread to get the stream for.
            last_event_id: The ID of the last event to get.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            An iterator of stream parts.

        ???+ example "Example Usage"

            ```python

            for chunk in client.threads.join_stream(
                thread_id="my_thread_id",
                last_event_id="my_event_id",
                stream_mode="run_modes",
            ):
                print(chunk)
            ```

        """
        query_params = {
            "stream_mode": stream_mode,
        }
        if params:
            query_params.update(params)
        return self.http.stream(
            f"/threads/{thread_id}/stream",
            "GET",
            headers={
                **({"Last-Event-ID": last_event_id} if last_event_id else {}),
                **(headers or {}),
            },
            params=query_params,
        )
