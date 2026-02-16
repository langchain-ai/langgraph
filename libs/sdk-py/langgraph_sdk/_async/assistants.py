"""Async client for managing assistants in LangGraph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast, overload

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk.schema import (
    Assistant,
    AssistantSelectField,
    AssistantSortBy,
    AssistantsSearchResponse,
    AssistantVersion,
    Config,
    Context,
    GraphSchema,
    Json,
    OnConflictBehavior,
    QueryParamTypes,
    SortOrder,
    Subgraphs,
)


class AssistantsClient:
    """Client for managing assistants in LangGraph.

    This class provides methods to interact with assistants,
    which are versioned configurations of your graph.

    ???+ example "Example"

        ```python
        client = get_client(url="http://localhost:2024")
        assistant = await client.assistants.get("assistant_id_123")
        ```
    """

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    async def get(
        self,
        assistant_id: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Assistant:
        """Get an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            Assistant: Assistant Object.

        ???+ example "Example Usage"

            ```python
            assistant = await client.assistants.get(
                assistant_id="my_assistant_id"
            )
            print(assistant)
            ```

            ```shell
            ----------------------------------------------------

            {
                'assistant_id': 'my_assistant_id',
                'graph_id': 'agent',
                'created_at': '2024-06-25T17:10:33.109781+00:00',
                'updated_at': '2024-06-25T17:10:33.109781+00:00',
                'config': {},
                'metadata': {'created_by': 'system'},
                'version': 1,
                'name': 'my_assistant'
            }
            ```
        """
        return await self.http.get(
            f"/assistants/{assistant_id}", headers=headers, params=params
        )

    async def get_graph(
        self,
        assistant_id: str,
        *,
        xray: int | bool = False,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Get the graph of an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get the graph of.
            xray: Include graph representation of subgraphs. If an integer value is provided, only subgraphs with a depth less than or equal to the value will be included.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            Graph: The graph information for the assistant in JSON format.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            graph_info = await client.assistants.get_graph(
                assistant_id="my_assistant_id"
            )
            print(graph_info)
            ```

            ```shell

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
            ```


        """
        query_params = {"xray": xray}
        if params:
            query_params.update(params)

        return await self.http.get(
            f"/assistants/{assistant_id}/graph", params=query_params, headers=headers
        )

    async def get_schemas(
        self,
        assistant_id: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> GraphSchema:
        """Get the schemas of an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get the schema of.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            GraphSchema: The graph schema for the assistant.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            schema = await client.assistants.get_schemas(
                assistant_id="my_assistant_id"
            )
            print(schema)
            ```

            ```shell

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
                'context_schema':
                    {
                        'title': 'Context',
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
            ```

        """
        return await self.http.get(
            f"/assistants/{assistant_id}/schemas", headers=headers, params=params
        )

    async def get_subgraphs(
        self,
        assistant_id: str,
        namespace: str | None = None,
        recurse: bool = False,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Subgraphs:
        """Get the schemas of an assistant by ID.

        Args:
            assistant_id: The ID of the assistant to get the schema of.
            namespace: Optional namespace to filter by.
            recurse: Whether to recursively get subgraphs.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            Subgraphs: The graph schema for the assistant.

        """
        get_params = {"recurse": recurse}
        if params:
            get_params = {**get_params, **params}
        if namespace is not None:
            return await self.http.get(
                f"/assistants/{assistant_id}/subgraphs/{namespace}",
                params=get_params,
                headers=headers,
            )
        else:
            return await self.http.get(
                f"/assistants/{assistant_id}/subgraphs",
                params=get_params,
                headers=headers,
            )

    async def create(
        self,
        graph_id: str | None,
        config: Config | None = None,
        *,
        context: Context | None = None,
        metadata: Json = None,
        assistant_id: str | None = None,
        if_exists: OnConflictBehavior | None = None,
        name: str | None = None,
        headers: Mapping[str, str] | None = None,
        description: str | None = None,
        params: QueryParamTypes | None = None,
    ) -> Assistant:
        """Create a new assistant.

        Useful when graph is configurable and you want to create different assistants based on different configurations.

        Args:
            graph_id: The ID of the graph the assistant should use. The graph ID is normally set in your langgraph.json configuration.
            config: Configuration to use for the graph.
            metadata: Metadata to add to assistant.
            context: Static context to add to the assistant.
                !!! version-added "Added in version 0.6.0"
            assistant_id: Assistant ID to use, will default to a random UUID if not provided.
            if_exists: How to handle duplicate creation. Defaults to 'raise' under the hood.
                Must be either 'raise' (raise error if duplicate), or 'do_nothing' (return existing assistant).
            name: The name of the assistant. Defaults to 'Untitled' under the hood.
            headers: Optional custom headers to include with the request.
            description: Optional description of the assistant.
                The description field is available for langgraph-api server version>=0.0.45
            params: Optional query parameters to include with the request.

        Returns:
            Assistant: The created assistant.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            assistant = await client.assistants.create(
                graph_id="agent",
                context={"model_name": "openai"},
                metadata={"number":1},
                assistant_id="my-assistant-id",
                if_exists="do_nothing",
                name="my_name"
            )
            ```
        """
        payload: dict[str, Any] = {
            "graph_id": graph_id,
        }
        if config:
            payload["config"] = config
        if context:
            payload["context"] = context
        if metadata:
            payload["metadata"] = metadata
        if assistant_id:
            payload["assistant_id"] = assistant_id
        if if_exists:
            payload["if_exists"] = if_exists
        if name:
            payload["name"] = name
        if description:
            payload["description"] = description
        return await self.http.post(
            "/assistants", json=payload, headers=headers, params=params
        )

    async def update(
        self,
        assistant_id: str,
        *,
        graph_id: str | None = None,
        config: Config | None = None,
        context: Context | None = None,
        metadata: Json = None,
        name: str | None = None,
        headers: Mapping[str, str] | None = None,
        description: str | None = None,
        params: QueryParamTypes | None = None,
    ) -> Assistant:
        """Update an assistant.

        Use this to point to a different graph, update the configuration, or change the metadata of an assistant.

        Args:
            assistant_id: Assistant to update.
            graph_id: The ID of the graph the assistant should use.
                The graph ID is normally set in your langgraph.json configuration. If `None`, assistant will keep pointing to same graph.
            config: Configuration to use for the graph.
            context: Static context to add to the assistant.
                !!! version-added "Added in version 0.6.0"
            metadata: Metadata to merge with existing assistant metadata.
            name: The new name for the assistant.
            headers: Optional custom headers to include with the request.
            description: Optional description of the assistant.
                The description field is available for langgraph-api server version>=0.0.45
            params: Optional query parameters to include with the request.

        Returns:
            The updated assistant.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            assistant = await client.assistants.update(
                assistant_id='e280dad7-8618-443f-87f1-8e41841c180f',
                graph_id="other-graph",
                context={"model_name": "anthropic"},
                metadata={"number":2}
            )
            ```

        """
        payload: dict[str, Any] = {}
        if graph_id:
            payload["graph_id"] = graph_id
        if config:
            payload["config"] = config
        if context:
            payload["context"] = context
        if metadata:
            payload["metadata"] = metadata
        if name:
            payload["name"] = name
        if description:
            payload["description"] = description
        return await self.http.patch(
            f"/assistants/{assistant_id}",
            json=payload,
            headers=headers,
            params=params,
        )

    async def delete(
        self,
        assistant_id: str,
        *,
        delete_threads: bool = False,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> None:
        """Delete an assistant.

        Args:
            assistant_id: The assistant ID to delete.
            delete_threads: If true, delete all threads with `metadata.assistant_id`
                matching this assistant, along with runs and checkpoints belonging to
                those threads.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            await client.assistants.delete(
                assistant_id="my_assistant_id"
            )
            ```

        """
        query_params: dict[str, Any] = {}
        if delete_threads:
            query_params["delete_threads"] = True
        if params:
            query_params.update(params)
        await self.http.delete(
            f"/assistants/{assistant_id}",
            headers=headers,
            params=query_params or None,
        )

    @overload
    async def search(
        self,
        *,
        metadata: Json = None,
        graph_id: str | None = None,
        name: str | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: AssistantSortBy | None = None,
        sort_order: SortOrder | None = None,
        select: list[AssistantSelectField] | None = None,
        response_format: Literal["object"],
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> AssistantsSearchResponse: ...

    @overload
    async def search(
        self,
        *,
        metadata: Json = None,
        graph_id: str | None = None,
        name: str | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: AssistantSortBy | None = None,
        sort_order: SortOrder | None = None,
        select: list[AssistantSelectField] | None = None,
        response_format: Literal["array"] = "array",
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> list[Assistant]: ...

    async def search(
        self,
        *,
        metadata: Json = None,
        graph_id: str | None = None,
        name: str | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: AssistantSortBy | None = None,
        sort_order: SortOrder | None = None,
        select: list[AssistantSelectField] | None = None,
        response_format: Literal["array", "object"] = "array",
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> AssistantsSearchResponse | list[Assistant]:
        """Search for assistants.

        Args:
            metadata: Metadata to filter by. Exact match filter for each KV pair.
            graph_id: The ID of the graph to filter by.
                The graph ID is normally set in your langgraph.json configuration.
            name: The name of the assistant to filter by.
                The filtering logic will match assistants where 'name' is a substring (case insensitive) of the assistant name.
            limit: The maximum number of results to return.
            offset: The number of results to skip.
            sort_by: The field to sort by.
            sort_order: The order to sort by.
            select: Specific assistant fields to include in the response.
            response_format: Controls the response shape. Use `"array"` (default)
                to return a bare list of assistants, or `"object"` to return
                a mapping containing assistants plus pagination metadata.
                Defaults to "array", though this default will be changed to "object" in a future release.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            A list of assistants (when `response_format="array"`) or a mapping
            with the assistants and the next pagination cursor (when
            `response_format="object"`).

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            response = await client.assistants.search(
                metadata = {"name":"my_name"},
                graph_id="my_graph_id",
                limit=5,
                offset=5,
                response_format="object"
            )
            next_cursor = response["next"]
            assistants = response["assistants"]
            ```
        """
        if response_format not in ("array", "object"):
            raise ValueError(
                f"response_format must be 'array' or 'object', got {response_format!r}"
            )
        payload: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if metadata:
            payload["metadata"] = metadata
        if graph_id:
            payload["graph_id"] = graph_id
        if name:
            payload["name"] = name
        if sort_by:
            payload["sort_by"] = sort_by
        if sort_order:
            payload["sort_order"] = sort_order
        if select:
            payload["select"] = select
        next_cursor: str | None = None

        def capture_pagination(response: httpx.Response) -> None:
            nonlocal next_cursor
            next_cursor = response.headers.get("X-Pagination-Next")

        assistants = cast(
            list[Assistant],
            await self.http.post(
                "/assistants/search",
                json=payload,
                headers=headers,
                params=params,
                on_response=capture_pagination if response_format == "object" else None,
            ),
        )
        if response_format == "object":
            return {"assistants": assistants, "next": next_cursor}
        return assistants

    async def count(
        self,
        *,
        metadata: Json = None,
        graph_id: str | None = None,
        name: str | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> int:
        """Count assistants matching filters.

        Args:
            metadata: Metadata to filter by. Exact match for each key/value.
            graph_id: Optional graph id to filter by.
            name: Optional name to filter by.
                The filtering logic will match assistants where 'name' is a substring (case insensitive) of the assistant name.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            int: Number of assistants matching the criteria.
        """
        payload: dict[str, Any] = {}
        if metadata:
            payload["metadata"] = metadata
        if graph_id:
            payload["graph_id"] = graph_id
        if name:
            payload["name"] = name
        return await self.http.post(
            "/assistants/count", json=payload, headers=headers, params=params
        )

    async def get_versions(
        self,
        assistant_id: str,
        metadata: Json = None,
        limit: int = 10,
        offset: int = 0,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> list[AssistantVersion]:
        """List all versions of an assistant.

        Args:
            assistant_id: The assistant ID to get versions for.
            metadata: Metadata to filter versions by. Exact match filter for each KV pair.
            limit: The maximum number of versions to return.
            offset: The number of versions to skip.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            A list of assistant versions.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            assistant_versions = await client.assistants.get_versions(
                assistant_id="my_assistant_id"
            )
            ```
        """

        payload: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if metadata:
            payload["metadata"] = metadata
        return await self.http.post(
            f"/assistants/{assistant_id}/versions",
            json=payload,
            headers=headers,
            params=params,
        )

    async def set_latest(
        self,
        assistant_id: str,
        version: int,
        *,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Assistant:
        """Change the version of an assistant.

        Args:
            assistant_id: The assistant ID to delete.
            version: The version to change to.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            Assistant Object.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            new_version_assistant = await client.assistants.set_latest(
                assistant_id="my_assistant_id",
                version=3
            )
            ```

        """

        payload: dict[str, Any] = {"version": version}

        return await self.http.post(
            f"/assistants/{assistant_id}/latest",
            json=payload,
            headers=headers,
            params=params,
        )
