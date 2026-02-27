"""Async Store client for LangGraph SDK."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._shared.utilities import _provided_vals
from langgraph_sdk.schema import (
    Item,
    ListNamespaceResponse,
    QueryParamTypes,
    SearchItemsResponse,
)


class StoreClient:
    """Client for interacting with the graph's shared storage.

    The Store provides a key-value storage system for persisting data across graph executions,
    allowing for stateful operations and data sharing across threads.

    ???+ example "Example"

        ```python
        client = get_client(url="http://localhost:2024")
        await client.store.put_item(["users", "user123"], "mem-123451342", {"name": "Alice", "score": 100})
        ```
    """

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    async def put_item(
        self,
        namespace: Sequence[str],
        /,
        key: str,
        value: Mapping[str, Any],
        index: Literal[False] | list[str] | None = None,
        ttl: int | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> None:
        """Store or update an item.

        Args:
            namespace: A list of strings representing the namespace path.
            key: The unique identifier for the item within the namespace.
            value: A dictionary containing the item's data.
            index: Controls search indexing - None (use defaults), False (disable), or list of field paths to index.
            ttl: Optional time-to-live in minutes for the item, or None for no expiration.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            await client.store.put_item(
                ["documents", "user123"],
                key="item456",
                value={"title": "My Document", "content": "Hello World"}
            )
            ```
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
            "ttl": ttl,
        }
        await self.http.put(
            "/store/items", json=_provided_vals(payload), headers=headers, params=params
        )

    async def get_item(
        self,
        namespace: Sequence[str],
        /,
        key: str,
        *,
        refresh_ttl: bool | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> Item:
        """Retrieve a single item.

        Args:
            key: The unique identifier for the item.
            namespace: Optional list of strings representing the namespace path.
            refresh_ttl: Whether to refresh the TTL on this read operation. If `None`, uses the store's default behavior.

        Returns:
            Item: The retrieved item.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            item = await client.store.get_item(
                ["documents", "user123"],
                key="item456",
            )
            print(item)
            ```
            ```shell

            ----------------------------------------------------------------

            {
                'namespace': ['documents', 'user123'],
                'key': 'item456',
                'value': {'title': 'My Document', 'content': 'Hello World'},
                'created_at': '2024-07-30T12:00:00Z',
                'updated_at': '2024-07-30T12:00:00Z'
            }
            ```
        """
        for label in namespace:
            if "." in label:
                raise ValueError(
                    f"Invalid namespace label '{label}'. Namespace labels cannot contain periods ('.')."
                )
        get_params = {"namespace": ".".join(namespace), "key": key}
        if refresh_ttl is not None:
            get_params["refresh_ttl"] = refresh_ttl
        if params:
            get_params = {**get_params, **dict(params)}
        return await self.http.get("/store/items", params=get_params, headers=headers)

    async def delete_item(
        self,
        namespace: Sequence[str],
        /,
        key: str,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> None:
        """Delete an item.

        Args:
            key: The unique identifier for the item.
            namespace: Optional list of strings representing the namespace path.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            `None`

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            await client.store.delete_item(
                ["documents", "user123"],
                key="item456",
            )
            ```
        """
        await self.http.delete(
            "/store/items",
            json={"namespace": namespace, "key": key},
            headers=headers,
            params=params,
        )

    async def search_items(
        self,
        namespace_prefix: Sequence[str],
        /,
        filter: Mapping[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        query: str | None = None,
        refresh_ttl: bool | None = None,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> SearchItemsResponse:
        """Search for items within a namespace prefix.

        Args:
            namespace_prefix: List of strings representing the namespace prefix.
            filter: Optional dictionary of key-value pairs to filter results.
            limit: Maximum number of items to return (default is 10).
            offset: Number of items to skip before returning results (default is 0).
            query: Optional query for natural language search.
            refresh_ttl: Whether to refresh the TTL on items returned by this search. If `None`, uses the store's default behavior.
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            A list of items matching the search criteria.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
            items = await client.store.search_items(
                ["documents"],
                filter={"author": "John Doe"},
                limit=5,
                offset=0
            )
            print(items)
            ```
            ```shell

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
            ```
        """
        payload = {
            "namespace_prefix": namespace_prefix,
            "filter": filter,
            "limit": limit,
            "offset": offset,
            "query": query,
            "refresh_ttl": refresh_ttl,
        }

        return await self.http.post(
            "/store/items/search",
            json=_provided_vals(payload),
            headers=headers,
            params=params,
        )

    async def list_namespaces(
        self,
        prefix: list[str] | None = None,
        suffix: list[str] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> ListNamespaceResponse:
        """List namespaces with optional match conditions.

        Args:
            prefix: Optional list of strings representing the prefix to filter namespaces.
            suffix: Optional list of strings representing the suffix to filter namespaces.
            max_depth: Optional integer specifying the maximum depth of namespaces to return.
            limit: Maximum number of namespaces to return (default is 100).
            offset: Number of namespaces to skip before returning results (default is 0).
            headers: Optional custom headers to include with the request.
            params: Optional query parameters to include with the request.

        Returns:
            A list of namespaces matching the criteria.

        ???+ example "Example Usage"

            ```python
            client = get_client(url="http://localhost:2024")
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
            ```
        """
        payload = {
            "prefix": prefix,
            "suffix": suffix,
            "max_depth": max_depth,
            "limit": limit,
            "offset": offset,
        }
        return await self.http.post(
            "/store/namespaces",
            json=_provided_vals(payload),
            headers=headers,
            params=params,
        )
