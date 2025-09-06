import gc
import json
import logging

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class AsyncMCPStoreClient:
    def __init__(self, host="localhost", port=8000, **kwargs):
        self.host = host
        self.port = port
        self.session = None
        self.url = f"http://{host}:{port}/mcp"
        self.client_args = {"url": self.url, **kwargs}
        self.client_gen = None
        self.read_stream = None
        self.write_stream = None
        self.get_status = None

    async def __aenter__(self):
        self.client_gen = streamablehttp_client(self.url)
        (
            self.read_stream,
            self.write_stream,
            self.get_status,
        ) = await self.client_gen.__aenter__()
        self.session = ClientSession(self.read_stream, self.write_stream)
        await self.session.__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            try:
                await self.session.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                if "cancel scope" not in str(e):
                    logger.error(f"Error during session cleanup: {e}")
            finally:
                self.session = None
        if hasattr(self, "client_gen") and self.client_gen:
            try:
                await self.client_gen.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                if "cancel scope" not in str(e):
                    logger.error(f"Error during client cleanup: {e}")
            finally:
                self.client_gen = None
        gc.collect()

    async def aput(self, namespace, key, value):
        if self.session is None:
            raise RuntimeError("MCP session not initialized")
        try:
            namespace_str = (
                ".".join(str(x) for x in namespace)
                if isinstance(namespace, tuple)
                else str(namespace)
            )
            result = await self.session.call_tool(
                name="store_put",
                arguments={"namespace": namespace_str, "key": key, "value": value},
            )
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Put operation failed: {e}")
            return {"success": False, "error": str(e)}

    async def adelete(self, namespace, key):
        if self.session is None:
            raise RuntimeError("MCP session not initialized")
        try:
            namespace_str = (
                ".".join(str(x) for x in namespace)
                if isinstance(namespace, tuple)
                else str(namespace)
            )
            result = await self.session.call_tool(
                name="store_delete", arguments={"namespace": namespace_str, "key": key}
            )
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Delete operation failed: {e}")
            return {"success": False, "error": str(e)}

    async def asearch(
        self, namespace_prefix, query=None, filter=None, limit=10, offset=0
    ) -> list[str]:
        if self.session is None:
            raise RuntimeError("MCP session not initialized")
        try:
            namespace = (
                ".".join(str(x) for x in namespace_prefix)
                if isinstance(namespace_prefix, tuple)
                else str(namespace_prefix)
            )
            arguments = {"namespace": namespace}
            if query is not None:
                arguments["query"] = query
            if filter is not None:
                arguments["filter"] = filter
            if limit is not None:
                arguments["limit"] = limit
            if offset is not None:
                arguments["offset"] = offset

            result = await self.session.call_tool(
                name="store_search", arguments=arguments
            )
            if result and hasattr(result, "content"):
                search_results = []
                for content_item in result.content:
                    if type(content_item) is types.TextContent:
                        if hasattr(content_item, "text"):
                            search_results.append(content_item.text)
                return search_results
            return []
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            return []

    async def aget(self, namespace, key) -> str | None:
        if self.session is None:
            raise RuntimeError("MCP session not initialized")
        try:
            namespace_str = (
                ".".join(str(x) for x in namespace)
                if isinstance(namespace, tuple)
                else str(namespace)
            )
            result = await self.session.call_tool(
                name="store_get", arguments={"namespace": namespace_str, "key": key}
            )
            if result and hasattr(result, "content"):
                for content_item in result.content:
                    if type(content_item) is types.TextContent:
                        if hasattr(content_item, "text"):
                            return str(content_item.text)
            return None
        except Exception as e:
            logger.error(f"Get operation failed: {e}")
            return None

    async def alist_namespaces(
        self, match_conditions=None, max_depth=None, limit=None, offset=None
    ):
        if self.session is None:
            raise RuntimeError("MCP session not initialized")
        try:
            result = await self.session.call_tool(
                name="store_list_namespaces",
                arguments={
                    "max_depth": max_depth or 10,
                    "limit": limit or 10,
                    "offset": offset or 0,
                },
            )
            namespaces = []
            if result and hasattr(result, "content"):
                for content_item in result.content:
                    if hasattr(content_item, "text"):
                        try:
                            data = json.loads(content_item.text)
                            if isinstance(data, list):
                                namespaces.extend(data)
                            else:
                                namespaces.append(data)
                        except json.JSONDecodeError:
                            namespaces.append(content_item.text)
            return namespaces
        except Exception as e:
            logger.error(f"List namespaces operation failed: {e}")
            return []
