# Simple MCP server example for testing
import json
import logging
import random
import sys
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting MCP server with comprehensive debugging...")

    # Create FastMCP server
    mcp = FastMCP("test-mcp-store")

    # In-memory storage for testing
    store_data: Dict[str, Dict[str, Any]] = {}

    logger.info("Registering MCP tools with debug logging...")

    @mcp.tool()
    async def store_put(namespace: str, key: str, value: dict) -> str:
        """Store an item in the MCP store."""
        logger.info("[TOOL] store_put called:")
        logger.info(f"   namespace: {namespace}")
        logger.info(f"   key: {key}")
        logger.info(f"   value: {json.dumps(value, indent=2)}")
        store_key = f"{namespace}:{key}"
        store_data[store_key] = {
            "namespace": namespace,
            "key": key,
            "value": value,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }
        result = f"Stored item {key} in namespace {namespace}"
        logger.info(f"[TOOL] store_put result: {result}")
        logger.info(f"[TOOL] Store now has {len(store_data)} items")
        return result

    @mcp.tool()
    async def store_search(namespace: str, query: str | None = None) -> list:
        """Search for items in the MCP store."""
        logger.info("[TOOL] store_search called:")
        logger.info(f"   namespace: {namespace}")
        logger.info(f"   query: {query}")
        results = []
        matches_found = 0
        for store_key, item in store_data.items():
            if item["namespace"] == namespace:
                # Attach a random score < 1.0 to each result
                item_with_score = dict(item)
                item_with_score["score"] = (
                    random.random()
                )  # Returns float in [0.0, 1.0)
                results.append(item_with_score)
                matches_found += 1
                logger.info(
                    f"[TOOL] Found match #{matches_found}: {store_key}"
                    f" (score: {item_with_score['score']:.3f})"
                )
        logger.info(f"[TOOL] store_search returning {len(results)} results")
        return results

    @mcp.tool()
    async def store_delete(namespace: str, key: str) -> str:
        """Delete a specific item from the MCP store."""
        logger.info("[TOOL] store_delete called:")
        logger.info(f"   namespace: {namespace}")
        logger.info(f"   key: {key}")
        store_key = f"{namespace}:{key}"
        if store_key in store_data:
            del store_data[store_key]
            logger.info(f"[TOOL] store_delete: item deleted for key {store_key}")
            return f"Deleted item {key} from namespace {namespace}"
        else:
            logger.info(f"[TOOL] store_delete: item not found for key {store_key}")
            return f"Item {key} not found in namespace {namespace}"

    @mcp.tool()
    async def store_get(namespace: str, key: str) -> dict | None:
        """Get a specific item from the MCP store."""
        logger.info("[TOOL] store_get called:")
        logger.info(f"   namespace: {namespace}")
        logger.info(f"   key: {key}")
        store_key = f"{namespace}:{key}"
        result = store_data.get(store_key)
        if result:
            logger.info(f"[TOOL] store_get found item: {json.dumps(result, indent=2)}")
        else:
            logger.info(f"[TOOL] store_get: item not found for key {store_key}")
        return result

    @mcp.tool()
    async def store_list_namespaces(
        max_depth: int = 10, limit: int = 10, offset: int = 0
    ) -> list:
        """List all namespaces in the MCP store."""
        logger.info("[TOOL] store_list_namespaces called:")
        logger.info(f"   max_depth: {max_depth}")
        logger.info(f"   limit: {limit}")
        logger.info(f"   offset: {offset}")

        # Extract unique namespaces from store_data
        namespaces = set()
        for store_key, item in store_data.items():
            namespace = item.get("namespace", "")
            if namespace:
                namespaces.add(namespace)

        # Convert to list and apply pagination
        namespace_list = sorted(list(namespaces))
        paginated_namespaces = (
            namespace_list[offset : offset + limit]
            if limit
            else namespace_list[offset:]
        )

        logger.info(
            f"[TOOL] store_list_namespaces: found {len(namespace_list)}"
            f" total namespaces"
        )
        logger.info(
            f"[TOOL] store_list_namespaces: returning {len(paginated_namespaces)}"
            f" namespaces after pagination"
        )
        logger.info(f"[TOOL] store_list_namespaces: {paginated_namespaces}")

        return paginated_namespaces

    logger.info("MCP tools registered")
    logger.info("Starting server on http://127.0.0.1:8000...")
    logger.info("MCP endpoint available at: http://127.0.0.1:8000/mcp")
    logger.info("Debug logs will show all HTTP requests and MCP protocol messages")

    # Run the server with comprehensive logging
    try:
        # Start the server with streamable HTTP transport
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        logger.info("\nMCP server stopped")
        sys.stdout.flush()
    except Exception as e:
        logger.info(f"\nMCP server failed to start: {e}")
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    main()
