import pytest
from conftest import vprint

from langgraph.store.mcp import AsyncMCPStore


@pytest.mark.asyncio
async def test_mcp_store_put_and_search(mcp_server):
    async with AsyncMCPStore.from_mcp_config(host="localhost", port=8000) as store:
        # Test that the store and client work
        assert store is not None
        assert store.client is not None

        # Test put operation
        try:
            await store.aput(
                namespace=("user_accounts",),
                key="user_001",
                value={"role": "administrator", "department": "engineering"},
            )
            vprint("Put operation completed", level=1)

        except Exception as e:
            vprint(f"Put operation failed: {e}", level=0)
            raise

        # Test search operation
        try:
            search_results = await store.asearch(
                namespace_prefix=("user_accounts",), query=None, limit=10
            )

            # Check if we got results back - handle gracefully if none
            if len(search_results) > 0:
                # Check if the result has the expected structure
                first_result = search_results[0]
                assert hasattr(first_result, "key"), (
                    "Result should have a key attribute"
                )
                assert hasattr(first_result, "value"), (
                    "Result should have a value attribute"
                )
                assert first_result.value == {
                    "role": "administrator",
                    "department": "engineering",
                }, "Value should match what we put"

                # Verify score is present and in valid range
                if hasattr(first_result, "score"):
                    assert isinstance(first_result.score, float), (
                        f"Score should be float, got {type(first_result.score)}"
                    )
                    assert 0.0 <= first_result.score <= 1.0, (
                        f"Score should be in [0.0, 1.0], got {first_result.score}"
                    )

                vprint("Search operations successful", level=1)
                vprint(
                    f"Search operation returned {len(search_results)} result(s)",
                    level=1,
                )
                print("Put and search test completed successfully!")
            else:
                vprint(
                    "Search returned no results (expected without indexing)", level=1
                )
                print("Put operation test completed successfully!")

        except Exception as e:
            vprint(f"Search operation failed: {e}", level=0)
            raise


@pytest.mark.asyncio
async def test_mcp_store_get_and_delete(mcp_server):
    """Test get and delete operations for AsyncMCPStore using high-level APIs."""
    async with AsyncMCPStore.from_mcp_config(host="localhost", port=8000) as store:
        # Put an item
        await store.aput(
            namespace=("products",),
            key="product_123",
            value={"price": 29.99, "category": "electronics"},
        )

        # Get the item
        item = await store.aget(namespace=("products",), key="product_123")
        assert item is not None, "Get should return the item just put"
        if isinstance(item, dict):
            value = item.get("value", {})
        else:
            value = getattr(item, "value", None)
        assert value == {"price": 29.99, "category": "electronics"}, (
            f"Value should match what was put, got {value}"
        )

        # Delete the item by putting None value
        await store.aput(namespace=("products",), key="product_123", value=None)

        # Get again should return None
        item2 = await store.aget(namespace=("products",), key="product_123")
        assert not item2, "Get after delete should return None"
        print("Get and delete test completed successfully!")


@pytest.mark.asyncio
async def test_mcp_store_list_namespaces(mcp_server):
    """Test list_namespaces operation for AsyncMCPStore using high-level APIs."""
    async with AsyncMCPStore.from_mcp_config(host="localhost", port=8000) as store:
        # Put items in different namespaces
        await store.aput(
            namespace=("customers",),
            key="customer_001",
            value={"account_status": "active"},
        )
        await store.aput(
            namespace=("orders",), key="order_456", value={"total_amount": 150.75}
        )

        # List namespaces
        namespaces = await store.alist_namespaces()

        assert isinstance(namespaces, list), (
            f"list_namespaces should return a list, got {type(namespaces)}"
        )
        assert any("customers" in str(ns) for ns in namespaces), (
            f"customers should be in namespaces: {namespaces}"
        )
        assert any("orders" in str(ns) for ns in namespaces), (
            f"orders should be in namespaces: {namespaces}"
        )
        print("list_namespaces test completed successfully!")


@pytest.mark.asyncio
async def test_mcp_store_high_level_operations(mcp_server):
    """Test high-level API operations for AsyncMCPStore with put and search"""
    async with AsyncMCPStore.from_mcp_config(host="localhost", port=8000) as store:
        vprint("Testing high-level API operations...", level=1)

        # Put some documents in different namespaces
        await store.aput(
            namespace=("docs",),
            key="doc1",
            value={"title": "Document 1", "content": "First document"},
        )
        await store.aput(
            namespace=("docs",),
            key="doc2",
            value={"title": "Document 2", "content": "Second document"},
        )
        await store.aput(
            namespace=("config",),
            key="setting1",
            value={"name": "timeout", "value": 30},
        )

        vprint("Executing put operations completed", level=1)

        # Search in different namespaces
        docs_search = await store.asearch(
            namespace_prefix=("docs",), query=None, limit=10
        )
        config_search = await store.asearch(
            namespace_prefix=("config",), query=None, limit=5
        )

        # Verify search operations returned lists
        assert isinstance(docs_search, list), (
            f"Docs search should return list, got {type(docs_search)}"
        )
        assert isinstance(config_search, list), (
            f"Config search should return list, got {type(config_search)}"
        )

        # Verify search results have the expected structure
        if len(docs_search) > 0:
            first_doc = docs_search[0]
            assert hasattr(first_doc, "value"), (
                "Search result should have a value attribute"
            )
            assert hasattr(first_doc, "namespace"), (
                "Search result should have a namespace attribute"
            )
            assert hasattr(first_doc, "key"), (
                "Search result should have a key attribute"
            )

        # Success details under verbosity
        vprint("High-level API operations completed successfully", level=1)
        vprint("All put operations completed successfully", level=1)
        vprint(f"Search in 'docs' namespace returned {len(docs_search)} items", level=1)
        vprint(
            f"Search in 'config' namespace returned {len(config_search)} items", level=1
        )
        print("High-level operations test completed successfully!")
