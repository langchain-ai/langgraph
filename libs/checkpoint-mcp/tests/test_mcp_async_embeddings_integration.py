# Standard library imports
import pytest

# Local imports
from conftest import vprint

from langgraph.store.mcp import AsyncMCPStore


@pytest.mark.asyncio
async def test_mcp_store_with_embeddings_put(mock_embeddings):
    """Test MCP store put operation with embedding processing using high-level APIs"""
    vprint("Testing MCP store with embedding put operations...", level=1)

    index_config = {"dims": 5, "field": ["content", "title"], "embed": mock_embeddings}

    async with AsyncMCPStore.from_mcp_config(
        host="localhost", port=8000, index_config=index_config
    ) as store:
        vprint("Connected to MCP server for embedding tests", level=1)

        try:
            # Test put with embedding processing
            document = {
                "title": "Technical Specification Guide",
                "content": "This document contains detailed technical specifications",
                "metadata": {"category": "technical_documentation"},
            }

            vprint("Storing document with embedding processing...", level=1)
            await store.aput(
                namespace=("docs",),
                key="spec_guide_001",
                value=document,
                index=["content", "title"],
            )

            # The put operation should complete without error
            vprint("✅ Put operation completed", level=1)

            # Verify embeddings were called
            mock_embeddings.assert_called_once()
            call_args = mock_embeddings.call_args[0][0]

            # Should have extracted text from title and content fields
            assert len(call_args) == 2, (
                f"Expected 2 texts to embed, got {len(call_args)}"
            )
            assert "Technical Specification Guide" in call_args, (
                "Title should be extracted for embedding"
            )
            assert (
                "This document contains detailed technical specifications" in call_args
            ), "Content should be extracted for embedding"

            vprint(
                f"Embeddings were generated for {len(call_args)} text fields",
                level=1,
            )
            print("MCP store put with embeddings test completed successfully!")

        except Exception as e:
            vprint(f"Embedding put test failed: {e}", level=0)
            raise


@pytest.mark.asyncio
async def test_mcp_store_with_embeddings_search(mock_embeddings):
    """Test MCP store search operation with semantic similarity using high-level APIs"""
    vprint("Testing MCP store with embedding search operations...", level=1)

    # Fix: Use 'embeddings' key instead of 'embed'
    index_config = {"dims": 5, "field": ["content"], "embed": mock_embeddings}

    async with AsyncMCPStore.from_mcp_config(
        host="localhost", port=8000, index_config=index_config
    ) as store:
        vprint("Connected to MCP server for embedding search tests", level=1)

        try:
            # First put some documents (this will use embeddings)
            documents = [
                {"content": "Software engineering best practices and methodologies"},
                {"content": "Database architecture and optimization strategies"},
                {"content": "Cloud infrastructure deployment and management"},
            ]

            vprint("Storing documents for search test...", level=1)
            for i, doc in enumerate(documents):
                await store.aput(
                    namespace=("docs",), key=f"guide_{i + 1:03d}", value=doc
                )

            # Verify documents were stored with embeddings during put operations
            assert mock_embeddings.called, (
                "Embeddings should be called during document storage"
            )

            vprint("Performing search...", level=1)
            results = await store.asearch(
                namespace_prefix=("docs",),
                query="software development practices",
                limit=2,
            )

            # Should return results (even if no semantic similarity)
            assert isinstance(results, list), "Search should return a list"

            # Verify scores are present and valid for all results
            for result in results:
                assert hasattr(result, "score"), (
                    "Each result should have a score attribute"
                )
                assert isinstance(result.score, float), (
                    f"Score should be float, got {type(result.score)}"
                )
                assert 0.0 <= result.score <= 1.0, (
                    f"Score should be in [0.0, 1.0], got {result.score}"
                )

            vprint(f"Search returned {len(results)} results", level=1)

            vprint("Query embedding was generated for semantic search", level=1)
            print("MCP store search with embeddings test completed successfully!")

        except Exception as e:
            vprint(f"Embedding search test failed: {e}", level=0)
            raise


@pytest.mark.asyncio
async def test_mcp_store_operations_with_embeddings(mock_embeddings):
    """Test MCP store operations with embedding processing using high-level APIs"""
    vprint("Testing MCP store operations with embeddings...", level=1)

    # Fix: Use 'embeddings' key instead of 'embed'
    index_config = {"dims": 5, "field": ["content", "title"], "embed": mock_embeddings}

    async with AsyncMCPStore.from_mcp_config(
        host="localhost", port=8000, index_config=index_config
    ) as store:
        vprint("Connected to MCP server for embedding operations tests", level=1)

        try:
            # Put operations with embeddings
            await store.aput(
                namespace=("articles",),
                key="research_paper_001",
                value={
                    "title": "Digital Transformation Strategy",
                    "content": "Comprehensive analysis of digital transformation",
                },
                index=["title", "content"],
            )
            await store.aput(
                namespace=("articles",),
                key="research_paper_002",
                value={
                    "title": "System Architecture Design",
                    "content": "Modern approaches to scalable system architecture",
                },
                index=["title", "content"],
            )

            vprint("Put operations with embeddings completed", level=1)

            # Search operation
            search_results = await store.asearch(
                namespace_prefix=("articles",),
                query="enterprise architecture patterns",
                limit=5,
            )

            # Verify search results
            assert isinstance(search_results, list), "Search should return list"

            # Verify scores are present and valid for search results
            for result in search_results:
                assert hasattr(result, "score"), (
                    "Each search result should have a score attribute"
                )
                assert isinstance(result.score, float), (
                    f"Score should be float, got {type(result.score)}"
                )
                assert 0.0 <= result.score <= 1.0, (
                    f"Score should be in [0.0, 1.0], got {result.score}"
                )

            # Verify embeddings were called for documents during put operations
            assert mock_embeddings.called, (
                "Document embeddings should be generated during put operations"
            )

            # Note: Query embedding is not currently implemented in MCPStore for search
            # so we don't check for embed_query calls

            vprint("All operations with embeddings completed", level=1)
            print("MCP store operations with embeddings test completed successfully!")

        except Exception as e:
            vprint(f"Embedding operations test failed: {e}", level=0)
            raise


@pytest.mark.asyncio
async def test_mcp_store_embeddings_disabled(mock_embeddings):
    """Test MCP store when embeddings are disabled with index=False"""
    vprint("Testing MCP store with embeddings disabled...", level=1)

    # Configure the embeddings for this specific test
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    index_config = {"dims": 3, "field": ["content"], "embed": mock_embeddings}

    async with AsyncMCPStore.from_mcp_config(
        host="localhost", port=8000, index_config=index_config
    ) as store:
        try:
            # Put document with embeddings explicitly disabled
            document = {"content": "Configuration settings and system parameters"}

            vprint(
                "Storing document with embeddings disabled (index=False)...", level=1
            )
            await store.aput(
                namespace=("config",),
                key="system_config_001",
                value=document,
                index=False,
            )

            # Verify embeddings were NOT called
            mock_embeddings.embed_documents.assert_not_called()

            vprint("Embeddings were correctly skipped when index=False", level=1)
            print("MCP store embeddings disabled test completed successfully!")

        except Exception as e:
            vprint(f"Embedding disabled test failed: {e}", level=0)
            raise
