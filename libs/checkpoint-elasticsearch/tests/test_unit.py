"""Unit tests for Elasticsearch checkpoint saver with proper mocking."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.checkpoint.elasticsearch.aio import AsyncElasticsearchSaver
from langgraph.checkpoint.elasticsearch.sync import ElasticsearchSaver


class TestElasticsearchSaverUnit:
    """Unit tests for synchronous ElasticsearchSaver with mocked dependencies."""

    @patch("langgraph.checkpoint.elasticsearch.base.Elasticsearch")
    def test_init_with_params(self, mock_es_class):
        """Test initialization with explicit parameters."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_es_class.return_value = mock_client

        saver = ElasticsearchSaver(
            es_url="https://localhost:9200",
            api_key="test-api-key",
            index_prefix="test_langgraph",
        )

        assert saver.checkpoints_index == "test_langgraph_checkpoints"
        assert saver.writes_index == "test_langgraph_writes"
        mock_es_class.assert_called_once()

    @patch("langgraph.checkpoint.elasticsearch.base.Elasticsearch")
    def test_init_missing_url(self, mock_es_class):
        """Test initialization fails without ES_URL."""
        with pytest.raises(ValueError, match="ES_URL must be provided"):
            ElasticsearchSaver(api_key="test-key")

    @patch("langgraph.checkpoint.elasticsearch.base.Elasticsearch")
    def test_init_missing_api_key(self, mock_es_class):
        """Test initialization fails without ES_API_KEY."""
        with pytest.raises(ValueError, match="ES_API_KEY must be provided"):
            ElasticsearchSaver(es_url="https://localhost:9200")

    @patch("langgraph.checkpoint.elasticsearch.base.Elasticsearch")
    def test_get_document_id(self, mock_es_class):
        """Test document ID generation."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_es_class.return_value = mock_client

        saver = ElasticsearchSaver(
            es_url="https://localhost:9200",
            api_key="test-api-key",
            index_prefix="test_langgraph",
        )

        doc_id = saver._get_document_id("thread1", "ns1", "checkpoint1")
        assert doc_id == "thread1#ns1#checkpoint1"

    @patch("langgraph.checkpoint.elasticsearch.base.Elasticsearch")
    def test_ensure_indices_creates_indices(self, mock_es_class):
        """Test that indices are created when they don't exist."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        mock_es_class.return_value = mock_client

        ElasticsearchSaver(
            es_url="https://localhost:9200",
            api_key="test-api-key",
        )

        # Should create both indices
        assert mock_client.indices.create.call_count == 2

    @patch("langgraph.checkpoint.elasticsearch.base.Elasticsearch")
    def test_ensure_indices_skips_existing(self, mock_es_class):
        """Test that existing indices are not recreated."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_es_class.return_value = mock_client

        ElasticsearchSaver(
            es_url="https://localhost:9200",
            api_key="test-api-key",
        )

        # Should not create any indices
        mock_client.indices.create.assert_not_called()


class TestAsyncElasticsearchSaverUnit:
    """Unit tests for asynchronous AsyncElasticsearchSaver with mocked dependencies."""

    @patch("langgraph.checkpoint.elasticsearch.aio.AsyncElasticsearch")
    def test_init_with_params(self, mock_async_es_class):
        """Test initialization with explicit parameters."""
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = True
        mock_async_es_class.return_value = mock_client

        saver = AsyncElasticsearchSaver(
            es_url="https://localhost:9200",
            api_key="test-api-key",
            index_prefix="test_langgraph",
        )

        assert saver.checkpoints_index == "test_langgraph_checkpoints"
        assert saver.writes_index == "test_langgraph_writes"
        mock_async_es_class.assert_called_once()

    @patch("langgraph.checkpoint.elasticsearch.aio.AsyncElasticsearch")
    def test_init_missing_url(self, mock_async_es_class):
        """Test initialization fails without ES_URL."""
        with pytest.raises(ValueError, match="ES_URL must be provided"):
            AsyncElasticsearchSaver(api_key="test-key")

    @patch("langgraph.checkpoint.elasticsearch.aio.AsyncElasticsearch")
    def test_init_missing_api_key(self, mock_async_es_class):
        """Test initialization fails without ES_API_KEY."""
        with pytest.raises(ValueError, match="ES_API_KEY must be provided"):
            AsyncElasticsearchSaver(es_url="https://localhost:9200")

    @patch("langgraph.checkpoint.elasticsearch.aio.AsyncElasticsearch")
    @pytest.mark.asyncio
    async def test_ensure_indices_creates_indices(self, mock_async_es_class):
        """Test that indices are created when they don't exist."""
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = False
        mock_async_es_class.return_value = mock_client

        saver = AsyncElasticsearchSaver(
            es_url="https://localhost:9200",
            api_key="test-api-key",
        )

        await saver._ensure_indices()

        # Should create both indices
        assert mock_client.indices.create.call_count == 2

    @patch("langgraph.checkpoint.elasticsearch.aio.AsyncElasticsearch")
    @pytest.mark.asyncio
    async def test_ensure_indices_skips_existing(self, mock_async_es_class):
        """Test that existing indices are not recreated."""
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = True
        mock_async_es_class.return_value = mock_client

        saver = AsyncElasticsearchSaver(
            es_url="https://localhost:9200",
            api_key="test-api-key",
        )

        await saver._ensure_indices()

        # Should not create any indices
        mock_client.indices.create.assert_not_called()

    @patch("langgraph.checkpoint.elasticsearch.aio.AsyncElasticsearch")
    @pytest.mark.asyncio
    async def test_aclose(self, mock_async_es_class):
        """Test closing the async client."""
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = True
        mock_async_es_class.return_value = mock_client

        saver = AsyncElasticsearchSaver(
            es_url="https://localhost:9200",
            api_key="test-api-key",
        )

        await saver.aclose()
        mock_client.close.assert_called_once()

    @patch("langgraph.checkpoint.elasticsearch.aio.AsyncElasticsearch")
    def test_sync_methods_not_implemented(self, mock_async_es_class):
        """Test that sync methods raise NotImplementedError."""
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = True
        mock_async_es_class.return_value = mock_client

        saver = AsyncElasticsearchSaver(
            es_url="https://localhost:9200",
            api_key="test-api-key",
        )

        with pytest.raises(NotImplementedError):
            saver.get_tuple({})

        with pytest.raises(NotImplementedError):
            list(saver.list(None))

        with pytest.raises(NotImplementedError):
            saver.put({}, {}, {}, {})

        with pytest.raises(NotImplementedError):
            saver.put_writes({}, [], "task")

        with pytest.raises(NotImplementedError):
            saver.delete_thread("thread")


class TestSyncAsyncMethodsNotImplemented:
    """Test that sync checkpointer doesn't support async methods."""

    @patch("langgraph.checkpoint.elasticsearch.base.Elasticsearch")
    @pytest.mark.asyncio
    async def test_async_methods_not_implemented(self, mock_es_class):
        """Test that async methods raise NotImplementedError."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_es_class.return_value = mock_client

        saver = ElasticsearchSaver(
            es_url="https://localhost:9200",
            api_key="test-api-key",
        )

        with pytest.raises(NotImplementedError):
            await saver.aget_tuple({})

        with pytest.raises(NotImplementedError):
            async for _ in saver.alist(None):
                pass

        with pytest.raises(NotImplementedError):
            await saver.aput({}, {}, {}, {})

        with pytest.raises(NotImplementedError):
            await saver.aput_writes({}, [], "task")

        with pytest.raises(NotImplementedError):
            await saver.adelete_thread("thread")
