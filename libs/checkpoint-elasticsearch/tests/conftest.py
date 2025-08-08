from unittest.mock import MagicMock, patch

import pytest


# Mock Elasticsearch dependencies for testing without actual ES cluster
@pytest.fixture(autouse=True)
def mock_elasticsearch():
    """Mock Elasticsearch imports for unit tests."""
    with patch("elasticsearch.Elasticsearch") as mock_sync_es:
        with patch("elasticsearch.AsyncElasticsearch") as mock_async_es:
            # Create mock clients that don't make real connections
            mock_es_client = MagicMock()
            mock_es_client.indices.exists.return_value = True
            mock_es_client.indices.create.return_value = {"acknowledged": True}
            mock_es_client.get.return_value = {"_source": {}}
            mock_es_client.search.return_value = {"hits": {"hits": []}}
            mock_es_client.index.return_value = {"_id": "test"}
            mock_es_client.bulk.return_value = {"items": []}
            mock_es_client.delete_by_query.return_value = {"deleted": 0}

            mock_async_es_client = MagicMock()
            mock_async_es_client.indices.exists.return_value = True
            mock_async_es_client.indices.create.return_value = {"acknowledged": True}
            mock_async_es_client.get.return_value = {"_source": {}}
            mock_async_es_client.search.return_value = {"hits": {"hits": []}}
            mock_async_es_client.index.return_value = {"_id": "test"}
            mock_async_es_client.bulk.return_value = {"items": []}
            mock_async_es_client.delete_by_query.return_value = {"deleted": 0}
            mock_async_es_client.close.return_value = None

            # Set up the mocks
            mock_sync_es.return_value = mock_es_client
            mock_async_es.return_value = mock_async_es_client

            yield {
                "sync_client": mock_es_client,
                "async_client": mock_async_es_client,
                "sync_class": mock_sync_es,
                "async_class": mock_async_es,
            }


@pytest.fixture
def es_config():
    """Test configuration for Elasticsearch."""
    return {
        "es_url": "https://localhost:9200",
        "api_key": "test-api-key",
        "index_prefix": "test_langgraph",
    }


@pytest.fixture
def mock_es_response():
    """Mock Elasticsearch response data."""
    return {
        "_source": {
            "thread_id": "test-thread",
            "checkpoint_ns": "",
            "checkpoint_id": "test-checkpoint-1",
            "parent_checkpoint_id": None,
            "checkpoint_type": "json",
            "checkpoint_data": "eyJ0ZXN0IjogInZhbHVlIn0=",  # base64 encoded {"test": "value"}
            "metadata": {"source": "input", "step": 1},
            "timestamp": "2024-01-01T00:00:00Z",
            "channel_versions": {},
        }
    }


@pytest.fixture
def sample_checkpoint():
    """Sample checkpoint data for testing."""
    return {
        "v": 1,
        "id": "test-checkpoint-1",
        "ts": "2024-01-01T00:00:00Z",
        "channel_values": {"test": "value"},
        "channel_versions": {"test": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }


@pytest.fixture
def sample_config():
    """Sample config for testing."""
    return {
        "configurable": {
            "thread_id": "test-thread",
            "checkpoint_ns": "",
            "checkpoint_id": "test-checkpoint-1",
        }
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {"source": "input", "step": 1, "writes": {}}
