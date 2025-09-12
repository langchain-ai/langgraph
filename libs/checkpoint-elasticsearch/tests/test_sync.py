from unittest.mock import MagicMock, patch

import pytest

from langgraph.checkpoint.elasticsearch.sync import ElasticsearchSaver


class TestElasticsearchSaver:
    """Test suite for the synchronous ElasticsearchSaver."""

    def test_init_with_params(self, es_config, mock_elasticsearch):
        """Test initialization with explicit parameters."""
        saver = ElasticsearchSaver(
            es_url=es_config["es_url"],
            api_key=es_config["api_key"],
            index_prefix=es_config["index_prefix"],
        )

        assert saver.checkpoints_index == f"{es_config['index_prefix']}_checkpoints"
        assert saver.writes_index == f"{es_config['index_prefix']}_writes"
        # Verify the mock Elasticsearch was called
        mock_elasticsearch.Elasticsearch.assert_called_once()

    def test_init_with_env_vars(self, es_config):
        """Test initialization with environment variables."""
        with patch.dict(
            "os.environ",
            {"ES_URL": es_config["es_url"], "ES_API_KEY": es_config["api_key"]},
        ):
            with patch("elasticsearch.Elasticsearch") as mock_es_class:
                mock_client = MagicMock()
                mock_es_class.return_value = mock_client
                mock_client.indices.exists.return_value = True

                saver = ElasticsearchSaver()

                assert saver.checkpoints_index == "langgraph_checkpoints"
                assert saver.writes_index == "langgraph_writes"

    def test_init_missing_url(self):
        """Test initialization fails without ES_URL."""
        with pytest.raises(ValueError, match="ES_URL must be provided"):
            ElasticsearchSaver(api_key="test-key")

    def test_init_missing_api_key(self):
        """Test initialization fails without ES_API_KEY."""
        with pytest.raises(ValueError, match="ES_API_KEY must be provided"):
            ElasticsearchSaver(es_url="https://localhost:9200")

    def test_get_document_id(self, es_config):
        """Test document ID generation."""
        with patch("elasticsearch.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_client.indices.exists.return_value = True

            saver = ElasticsearchSaver(**es_config)
            doc_id = saver._get_document_id("thread1", "ns1", "checkpoint1")

            assert doc_id == "thread1#ns1#checkpoint1"

    def test_get_tuple_found(self, es_config, mock_es_response, sample_config):
        """Test getting an existing checkpoint tuple."""
        with patch("elasticsearch.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_client.indices.exists.return_value = True

            # Mock get response
            mock_client.get.return_value = mock_es_response

            # Mock search for writes (empty)
            mock_client.search.return_value = {"hits": {"hits": []}}

            saver = ElasticsearchSaver(**es_config)

            # Mock the serde to return a simple checkpoint
            saver.serde.loads_typed = MagicMock(
                return_value={
                    "v": 1,
                    "id": "test-checkpoint-1",
                    "ts": "2024-01-01T00:00:00Z",
                    "channel_values": {"test": "value"},
                    "channel_versions": {"test": "1"},
                    "versions_seen": {},
                    "pending_sends": [],
                }
            )

            result = saver.get_tuple(sample_config)

            assert result is not None
            assert result.config == sample_config
            assert result.checkpoint["id"] == "test-checkpoint-1"
            assert result.metadata["source"] == "input"
            assert result.pending_writes == []

    def test_get_tuple_not_found(self, es_config, sample_config):
        """Test getting a non-existent checkpoint tuple."""
        from elasticsearch.exceptions import NotFoundError

        with patch("elasticsearch.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_client.indices.exists.return_value = True

            # Mock get to raise NotFoundError
            mock_client.get.side_effect = NotFoundError("Not found")

            saver = ElasticsearchSaver(**es_config)
            result = saver.get_tuple(sample_config)

            assert result is None

    def test_get_tuple_latest(self, es_config, mock_es_response):
        """Test getting the latest checkpoint for a thread."""
        with patch("elasticsearch.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_client.indices.exists.return_value = True

            # Config without checkpoint_id to get latest
            config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": ""}}

            # Mock search response for latest checkpoint
            mock_client.search.return_value = {"hits": {"hits": [mock_es_response]}}

            saver = ElasticsearchSaver(**es_config)

            # Mock the serde
            saver.serde.loads_typed = MagicMock(
                return_value={
                    "v": 1,
                    "id": "test-checkpoint-1",
                    "ts": "2024-01-01T00:00:00Z",
                    "channel_values": {"test": "value"},
                    "channel_versions": {"test": "1"},
                    "versions_seen": {},
                    "pending_sends": [],
                }
            )

            result = saver.get_tuple(config)

            assert result is not None
            assert result.checkpoint["id"] == "test-checkpoint-1"

    def test_put_checkpoint(
        self, es_config, sample_checkpoint, sample_config, sample_metadata
    ):
        """Test storing a checkpoint."""
        with patch("elasticsearch.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_client.indices.exists.return_value = True

            saver = ElasticsearchSaver(**es_config)

            # Mock the serde
            saver.serde.dumps_typed = MagicMock(
                return_value=("json", b'{"test": "data"}')
            )

            result_config = saver.put(
                sample_config, sample_checkpoint, sample_metadata, {}
            )

            # Verify index was called
            mock_client.index.assert_called_once()
            call_args = mock_client.index.call_args

            assert call_args[1]["index"] == saver.checkpoints_index
            assert call_args[1]["id"] == "test-thread##test-checkpoint-1"
            assert "checkpoint_data" in call_args[1]["body"]

            # Verify returned config
            assert result_config["configurable"]["thread_id"] == "test-thread"
            assert result_config["configurable"]["checkpoint_id"] == "test-checkpoint-1"

    def test_put_writes(self, es_config, sample_config):
        """Test storing writes."""
        with patch("elasticsearch.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_client.indices.exists.return_value = True

            saver = ElasticsearchSaver(**es_config)

            # Mock the serde
            saver.serde.dumps_typed = MagicMock(
                return_value=("json", b'{"test": "write"}')
            )

            writes = [
                ("channel1", {"data": "value1"}),
                ("channel2", {"data": "value2"}),
            ]

            saver.put_writes(sample_config, writes, "task-1", "path")

            # Verify bulk was called
            mock_client.bulk.assert_called_once()
            bulk_body = mock_client.bulk.call_args[1]["body"]

            # Should have 2 writes * 2 (operation + document) = 4 items
            assert len(bulk_body) == 4

    def test_list_checkpoints(self, es_config, mock_es_response):
        """Test listing checkpoints."""
        with patch("elasticsearch.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_client.indices.exists.return_value = True

            # Mock search responses
            search_responses = [
                {"hits": {"hits": [mock_es_response]}},  # Checkpoints search
                {"hits": {"hits": []}},  # Writes search
            ]
            mock_client.search.side_effect = search_responses

            saver = ElasticsearchSaver(**es_config)

            # Mock the serde
            saver.serde.loads_typed = MagicMock(
                return_value={
                    "v": 1,
                    "id": "test-checkpoint-1",
                    "ts": "2024-01-01T00:00:00Z",
                    "channel_values": {"test": "value"},
                    "channel_versions": {"test": "1"},
                    "versions_seen": {},
                    "pending_sends": [],
                }
            )

            config = {"configurable": {"thread_id": "test-thread"}}
            checkpoints = list(saver.list(config, limit=10))

            assert len(checkpoints) == 1
            assert checkpoints[0].checkpoint["id"] == "test-checkpoint-1"

    def test_delete_thread(self, es_config):
        """Test deleting all data for a thread."""
        with patch("elasticsearch.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_client.indices.exists.return_value = True

            saver = ElasticsearchSaver(**es_config)

            saver.delete_thread("test-thread")

            # Verify delete_by_query was called twice (checkpoints and writes)
            assert mock_client.delete_by_query.call_count == 2

    @pytest.mark.asyncio
    async def test_async_methods_not_implemented(self, es_config):
        """Test that async methods raise NotImplementedError."""
        with patch("elasticsearch.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_client.indices.exists.return_value = True

            saver = ElasticsearchSaver(**es_config)

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
