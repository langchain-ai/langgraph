import pytest
from unittest.mock import MagicMock, AsyncMock

from langgraph.langgraph.pregel.remote import RemoteGraph

def test_stream_forwards_durability_to_sync_client():
    mock_sync_client = MagicMock()
    # make the iterator empty so loop exits
    mock_sync_client.runs.stream.return_value = []

    remote = RemoteGraph("test_graph_id", sync_client=mock_sync_client)

    # call stream with durability
    list(remote.stream({"input": "data"}, config={"configurable": {"thread_id": "thread_1"}}, durability="sync"))

    # assert durability was forwarded as a kwarg
    assert mock_sync_client.runs.stream.call_args is not None
    assert mock_sync_client.runs.stream.call_args.kwargs.get("durability") == "sync"
@pytest.mark.anyio
async def test_astream_forwards_durability_to_async_client():
    mock_async_client = MagicMock()
    async_iter = AsyncMock()
    async_iter.__aiter__.return_value = async_iter
    async_iter.__anext__.side_effect = StopAsyncIteration
    mock_async_client.runs.stream.return_value = async_iter
    remote = RemoteGraph("test_graph_id", client=mock_async_client)

    # exhaust the async generator
    async for _ in remote.astream({"input": "data"}, config={"configurable": {"thread_id": "thread_1"}}, durability="sync"):
        pass

    assert mock_async_client.runs.stream.call_args is not None
    assert mock_async_client.runs.stream.call_args.kwargs.get("durability") == "sync"
    
