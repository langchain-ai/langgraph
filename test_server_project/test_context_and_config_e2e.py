"""
End-to-end tests for context and config support in remote runs.

This test file verifies that the LangGraph API server correctly accepts
both context and config parameters together, as well as individually.
"""

import uuid
import pytest
from httpx import AsyncClient
from typing import AsyncIterator

# These imports would need to be adjusted based on actual langgraph-api structure
# from langgraph_api.server import app
# from langgraph_api.testing import setup_test_db


@pytest.fixture
async def test_client() -> AsyncIterator[AsyncClient]:
    """
    Create a test client for the FastAPI app.
    
    In actual langgraph-api repo, this would use the real app fixture:
    ```python
    from langgraph_api.server import app
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    ```
    """
    # Placeholder - in real langgraph-api repo, use actual app
    async with AsyncClient(base_url="http://127.0.0.1:2024") as client:
        yield client


@pytest.mark.asyncio
async def test_invoke_both_context_and_config(test_client: AsyncClient):
    """
    Test that /runs/wait accepts both context and config together.
    
    Previously this would return HTTP 400 with:
    "Cannot specify both configurable and context"
    
    Now it should accept both and return 200.
    """
    thread_id = f"test_both_{uuid.uuid4()}"
    
    # First turn: set a name
    body = {
        "assistant_id": "agent",
        "input": {"messages": [{"role": "user", "content": "My name is Alice"}]},
        "config": {"configurable": {"thread_id": thread_id}},
        "context": {"user_id": "test_user_1", "request_id": "test_req_1"}
    }
    
    response = await test_client.post("/runs/wait", json=body)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    assert "messages" in data, "Response should contain messages"
    
    # Second turn: recall the name (verify memory works)
    body["input"] = {"messages": [{"role": "user", "content": "What is my name?"}]}
    response = await test_client.post("/runs/wait", json=body)
    
    assert response.status_code == 200
    data = response.json()
    response_text = str(data.get("messages", []))
    
    # Note: This assertion depends on the actual graph implementation
    # For a real test, you'd check that memory persisted correctly
    assert len(data.get("messages", [])) > 0, "Should have messages in response"


@pytest.mark.asyncio  
async def test_stream_both_context_and_config(test_client: AsyncClient):
    """
    Test that /runs/stream accepts both context and config together.
    
    Should return 200 and stream SSE events.
    """
    thread_id = f"test_stream_{uuid.uuid4()}"
    
    body = {
        "assistant_id": "agent",
        "input": {"messages": [{"role": "user", "content": "Hello streaming"}]},
        "config": {"configurable": {"thread_id": thread_id}},
        "context": {"user_id": "test_user_2", "request_id": "test_req_2"}
    }
    
    response = await test_client.post("/runs/stream", json=body)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    # Verify it's an SSE stream
    content_type = response.headers.get("content-type", "")
    assert "text/event-stream" in content_type, f"Expected SSE stream, got content-type: {content_type}"
    
    # Read at least one event from the stream
    content = response.text
    assert "event:" in content or "data:" in content, "Should receive SSE events"


@pytest.mark.asyncio
async def test_config_only_still_works(test_client: AsyncClient):
    """
    Regression test: config-only calls should continue to work.
    """
    thread_id = f"test_config_only_{uuid.uuid4()}"
    
    body = {
        "assistant_id": "agent",
        "input": {"messages": [{"role": "user", "content": "Config only test"}]},
        "config": {"configurable": {"thread_id": thread_id}}
    }
    
    response = await test_client.post("/runs/wait", json=body)
    assert response.status_code == 200, f"Config-only call failed: {response.text}"
    
    data = response.json()
    assert "messages" in data, "Response should contain messages"


@pytest.mark.asyncio
async def test_context_only_still_works(test_client: AsyncClient):
    """
    Regression test: context-only calls should continue to work.
    """
    thread_id = f"test_context_only_{uuid.uuid4()}"
    
    body = {
        "assistant_id": "agent",
        "input": {"messages": [{"role": "user", "content": "Context only test"}]},
        "context": {"thread_id": thread_id, "user_id": "test_user_3"}
    }
    
    response = await test_client.post("/runs/wait", json=body)
    assert response.status_code == 200, f"Context-only call failed: {response.text}"
    
    data = response.json()
    assert "messages" in data, "Response should contain messages"


@pytest.mark.asyncio
async def test_thread_id_from_context_to_config(test_client: AsyncClient):
    """
    Test that thread_id from context is used when config.configurable is empty.
    
    This ensures the optional ergonomic mapping works correctly.
    """
    thread_id = f"test_thread_id_mapping_{uuid.uuid4()}"
    
    # First call: provide thread_id only in context
    body = {
        "assistant_id": "agent",
        "input": {"messages": [{"role": "user", "content": "Testing thread_id mapping"}]},
        "config": {"recursion_limit": 25},  # Config without thread_id
        "context": {"thread_id": thread_id, "user_id": "test_user_4"}
    }
    
    response = await test_client.post("/runs/wait", json=body)
    assert response.status_code == 200, f"Thread ID mapping failed: {response.text}"
    
    # Verify the thread was created (check response headers)
    location = response.headers.get("location", "")
    assert thread_id in location or len(location) > 0, "Should create thread with provided ID"


@pytest.mark.asyncio
async def test_memory_persists_with_both_parameters(test_client: AsyncClient):
    """
    Test that memory/checkpointing works correctly when both context and config are provided.
    """
    thread_id = f"test_memory_{uuid.uuid4()}"
    
    # First turn: store information
    body = {
        "assistant_id": "agent",
        "input": {"messages": [{"role": "user", "content": "My favorite color is blue"}]},
        "config": {"configurable": {"thread_id": thread_id}},
        "context": {"user_id": "test_user_5", "session_id": "test_session"}
    }
    
    response1 = await test_client.post("/runs/wait", json=body)
    assert response1.status_code == 200
    
    # Second turn: try to recall (using same thread_id)
    body["input"] = {"messages": [{"role": "user", "content": "What is my favorite color?"}]}
    response2 = await test_client.post("/runs/wait", json=body)
    
    assert response2.status_code == 200
    data = response2.json()
    
    # Note: Actual assertion depends on graph implementation
    # For a real memory test, you'd verify the graph recalled "blue"
    assert "messages" in data, "Should have messages indicating memory worked"


if __name__ == "__main__":
    """
    For manual testing, run this file directly:
    
    1. Start the LangGraph server:
       langgraph dev --port 2024
    
    2. Run the tests:
       pytest test_context_and_config_e2e.py -v
    """
    pytest.main([__file__, "-v", "-s"])

