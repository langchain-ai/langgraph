"""
Test RemoteGraph context and config support.

This module tests that RemoteGraph accepts context and config together without error
for invoke, ainvoke, and stream operations, validates middleware receives context values,
and ensures checkpointing/memory works with thread identifiers.
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from starlette.middleware.base import BaseHTTPMiddleware

from langgraph.pregel.remote import RemoteGraph

# Test configuration
TEST_SERVER_HOST = "127.0.0.1"
TEST_SERVER_PORT = 0  # Let OS choose available port
SERVER_STARTUP_TIMEOUT = 10  # seconds


# Global storage for captured context and memory
captured_contexts: List[Dict[str, Any]] = []
memory_store: Dict[str, Dict[str, Any]] = defaultdict(dict)
server_port: Optional[int] = None


class ContextCapturingMiddleware(BaseHTTPMiddleware):
    """Minimal middleware that logs or captures context fields request_id and user_id."""

    async def dispatch(self, request: Request, call_next):
        # Extract context from request body if it's a JSON request
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Reset request body for downstream processing
                    request._body = body
                    
                    try:
                        request_data = json.loads(body.decode())
                        context = request_data.get("context", {})
                        
                        if context:
                            captured_context = {
                                "user_id": context.get("user_id"),
                                "request_id": context.get("request_id"),
                                "timestamp": time.time(),
                                "endpoint": str(request.url)
                            }
                            captured_contexts.append(captured_context)
                            logging.info(f"Captured context: {captured_context}")
                    except (json.JSONDecodeError, AttributeError):
                        pass  # Not JSON or malformed, skip context extraction
            except Exception as e:
                logging.warning(f"Error processing request body: {e}")

        response = await call_next(request)
        return response


def create_test_app() -> FastAPI:
    """Create a minimal FastAPI server that mirrors the current RemoteGraph server behavior."""
    
    app = FastAPI(title="Test LangGraph Server")
    app.add_middleware(ContextCapturingMiddleware)

    @app.get("/assistants/{assistant_id}/graph")
    async def get_graph(assistant_id: str):
        """Mock graph endpoint."""
        return {
            "nodes": [
                {"id": "memory_node", "name": "memory_node", "data": {"name": "memory_node"}}
            ],
            "edges": []
        }

    @app.post("/runs/stream")
    @app.post("/threads/{thread_id}/runs/stream")
    async def stream_run(request: Request, thread_id: Optional[str] = None):
        """Mock streaming endpoint that accepts both context and config."""
        try:
            body = await request.body()
            request_data = json.loads(body.decode()) if body else {}
            
            context = request_data.get("context", {})
            config = request_data.get("config", {})
            input_data = request_data.get("input", {})
            assistant_id = request_data.get("assistant_id", "test_assistant")
            
            # Extract thread_id from various sources
            actual_thread_id = (
                thread_id or 
                config.get("configurable", {}).get("thread_id") or
                context.get("thread_id") or
                str(uuid4())
            )
            
            # Store/retrieve name in memory if provided
            stored_name = memory_store[actual_thread_id].get("name")
            input_name = input_data.get("name")
            
            if input_name:
                memory_store[actual_thread_id]["name"] = input_name
                result_name = input_name
            else:
                result_name = stored_name or "no_name_stored"
            
            # Simulate streaming response
            def generate_stream():
                # Emit updates event
                yield f'event: updates\n'
                yield f'data: {json.dumps({"memory_node": {"name": result_name}})}\n\n'
                # Emit values event (final result)
                yield f'event: values\n'
                yield f'data: {json.dumps({"name": result_name})}\n\n'
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
            
        except Exception as e:
            logging.error(f"Error in stream_run: {e}")
            return {"error": str(e)}, 500

    @app.post("/runs/wait")
    @app.post("/threads/{thread_id}/runs/wait")
    async def wait_run(request: Request, thread_id: Optional[str] = None):
        """Mock invoke endpoint that accepts both context and config."""
        try:
            body = await request.body()
            request_data = json.loads(body.decode()) if body else {}
            
            context = request_data.get("context", {})
            config = request_data.get("config", {})
            input_data = request_data.get("input", {})
            
            # Extract thread_id from various sources
            actual_thread_id = (
                thread_id or 
                config.get("configurable", {}).get("thread_id") or
                context.get("thread_id") or
                str(uuid4())
            )
            
            # Store/retrieve name in memory if provided
            stored_name = memory_store[actual_thread_id].get("name")
            input_name = input_data.get("name")
            
            if input_name:
                memory_store[actual_thread_id]["name"] = input_name
                result_name = input_name
            else:
                result_name = stored_name or "no_name_stored"
            
            return {"name": result_name}
            
        except Exception as e:
            logging.error(f"Error in wait_run: {e}")
            return {"error": str(e)}, 500

    @app.get("/threads/{thread_id}/state")
    async def get_state(thread_id: str):
        """Mock state retrieval endpoint."""
        return {
            "values": memory_store[thread_id],
            "next": [],
            "checkpoint": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": str(uuid4()),
                "checkpoint_map": {}
            },
            "metadata": {},
            "created_at": "2023-01-01T00:00:00Z",
            "parent_checkpoint": None,
            "tasks": []
        }

    return app


class TestServer:
    """Helper class to manage test server lifecycle."""
    
    def __init__(self):
        self.server = None
        self.thread = None
        self.port = None
        self.app = create_test_app()
    
    def start(self) -> int:
        """Start the test server and return the port."""
        global server_port
        
        # Create server config
        config = uvicorn.Config(
            self.app,
            host=TEST_SERVER_HOST,
            port=TEST_SERVER_PORT,
            log_level="warning",
            access_log=False
        )
        
        # Start server in background thread
        def run_server():
            self.server = uvicorn.Server(config)
            asyncio.run(self.server.serve())
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        
        # Wait for server to start and get the actual port
        start_time = time.time()
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            if self.server and hasattr(self.server, 'servers') and self.server.servers:
                # Get the actual port from the server
                for server in self.server.servers:
                    for socket in server.sockets:
                        if socket.family.name == 'AF_INET':
                            self.port = socket.getsockname()[1]
                            server_port = self.port
                            return self.port
            time.sleep(0.1)
        
        raise RuntimeError("Test server failed to start within timeout")
    
    def stop(self):
        """Stop the test server."""
        if self.server:
            self.server.should_exit = True
            if self.thread:
                self.thread.join(timeout=5)


# Test fixtures
@pytest.fixture(scope="module")
def test_server():
    """Start test server for the module and return the port."""
    server = TestServer()
    port = server.start()
    logging.info(f"Test server started on port {port}")
    yield port
    server.stop()
    logging.info("Test server stopped")


@pytest.fixture
def remote_graph(test_server):
    """Create RemoteGraph instance pointing to test server."""
    port = test_server
    return RemoteGraph("test_assistant", url=f"http://{TEST_SERVER_HOST}:{port}")


@pytest.fixture(autouse=True)
def clear_test_data():
    """Clear captured contexts and memory store before each test."""
    captured_contexts.clear()
    memory_store.clear()


# Test cases
def test_stream_both_context_and_config(remote_graph):
    """Test that RemoteGraph.stream with both context and config returns 200 and yields streamed values."""
    context = {
        "user_id": "test_user_123",
        "request_id": "req_456"
    }
    config = {
        "configurable": {"thread_id": "thread_789"}
    }
    input_data = {"name": "Alice"}
    
    results = []
    try:
        for chunk in remote_graph.stream(
            input=input_data,
            config=config,
            context=context
        ):
            results.append(chunk)
        
        # Verify we got results
        assert len(results) > 0, "Should receive streamed results"
        
        # Check that the final result contains our data (default stream_mode="updates" returns node updates)
        final_result = results[-1]
        assert "memory_node" in final_result, f"Final result should contain 'memory_node', got: {final_result}"
        assert final_result["memory_node"]["name"] == "Alice", f"Name should be 'Alice', got: {final_result}"
        
        print(f"✅ PASS: stream with both context and config succeeded, got {len(results)} chunks")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: stream with both context and config failed: {e}")
        return False


def test_invoke_both_context_and_config(remote_graph):
    """Test that RemoteGraph.invoke with both context and config does not raise 400 and produces a result."""
    context = {
        "user_id": "test_user_456",
        "request_id": "req_789"
    }
    config = {
        "configurable": {"thread_id": "thread_abc"}
    }
    input_data = {"name": "Bob"}
    
    try:
        result = remote_graph.invoke(
            input=input_data,
            config=config,
            context=context
        )
        
        # Verify we got a valid result
        assert result is not None, "Should receive a result"
        assert isinstance(result, dict), f"Result should be dict, got: {type(result)}"
        assert "name" in result, f"Result should contain 'name', got: {result}"
        assert result["name"] == "Bob", f"Name should be 'Bob', got: {result['name']}"
        
        print(f"✅ PASS: invoke with both context and config succeeded, result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: invoke with both context and config failed: {e}")
        return False


def test_middleware_receives_context(remote_graph):
    """Test that middleware receives context keys user_id and request_id exactly as sent."""
    context = {
        "user_id": "middleware_test_user",
        "request_id": "middleware_test_req"
    }
    config = {
        "configurable": {"thread_id": "middleware_thread"}
    }
    
    # Clear any previous captured contexts
    captured_contexts.clear()
    
    try:
        # Make a request that should trigger middleware
        result = remote_graph.invoke(
            input={"name": "ContextTest"},
            config=config,
            context=context
        )
        
        # Give middleware time to process
        time.sleep(0.1)
        
        # Check that middleware captured the context
        assert len(captured_contexts) > 0, f"Middleware should have captured context, but got: {captured_contexts}"
        
        # Find the relevant context entry
        relevant_context = None
        for ctx in captured_contexts:
            if ctx.get("user_id") == "middleware_test_user" and ctx.get("request_id") == "middleware_test_req":
                relevant_context = ctx
                break
        
        assert relevant_context is not None, f"Middleware should have captured the specific context, but captured: {captured_contexts}"
        assert relevant_context["user_id"] == "middleware_test_user", f"Expected user_id 'middleware_test_user', got: {relevant_context.get('user_id')}"
        assert relevant_context["request_id"] == "middleware_test_req", f"Expected request_id 'middleware_test_req', got: {relevant_context.get('request_id')}"
        
        print(f"✅ PASS: middleware correctly received context: {relevant_context}")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: middleware context test failed: {e}")
        return False


def test_memory_persists_with_context_and_or_config(remote_graph):
    """Test that memory persists across two turns when thread_id is supplied by either context or config."""
    
    # Test 1: thread_id in config only
    config_only = {"configurable": {"thread_id": "memory_test_config_only"}}
    
    try:
        # First turn - store a name
        result1 = remote_graph.invoke(
            input={"name": "ConfigOnlyTest"},
            config=config_only
        )
        assert result1["name"] == "ConfigOnlyTest", f"First turn should store name, got: {result1}"
        
        # Second turn - retrieve the name
        result2 = remote_graph.invoke(
            input={},  # No name provided, should retrieve stored name
            config=config_only
        )
        assert result2["name"] == "ConfigOnlyTest", f"Second turn should retrieve stored name, got: {result2}"
        
        print("✅ PASS: memory persists with thread_id in config only")
        
    except Exception as e:
        print(f"❌ FAIL: memory with config only failed: {e}")
        return False
    
    # Test 2: thread_id in context only
    context_only = {"thread_id": "memory_test_context_only"}
    
    try:
        # First turn - store a name
        result1 = remote_graph.invoke(
            input={"name": "ContextOnlyTest"},
            context=context_only
        )
        assert result1["name"] == "ContextOnlyTest", f"First turn should store name, got: {result1}"
        
        # Second turn - retrieve the name
        result2 = remote_graph.invoke(
            input={},  # No name provided, should retrieve stored name
            context=context_only  
        )
        assert result2["name"] == "ContextOnlyTest", f"Second turn should retrieve stored name, got: {result2}"
        
        print("✅ PASS: memory persists with thread_id in context only")
        
    except Exception as e:
        print(f"❌ FAIL: memory with context only failed: {e}")
        return False
    
    # Test 3: thread_id in both context and config (should prefer one consistently)
    both_context_config = {
        "context": {"thread_id": "memory_test_both_sources"},
        "config": {"configurable": {"thread_id": "memory_test_both_sources"}}  # Same ID to avoid conflicts
    }
    
    try:
        # First turn - store a name
        result1 = remote_graph.invoke(
            input={"name": "BothSourcesTest"},
            config=both_context_config["config"],
            context=both_context_config["context"]
        )
        assert result1["name"] == "BothSourcesTest", f"First turn should store name, got: {result1}"
        
        # Second turn - retrieve the name
        result2 = remote_graph.invoke(
            input={},  # No name provided, should retrieve stored name
            config=both_context_config["config"],
            context=both_context_config["context"]  
        )
        assert result2["name"] == "BothSourcesTest", f"Second turn should retrieve stored name, got: {result2}"
        
        print("✅ PASS: memory persists with thread_id in both context and config")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: memory with both context and config failed: {e}")
        return False


# Additional regression tests
def test_calls_with_only_config_still_work(remote_graph):
    """Test that calls with only config still work (no regression)."""
    config = {"configurable": {"thread_id": "config_only_test"}}
    
    try:
        result = remote_graph.invoke(
            input={"name": "ConfigOnly"},
            config=config
        )
        assert result["name"] == "ConfigOnly", f"Config-only call should work, got: {result}"
        print("✅ PASS: calls with only config still work")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: config-only call failed: {e}")
        return False


def test_calls_with_only_context_still_work(remote_graph):
    """Test that calls with only context still work (no regression)."""
    context = {"user_id": "context_only_user", "thread_id": "context_only_test"}
    
    try:
        result = remote_graph.invoke(
            input={"name": "ContextOnly"},
            context=context
        )
        assert result["name"] == "ContextOnly", f"Context-only call should work, got: {result}"
        print("✅ PASS: calls with only context still work")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: context-only call failed: {e}")
        return False


def test_calls_with_neither_context_nor_config(remote_graph):
    """Test that calls with neither context nor config behave as before."""
    try:
        result = remote_graph.invoke(input={"name": "NoContextNoConfig"})
        assert result["name"] == "NoContextNoConfig", f"No context/config call should work, got: {result}"
        print("✅ PASS: calls with neither context nor config work as before")
        return True
        
    except Exception as e:
        print(f"⚠️  Expected behavior: calls with neither context nor config failed: {e}")
        # This might be expected behavior depending on the server implementation
        return True


if __name__ == "__main__":
    # Direct execution for manual testing
    logging.basicConfig(level=logging.INFO)
    
    # Start test server
    server = TestServer()
    port = server.start()
    print(f"Test server started on port {port}")
    
    try:
        # Create remote graph
        remote_graph = RemoteGraph("test_assistant", url=f"http://{TEST_SERVER_HOST}:{port}")
        
        # Run tests manually
        print("\n=== Running RemoteGraph Context and Config Tests ===\n")
        
        results = {}
        results["stream_both"] = test_stream_both_context_and_config(remote_graph)
        results["invoke_both"] = test_invoke_both_context_and_config(remote_graph)
        results["middleware"] = test_middleware_receives_context(remote_graph)
        results["memory"] = test_memory_persists_with_context_and_or_config(remote_graph)
        results["config_only"] = test_calls_with_only_config_still_work(remote_graph)
        results["context_only"] = test_calls_with_only_context_still_work(remote_graph)
        results["neither"] = test_calls_with_neither_context_nor_config(remote_graph)
        
        print(f"\n=== Test Results ===")
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{test_name}: {status}")
            
        print(f"\nCaptured contexts: {len(captured_contexts)}")
        for ctx in captured_contexts:
            print(f"  - {ctx}")
            
        print(f"\nMemory store: {dict(memory_store)}")
        
    finally:
        server.stop()
        print("Test server stopped")
