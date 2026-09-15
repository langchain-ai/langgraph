"""
Tests for the LangGraph A2A (Agent-to-Agent) Protocol module.

This test module covers:
- A2ACapabilities validation and serialization
- AgentCard creation and discovery
- A2A message types and serialization
- A2AProtocolHandler integration with LangGraph
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from typing_extensions import TypedDict

from langgraph.a2a import (
    A2ACapabilities,
    A2AMessage,
    A2AMessageType,
    A2AProtocolHandler,
    A2ARequest,
    A2AResponse,
    A2ATaskStatus,
    AgentCard,
)
from langgraph.a2a.card import AgentEndpoint
from langgraph.graph.state import StateGraph

# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleState(TypedDict):
    """Simple state for testing."""

    input: str
    output: str


def simple_node(state: SimpleState) -> dict:
    """A simple node that echoes input to output."""
    return {"output": f"Processed: {state.get('input', '')}"}


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    graph = StateGraph(SimpleState)
    graph.add_node("process", simple_node)
    graph.add_edge("__start__", "process")
    return graph.compile()


@pytest.fixture
def sample_capabilities():
    """Create sample A2A capabilities."""
    return A2ACapabilities(
        name="test-agent",
        description="A test agent for unit testing",
        skills=("echo", "transform", "analyze"),
        version="1.0.0",
        tags=("test", "demo"),
        metadata={"environment": "test"},
    )


# =============================================================================
# A2ACapabilities Tests
# =============================================================================


class TestA2ACapabilities:
    """Tests for A2ACapabilities class."""

    def test_basic_creation(self) -> None:
        """Test basic capabilities creation."""
        caps = A2ACapabilities(
            name="my-agent",
            description="Test description",
            skills=("skill1", "skill2"),
        )
        assert caps.name == "my-agent"
        assert caps.description == "Test description"
        assert caps.skills == ("skill1", "skill2")
        assert caps.version == "1.0.0"

    def test_name_validation_empty(self) -> None:
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            A2ACapabilities(name="")

    def test_name_validation_special_chars(self) -> None:
        """Test that special characters in name raise error."""
        with pytest.raises(ValueError, match="must be URL-safe"):
            A2ACapabilities(name="my agent!")  # space and ! are invalid

    def test_name_validation_valid(self) -> None:
        """Test that valid names work."""
        # These should not raise
        A2ACapabilities(name="my-agent")
        A2ACapabilities(name="my_agent")
        A2ACapabilities(name="myAgent123")

    def test_get_skill_names_strings(self) -> None:
        """Test getting skill names from string skills."""
        caps = A2ACapabilities(
            name="test",
            skills=("skill1", "skill2", "skill3"),
        )
        assert caps.get_skill_names() == ["skill1", "skill2", "skill3"]

    def test_get_skill_names_dicts(self) -> None:
        """Test getting skill names from dict skills."""
        caps = A2ACapabilities(
            name="test",
            skills=(
                {"name": "search", "description": "Search the web"},
                {"name": "summarize", "description": "Summarize text"},
            ),
        )
        assert caps.get_skill_names() == ["search", "summarize"]

    def test_get_skill_names_mixed(self) -> None:
        """Test getting skill names from mixed skills."""
        caps = A2ACapabilities(
            name="test",
            skills=(
                "simple_skill",
                {"name": "complex_skill", "description": "A complex skill"},
            ),
        )
        assert caps.get_skill_names() == ["simple_skill", "complex_skill"]

    def test_has_skill(self) -> None:
        """Test has_skill method."""
        caps = A2ACapabilities(
            name="test",
            skills=("search", "summarize"),
        )
        assert caps.has_skill("search") is True
        assert caps.has_skill("summarize") is True
        assert caps.has_skill("translate") is False

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        caps = A2ACapabilities(
            name="test-agent",
            description="Test",
            skills=("skill1",),
            version="2.0.0",
            tags=("tag1",),
            metadata={"key": "value"},
        )
        data = caps.to_dict()
        assert data["name"] == "test-agent"
        assert data["description"] == "Test"
        assert data["skills"] == ["skill1"]
        assert data["version"] == "2.0.0"
        assert data["tags"] == ["tag1"]
        assert data["metadata"] == {"key": "value"}

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "name": "restored-agent",
            "description": "Restored from dict",
            "skills": ["skill1", "skill2"],
            "version": "1.5.0",
            "tags": ["test"],
            "metadata": {"restored": True},
        }
        caps = A2ACapabilities.from_dict(data)
        assert caps.name == "restored-agent"
        assert caps.description == "Restored from dict"
        assert caps.skills == ("skill1", "skill2")
        assert caps.version == "1.5.0"

    def test_roundtrip_serialization(
        self, sample_capabilities: A2ACapabilities
    ) -> None:
        """Test that to_dict and from_dict are inverses."""
        data = sample_capabilities.to_dict()
        restored = A2ACapabilities.from_dict(data)
        assert restored.name == sample_capabilities.name
        assert restored.description == sample_capabilities.description
        assert restored.skills == sample_capabilities.skills
        assert restored.version == sample_capabilities.version


# =============================================================================
# AgentCard Tests
# =============================================================================


class TestAgentCard:
    """Tests for AgentCard class."""

    def test_basic_creation(self, sample_capabilities: A2ACapabilities) -> None:
        """Test basic agent card creation."""
        card = AgentCard(
            agent_id="agent-001",
            capabilities=sample_capabilities,
        )
        assert card.agent_id == "agent-001"
        assert card.capabilities == sample_capabilities
        assert card.endpoints == ()

    def test_agent_id_validation(self, sample_capabilities: A2ACapabilities) -> None:
        """Test that empty agent_id raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            AgentCard(agent_id="", capabilities=sample_capabilities)

    def test_with_endpoints(self, sample_capabilities: A2ACapabilities) -> None:
        """Test agent card with endpoints."""
        endpoint = AgentEndpoint(
            url="https://api.example.com/agent",
            protocol="https",
            methods=("POST", "GET"),
        )
        card = AgentCard(
            agent_id="agent-001",
            capabilities=sample_capabilities,
            endpoints=(endpoint,),
        )
        assert len(card.endpoints) == 1
        assert card.endpoints[0].url == "https://api.example.com/agent"

    def test_has_endpoint(self, sample_capabilities: A2ACapabilities) -> None:
        """Test has_endpoint method."""
        card = AgentCard(
            agent_id="agent-001",
            capabilities=sample_capabilities,
            endpoints=(
                AgentEndpoint(url="https://api.example.com", protocol="https"),
                AgentEndpoint(url="grpc://api.example.com", protocol="grpc"),
            ),
        )
        assert card.has_endpoint("https") is True
        assert card.has_endpoint("grpc") is True
        assert card.has_endpoint("http") is False

    def test_get_endpoint(self, sample_capabilities: A2ACapabilities) -> None:
        """Test get_endpoint method."""
        https_endpoint = AgentEndpoint(url="https://api.example.com", protocol="https")
        card = AgentCard(
            agent_id="agent-001",
            capabilities=sample_capabilities,
            endpoints=(https_endpoint,),
        )
        assert card.get_endpoint("https") == https_endpoint
        assert card.get_endpoint("grpc") is None

    def test_is_expired_no_expiry(self, sample_capabilities: A2ACapabilities) -> None:
        """Test is_expired when no expiry is set."""
        card = AgentCard(
            agent_id="agent-001",
            capabilities=sample_capabilities,
        )
        assert card.is_expired() is False

    def test_is_expired_future(self, sample_capabilities: A2ACapabilities) -> None:
        """Test is_expired with future expiry."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        card = AgentCard(
            agent_id="agent-001",
            capabilities=sample_capabilities,
            expires_at=future.isoformat(),
        )
        assert card.is_expired() is False

    def test_is_expired_past(self, sample_capabilities: A2ACapabilities) -> None:
        """Test is_expired with past expiry."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        card = AgentCard(
            agent_id="agent-001",
            capabilities=sample_capabilities,
            expires_at=past.isoformat(),
        )
        assert card.is_expired() is True

    def test_to_dict(self, sample_capabilities: A2ACapabilities) -> None:
        """Test serialization to dictionary."""
        card = AgentCard(
            agent_id="agent-001",
            capabilities=sample_capabilities,
            endpoints=(AgentEndpoint(url="https://test.com"),),
        )
        data = card.to_dict()
        assert data["agent_id"] == "agent-001"
        assert data["protocol_version"] == "1.0"
        assert "capabilities" in data
        assert len(data["endpoints"]) == 1

    def test_to_json(self, sample_capabilities: A2ACapabilities) -> None:
        """Test JSON serialization."""
        card = AgentCard(
            agent_id="agent-001",
            capabilities=sample_capabilities,
        )
        json_str = card.to_json()
        parsed = json.loads(json_str)
        assert parsed["agent_id"] == "agent-001"

    def test_from_json(self, sample_capabilities: A2ACapabilities) -> None:
        """Test JSON deserialization."""
        card = AgentCard(
            agent_id="agent-001",
            capabilities=sample_capabilities,
        )
        json_str = card.to_json()
        restored = AgentCard.from_json(json_str)
        assert restored.agent_id == card.agent_id
        assert restored.capabilities.name == card.capabilities.name


# =============================================================================
# A2AMessage Tests
# =============================================================================


class TestA2AMessage:
    """Tests for A2A message types."""

    def test_message_creation(self) -> None:
        """Test basic message creation."""
        msg = A2AMessage(
            message_type=A2AMessageType.TASK_CREATE,
            payload={"skill": "search", "input": {"query": "test"}},
        )
        assert msg.message_type == A2AMessageType.TASK_CREATE
        assert msg.payload["skill"] == "search"
        assert msg.message_id  # Should be auto-generated

    def test_message_to_dict(self) -> None:
        """Test message serialization."""
        msg = A2AMessage(
            message_type=A2AMessageType.TASK_CREATE,
            payload={"key": "value"},
            sender_id="agent-001",
        )
        data = msg.to_dict()
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "task.create"
        assert data["params"] == {"key": "value"}
        assert data["sender_id"] == "agent-001"

    def test_message_from_dict(self) -> None:
        """Test message deserialization."""
        data = {
            "id": "msg-001",
            "method": "task.create",
            "params": {"skill": "echo"},
            "sender_id": "agent-001",
        }
        msg = A2AMessage.from_dict(data)
        assert msg.message_id == "msg-001"
        assert msg.message_type == A2AMessageType.TASK_CREATE
        assert msg.payload["skill"] == "echo"


class TestA2ARequest:
    """Tests for A2A request type."""

    def test_request_creation(self) -> None:
        """Test basic request creation."""
        req = A2ARequest(
            skill="search",
            input_data={"query": "test query"},
        )
        assert req.skill == "search"
        assert req.input_data == {"query": "test query"}
        assert req.task_id  # Should be auto-generated
        assert req.priority == 0

    def test_request_with_all_fields(self) -> None:
        """Test request with all fields."""
        req = A2ARequest(
            skill="analyze",
            input_data={"data": [1, 2, 3]},
            task_id="task-123",
            context={"user_id": "user-001"},
            timeout_seconds=30.0,
            priority=5,
            sender_agent_id="requester-agent",
        )
        assert req.task_id == "task-123"
        assert req.timeout_seconds == 30.0
        assert req.priority == 5

    def test_request_to_message(self) -> None:
        """Test converting request to message."""
        req = A2ARequest(
            skill="echo",
            input_data={"text": "hello"},
            sender_agent_id="agent-001",
        )
        msg = req.to_message()
        assert msg.message_type == A2AMessageType.TASK_CREATE
        assert msg.payload["skill"] == "echo"
        assert msg.sender_id == "agent-001"

    def test_request_from_message(self) -> None:
        """Test creating request from message."""
        msg = A2AMessage(
            message_type=A2AMessageType.TASK_CREATE,
            payload={
                "task_id": "task-456",
                "skill": "summarize",
                "input": {"text": "long text..."},
            },
            sender_id="requester",
        )
        req = A2ARequest.from_message(msg)
        assert req.task_id == "task-456"
        assert req.skill == "summarize"
        assert req.sender_agent_id == "requester"


class TestA2AResponse:
    """Tests for A2A response type."""

    def test_success_response(self) -> None:
        """Test creating successful response."""
        resp = A2AResponse.success(
            task_id="task-001",
            result={"output": "processed"},
            responder_agent_id="agent-001",
        )
        assert resp.status == A2ATaskStatus.COMPLETED
        assert resp.result == {"output": "processed"}
        assert resp.error is None

    def test_error_response(self) -> None:
        """Test creating error response."""
        resp = A2AResponse.create_error(
            task_id="task-001",
            error_message="Something went wrong",
            error_code="ERR_001",
        )
        assert resp.status == A2ATaskStatus.FAILED
        assert resp.error["message"] == "Something went wrong"
        assert resp.error["code"] == "ERR_001"
        assert resp.result is None

    def test_response_to_message(self) -> None:
        """Test converting response to message."""
        resp = A2AResponse.success(
            task_id="task-001",
            result={"data": "result"},
        )
        msg = resp.to_message(correlation_id="req-001")
        assert msg.message_type == A2AMessageType.TASK_RESULT
        assert msg.payload["status"] == "completed"
        assert msg.correlation_id == "req-001"

    def test_running_response_message_type(self) -> None:
        """Test that running responses use TASK_STATUS type."""
        resp = A2AResponse(
            task_id="task-001",
            status=A2ATaskStatus.RUNNING,
            progress=50.0,
        )
        msg = resp.to_message()
        assert msg.message_type == A2AMessageType.TASK_STATUS


# =============================================================================
# A2AProtocolHandler Tests
# =============================================================================


class TestA2AProtocolHandler:
    """Tests for A2AProtocolHandler class."""

    def test_handler_creation(self, simple_graph, sample_capabilities) -> None:
        """Test basic handler creation."""
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=sample_capabilities,
            agent_id="handler-agent-001",
        )
        assert handler.agent_id == "handler-agent-001"
        assert handler.capabilities == sample_capabilities

    def test_handler_default_agent_id(self, simple_graph, sample_capabilities) -> None:
        """Test that agent_id defaults to capabilities name."""
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=sample_capabilities,
        )
        assert handler.agent_id == sample_capabilities.name

    def test_get_agent_card(self, simple_graph, sample_capabilities) -> None:
        """Test agent card generation."""
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=sample_capabilities,
            agent_id="card-test-agent",
        )
        card = handler.get_agent_card()
        assert card.agent_id == "card-test-agent"
        assert card.capabilities == sample_capabilities

    def test_can_handle_skill(self, simple_graph, sample_capabilities) -> None:
        """Test skill checking."""
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=sample_capabilities,
        )
        assert handler.can_handle_skill("echo") is True
        assert handler.can_handle_skill("transform") is True
        assert handler.can_handle_skill("unknown_skill") is False

    def test_handle_discover_message(self, simple_graph, sample_capabilities) -> None:
        """Test handling discovery messages."""
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=sample_capabilities,
        )
        msg = A2AMessage(
            message_type=A2AMessageType.AGENT_DISCOVER,
            payload={},
        )
        response = handler.handle_message(msg)
        assert response.message_type == A2AMessageType.AGENT_CARD
        assert response.payload["agent_id"] == sample_capabilities.name

    def test_handle_capability_query(self, simple_graph, sample_capabilities) -> None:
        """Test handling capability query messages."""
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=sample_capabilities,
        )
        # Query for a skill we have
        msg = A2AMessage(
            message_type=A2AMessageType.CAPABILITY_QUERY,
            payload={"skill": "echo"},
        )
        response = handler.handle_message(msg)
        assert response.message_type == A2AMessageType.CAPABILITY_RESPONSE
        assert response.payload["has_skill"] is True

        # Query for a skill we don't have
        msg = A2AMessage(
            message_type=A2AMessageType.CAPABILITY_QUERY,
            payload={"skill": "unknown"},
        )
        response = handler.handle_message(msg)
        assert response.payload["has_skill"] is False

    def test_handle_request_success(self, simple_graph) -> None:
        """Test successful request handling."""
        caps = A2ACapabilities(
            name="echo-agent",
            skills=("process",),
        )
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=caps,
        )
        request = A2ARequest(
            skill="process",
            input_data={"input": "hello world"},
        )
        response = handler.handle_request(request)
        assert response.status == A2ATaskStatus.COMPLETED
        assert "output" in response.result

    def test_handle_request_skill_not_found(
        self, simple_graph, sample_capabilities
    ) -> None:
        """Test request with unknown skill."""
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=sample_capabilities,
        )
        request = A2ARequest(
            skill="unknown_skill",
            input_data={},
        )
        response = handler.handle_request(request)
        assert response.status == A2ATaskStatus.FAILED
        assert response.error["code"] == "SKILL_NOT_FOUND"

    def test_handle_unsupported_message_type(
        self, simple_graph, sample_capabilities
    ) -> None:
        """Test handling unsupported message types."""
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=sample_capabilities,
        )
        msg = A2AMessage(
            message_type=A2AMessageType.TASK_CANCEL,  # Not implemented
            payload={},
        )
        response = handler.handle_message(msg)
        assert response.message_type == A2AMessageType.ERROR
        assert "UNSUPPORTED" in response.payload["code"]


@pytest.mark.anyio
class TestA2AProtocolHandlerAsync:
    """Async tests for A2AProtocolHandler."""

    async def test_ahandle_request_success(self, simple_graph) -> None:
        """Test async request handling."""
        caps = A2ACapabilities(
            name="async-agent",
            skills=("process",),
        )
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=caps,
        )
        request = A2ARequest(
            skill="process",
            input_data={"input": "async hello"},
        )
        response = await handler.ahandle_request(request)
        assert response.status == A2ATaskStatus.COMPLETED
        assert response.result is not None

    async def test_ahandle_request_skill_not_found(self, simple_graph) -> None:
        """Test async request with unknown skill."""
        caps = A2ACapabilities(
            name="async-agent",
            skills=("other_skill",),
        )
        handler = A2AProtocolHandler(
            graph=simple_graph,
            capabilities=caps,
        )
        request = A2ARequest(
            skill="unknown",
            input_data={},
        )
        response = await handler.ahandle_request(request)
        assert response.status == A2ATaskStatus.FAILED


# =============================================================================
# Integration Tests
# =============================================================================


class TestA2AIntegration:
    """Integration tests for full A2A workflow."""

    def test_full_discovery_and_invoke_workflow(self) -> None:
        """Test complete workflow: discovery -> invoke -> response."""
        # 1. Create a graph
        graph = StateGraph(SimpleState)
        graph.add_node("process", simple_node)
        graph.add_edge("__start__", "process")
        compiled = graph.compile()

        # 2. Create capabilities and handler
        caps = A2ACapabilities(
            name="integration-agent",
            description="An agent for integration testing",
            skills=("process",),
        )
        handler = A2AProtocolHandler(
            graph=compiled,
            capabilities=caps,
            endpoints=[AgentEndpoint(url="https://test.local/agent")],
        )

        # 3. Discover the agent
        discover_msg = A2AMessage(
            message_type=A2AMessageType.AGENT_DISCOVER,
            payload={},
        )
        card_response = handler.handle_message(discover_msg)
        assert card_response.message_type == A2AMessageType.AGENT_CARD

        # 4. Check agent has the skill we need
        card_data = card_response.payload
        assert "process" in card_data["capabilities"]["skills"]

        # 5. Invoke the agent
        request = A2ARequest(
            skill="process",
            input_data={"input": "integration test input"},
        )
        task_msg = request.to_message()
        task_response = handler.handle_message(task_msg)

        # 6. Verify successful response
        response = A2AResponse.from_message(task_response)
        assert response.status == A2ATaskStatus.COMPLETED
        assert "Processed: integration test input" in response.result.get("output", "")

    def test_agent_card_json_roundtrip(self, sample_capabilities) -> None:
        """Test that agent cards survive JSON serialization."""
        original_card = AgentCard(
            agent_id="roundtrip-agent",
            capabilities=sample_capabilities,
            endpoints=(AgentEndpoint(url="https://api.test.com"),),
            metadata={"custom": "data"},
        )

        # Serialize to JSON and back
        json_str = original_card.to_json()
        restored_card = AgentCard.from_json(json_str)

        # Verify all fields match
        assert restored_card.agent_id == original_card.agent_id
        assert restored_card.capabilities.name == original_card.capabilities.name
        assert len(restored_card.endpoints) == len(original_card.endpoints)
        assert restored_card.metadata == original_card.metadata
