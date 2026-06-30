"""
A2A Protocol Handler for LangGraph.

This module provides the protocol handler that enables LangGraph agents
to receive and process A2A protocol messages.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langgraph.a2a.capabilities import A2ACapabilities
from langgraph.a2a.card import AgentCard, AgentEndpoint
from langgraph.a2a.message import (
    A2AMessage,
    A2AMessageType,
    A2ARequest,
    A2AResponse,
    A2ATaskStatus,
)

if TYPE_CHECKING:
    from langgraph.pregel import Pregel


class A2AProtocolError(Exception):
    """Base exception for A2A protocol errors."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class SkillNotFoundError(A2AProtocolError):
    """Raised when a requested skill is not available."""

    def __init__(self, skill: str) -> None:
        super().__init__(f"Skill not found: {skill}", code="SKILL_NOT_FOUND")
        self.skill = skill


class A2AProtocolHandler:
    """
    Handles A2A protocol communication for a LangGraph agent.

    The A2AProtocolHandler wraps a compiled LangGraph and exposes it
    as an A2A-compliant agent, handling:
    - Agent card generation for discovery
    - Incoming task requests
    - Response formatting

    Attributes:
        graph: The compiled LangGraph agent.
        capabilities: The agent's A2A capabilities.
        agent_id: Unique identifier for this agent instance.

    Example:
        ```python
        capabilities = A2ACapabilities(
            name="research-agent",
            description="Performs web research",
            skills=["web_search"],
        )

        graph = StateGraph(MyState)
        # ... build graph
        compiled = graph.compile()

        handler = A2AProtocolHandler(
            graph=compiled,
            capabilities=capabilities,
            agent_id="research-agent-001",
        )

        # Get agent card for discovery
        card = handler.get_agent_card()

        # Handle incoming A2A request
        response = handler.handle_request(request)
        ```
    """

    def __init__(
        self,
        graph: Pregel,
        capabilities: A2ACapabilities,
        agent_id: str | None = None,
        endpoints: list[AgentEndpoint] | None = None,
        skill_mapping: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the A2A protocol handler.

        Args:
            graph: The compiled LangGraph agent.
            capabilities: The agent's A2A capabilities.
            agent_id: Unique identifier for this agent instance.
                     Defaults to capabilities.name if not provided.
            endpoints: List of endpoints where this agent can be reached.
            skill_mapping: Optional mapping from skill names to graph input keys.
        """
        self.graph = graph
        self.capabilities = capabilities
        self.agent_id = agent_id or capabilities.name
        self.endpoints = tuple(endpoints or [])
        self.skill_mapping = skill_mapping or {}
        self._agent_card: AgentCard | None = None

    def get_agent_card(self) -> AgentCard:
        """
        Generate the A2A Agent Card for this agent.

        Returns:
            AgentCard describing this agent for discovery.
        """
        if self._agent_card is None:
            self._agent_card = AgentCard(
                agent_id=self.agent_id,
                capabilities=self.capabilities,
                endpoints=self.endpoints,
            )
        return self._agent_card

    def can_handle_skill(self, skill: str) -> bool:
        """
        Check if this agent can handle a specific skill.

        Args:
            skill: The skill name to check.

        Returns:
            True if this agent can handle the skill.
        """
        return self.capabilities.has_skill(skill)

    def handle_message(self, message: A2AMessage) -> A2AMessage:
        """
        Handle an incoming A2A protocol message.

        Args:
            message: The incoming A2A message.

        Returns:
            Response message.

        Raises:
            A2AProtocolError: If the message cannot be handled.
        """
        if message.message_type == A2AMessageType.AGENT_DISCOVER:
            # Return agent card
            return A2AMessage(
                message_type=A2AMessageType.AGENT_CARD,
                payload=self.get_agent_card().to_dict(),
                sender_id=self.agent_id,
                correlation_id=message.message_id,
            )

        elif message.message_type == A2AMessageType.CAPABILITY_QUERY:
            # Return capabilities
            queried_skill = message.payload.get("skill")
            has_skill = self.can_handle_skill(queried_skill) if queried_skill else True
            return A2AMessage(
                message_type=A2AMessageType.CAPABILITY_RESPONSE,
                payload={
                    "has_skill": has_skill,
                    "capabilities": self.capabilities.to_dict(),
                },
                sender_id=self.agent_id,
                correlation_id=message.message_id,
            )

        elif message.message_type == A2AMessageType.TASK_CREATE:
            # Handle task request
            request = A2ARequest.from_message(message)
            response = self.handle_request(request)
            return response.to_message(correlation_id=message.message_id)

        else:
            return A2AMessage(
                message_type=A2AMessageType.ERROR,
                payload={
                    "code": "UNSUPPORTED_MESSAGE_TYPE",
                    "message": f"Unsupported message type: {message.message_type}",
                },
                sender_id=self.agent_id,
                correlation_id=message.message_id,
            )

    def handle_request(self, request: A2ARequest) -> A2AResponse:
        """
        Handle an A2A task request synchronously.

        Args:
            request: The incoming task request.

        Returns:
            A2AResponse with task result or error.
        """
        # Validate skill
        if not self.can_handle_skill(request.skill):
            return A2AResponse.create_error(
                task_id=request.task_id,
                error_message=f"Skill not available: {request.skill}",
                error_code="SKILL_NOT_FOUND",
                responder_agent_id=self.agent_id,
            )

        try:
            # Prepare input state
            input_state = self._prepare_input(request)

            # Invoke the graph
            result = self.graph.invoke(input_state)

            # Return success response
            return A2AResponse.success(
                task_id=request.task_id,
                result=self._prepare_output(result),
                responder_agent_id=self.agent_id,
            )

        except Exception as e:
            return A2AResponse.create_error(
                task_id=request.task_id,
                error_message=str(e),
                error_code="EXECUTION_ERROR",
                responder_agent_id=self.agent_id,
            )

    async def ahandle_request(self, request: A2ARequest) -> A2AResponse:
        """
        Handle an A2A task request asynchronously.

        Args:
            request: The incoming task request.

        Returns:
            A2AResponse with task result or error.
        """
        # Validate skill
        if not self.can_handle_skill(request.skill):
            return A2AResponse.create_error(
                task_id=request.task_id,
                error_message=f"Skill not available: {request.skill}",
                error_code="SKILL_NOT_FOUND",
                responder_agent_id=self.agent_id,
            )

        try:
            # Prepare input state
            input_state = self._prepare_input(request)

            # Invoke the graph asynchronously
            result = await self.graph.ainvoke(input_state)

            # Return success response
            return A2AResponse.success(
                task_id=request.task_id,
                result=self._prepare_output(result),
                responder_agent_id=self.agent_id,
            )

        except Exception as e:
            return A2AResponse.create_error(
                task_id=request.task_id,
                error_message=str(e),
                error_code="EXECUTION_ERROR",
                responder_agent_id=self.agent_id,
            )

    def _prepare_input(self, request: A2ARequest) -> dict[str, Any]:
        """
        Prepare input state from an A2A request.

        Args:
            request: The incoming request.

        Returns:
            Input state dictionary for the graph.
        """
        input_state = dict(request.input_data)

        # Add context if provided
        if request.context:
            input_state["__a2a_context__"] = request.context

        # Add task metadata
        input_state["__a2a_task_id__"] = request.task_id
        input_state["__a2a_skill__"] = request.skill

        return input_state

    def _prepare_output(self, result: Any) -> dict[str, Any]:
        """
        Prepare output from graph result.

        Args:
            result: The graph execution result.

        Returns:
            Output dictionary for the A2A response.
        """
        if isinstance(result, dict):
            # Remove internal A2A keys
            output = {k: v for k, v in result.items() if not k.startswith("__a2a_")}
            return output
        else:
            return {"result": result}

    def stream_request(
        self,
        request: A2ARequest,
        on_update: Callable[[A2AResponse], None] | None = None,
    ) -> A2AResponse:
        """
        Handle a request with streaming updates.

        Args:
            request: The incoming task request.
            on_update: Optional callback for intermediate updates.

        Returns:
            Final A2AResponse with task result.
        """
        if not self.can_handle_skill(request.skill):
            return A2AResponse.create_error(
                task_id=request.task_id,
                error_message=f"Skill not available: {request.skill}",
                error_code="SKILL_NOT_FOUND",
                responder_agent_id=self.agent_id,
            )

        try:
            input_state = self._prepare_input(request)

            # Send initial running status
            if on_update:
                on_update(
                    A2AResponse(
                        task_id=request.task_id,
                        status=A2ATaskStatus.RUNNING,
                        progress=0.0,
                        responder_agent_id=self.agent_id,
                    )
                )

            # Stream through the graph
            final_result = None
            step_count = 0
            for chunk in self.graph.stream(input_state, stream_mode="updates"):
                step_count += 1
                final_result = chunk
                if on_update:
                    on_update(
                        A2AResponse(
                            task_id=request.task_id,
                            status=A2ATaskStatus.RUNNING,
                            progress=min(90.0, step_count * 10.0),
                            result={"step": step_count, "update": chunk},
                            responder_agent_id=self.agent_id,
                        )
                    )

            return A2AResponse.success(
                task_id=request.task_id,
                result=self._prepare_output(final_result) if final_result else {},
                responder_agent_id=self.agent_id,
            )

        except Exception as e:
            return A2AResponse.create_error(
                task_id=request.task_id,
                error_message=str(e),
                error_code="EXECUTION_ERROR",
                responder_agent_id=self.agent_id,
            )
