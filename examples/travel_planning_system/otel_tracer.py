"""Custom OpenTelemetry tracer for LangChain that exports to Jaeger with GenAI semantic conventions."""
# flake8: noqa

import os
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID
from datetime import datetime
from urllib.parse import urlparse

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from opentelemetry import trace, context
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Import semantic conventions
from langchain_azure_ai.callbacks.tracers._semantic_conventions_gen_ai import (
    GEN_AI_SYSTEM,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_REQUEST_MAX_OUTPUT_TOKENS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_EVENT_CONTENT,
    GEN_AI_TOOL_INPUT,
    GEN_AI_TOOL_OUTPUT,
    INPUTS,
    OUTPUTS,
    TAGS,
    ERROR_TYPE,
    SERVER_ADDRESS,
)

logger = logging.getLogger(__name__)


def _serialize_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Serialize messages to a format suitable for tracing."""
    serialized = []
    for msg in messages:
        msg_dict = {
            "role": msg.type,
            "content": msg.content,
        }
        
        # Add tool calls if present
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
            
        # Add tool call ID if present
        if isinstance(msg, ToolMessage):
            msg_dict["tool_call_id"] = msg.tool_call_id
            msg_dict["name"] = msg.name
            
        serialized.append(msg_dict)
    return serialized


class OpenTelemetryTracer(BaseTracer):
    """LangChain tracer that exports spans to OpenTelemetry collectors like Jaeger using GenAI semantic conventions."""
    
    name: str = "opentelemetry_tracer"
    
    def __init__(
        self,
        *,
        service_name: str = "langchain-app",
        otlp_endpoint: Optional[str] = None,
        insecure: bool = True,
        enable_content_recording: bool = True,
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenTelemetry tracer."""
        super().__init__(**kwargs)
        
        self.enable_content_recording = enable_content_recording
        self.debug = debug
        
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        
        # Get endpoint from env or parameter
        endpoint = otlp_endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", 
            "http://localhost:4318/v1/traces"
        )
        
        # Create resource with GenAI system attribute
        resource = Resource.create({
            SERVICE_NAME: service_name,
            SERVICE_VERSION: "1.0.0",
            GEN_AI_SYSTEM: "langchain",
            "deployment.environment": os.environ.get("ENVIRONMENT", "development"),
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
        })
        
        # Set up tracer provider
        provider = TracerProvider(resource=resource)
        
        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={"Content-Type": "application/json"}
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)
        
        # Set as global tracer provider
        trace.set_tracer_provider(provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(
            instrumenting_module_name=__name__,
            tracer_provider=provider
        )
        self.spans: Dict[str, Any] = {}
        self.root_span: Optional[Any] = None
        self.root_run_id: Optional[str] = None
        
        print(f"✅ OpenTelemetry tracer initialized - sending to: {endpoint}")
        print(f"   Content recording: {'enabled' if enable_content_recording else 'disabled'}")
        
        # Add external root span tracking
        self.external_root_span = None
        
    def set_root_span(self, span: Any, session_id: str):
        """Set an externally created root span."""
        self.external_root_span = span
        self.root_span = span
        self.root_run_id = session_id
        # Store it with a special key so child spans can find it
        self.spans["root"] = span
    
    def _get_parent_context(self, parent_run_id: Optional[UUID]):
        """Get the proper parent context for a span."""
        parent_context = None
        
        # First check for external root span
        if self.external_root_span and not parent_run_id:
            # This is a top-level operation, use external root as parent
            parent_context = trace.set_span_in_context(self.external_root_span)
        elif parent_run_id:
            # Look for parent span
            parent_span = self.spans.get(str(parent_run_id))
            if parent_span:
                parent_context = trace.set_span_in_context(parent_span)
            elif self.external_root_span:
                # Fall back to external root if parent not found
                parent_context = trace.set_span_in_context(self.external_root_span)
        elif self.root_span:
            # Use internal root span
            parent_context = trace.set_span_in_context(self.root_span)
        
        return parent_context
    
    def _serialize_json(self, obj: Any) -> str:
        """Safely serialize objects to JSON."""
        try:
            return json.dumps(obj, default=str)
        except (TypeError, ValueError):
            return str(obj)
    
    def _on_run_create(self, run: Run) -> None:
        """Process a run when it's created."""
        # This is called by BaseTracer to index runs
        pass
    
    def _on_run_update(self, run: Run) -> None:
        """Process a run when it's updated."""
        # This is called by BaseTracer to update indexed runs
        pass
    
    def _on_llm_start(self, run: Run) -> None:
        """Handle LLM start for base tracer compatibility."""
        # We handle this in on_chat_model_start
        pass
    
    def _on_llm_end(self, run: Run) -> None:
        """Handle LLM end for base tracer compatibility."""
        # We handle this in on_llm_end
        pass
    
    def _on_llm_error(self, run: Run) -> None:
        """Handle LLM error for base tracer compatibility."""
        # We handle this in on_llm_error
        pass
    
    def _on_chain_start(self, run: Run) -> None:
        """Handle chain start for base tracer compatibility."""
        # We handle this in on_chain_start
        pass
    
    def _on_chain_end(self, run: Run) -> None:
        """Handle chain end for base tracer compatibility."""
        # We handle this in on_chain_end
        pass
    
    def _on_chain_error(self, run: Run) -> None:
        """Handle chain error for base tracer compatibility."""
        # We handle this in on_chain_error
        pass
    
    def _on_tool_start(self, run: Run) -> None:
        """Handle tool start for base tracer compatibility."""
        # We handle this in on_tool_start
        pass
    
    def _on_tool_end(self, run: Run) -> None:
        """Handle tool end for base tracer compatibility."""
        # We handle this in on_tool_end
        pass
    
    def _on_tool_error(self, run: Run) -> None:
        """Handle tool error for base tracer compatibility."""
        # We handle this in on_tool_error
        pass
        
    def _persist_run(self, run: Run) -> None:
        """Persist a run to OpenTelemetry."""
        # Nothing to persist as spans are sent automatically
        pass
    
    def _log_span_info(self, span: Any, span_type: str):
        """Log span information for debugging and evaluation."""
        if span and self.debug:
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            
            print(f"\n{'='*60}")
            print(f"📍 {span_type} Span Created")
            print(f"{'='*60}")
            print(f"Span Name: {span.name}")
            print(f"Trace ID: {trace_id}")
            print(f"Span ID: {span_id}")
            print(f"\n💡 To evaluate this span, add to your .env:")
            print(f"LANGGRAPH_SPAN_ID={span_id}")
            if span_type == "Root Session":
                print(f"LANGGRAPH_ROOT_SPAN_ID={span_id}")
            print(f"{'='*60}\n")
    
    def _serialize_json(self, obj: Any, max_length: Optional[int] = None) -> str:
        """Serialize object to JSON string."""
        try:
            json_str = json.dumps(obj, indent=2, default=str)
            if max_length and len(json_str) > max_length:
                # Don't truncate, compress instead
                json_str = json.dumps(obj, separators=(',', ':'), default=str)
            return json_str
        except Exception:
            return str(obj)
    
    def _add_event_to_span(self, span: Any, event_name: str, content: Dict[str, Any]):
        """Add an event to a span following GenAI conventions."""
        if not span or not self.enable_content_recording:
            return
            
        # Create attributes for the event
        attributes = {
            GEN_AI_EVENT_CONTENT: self._serialize_json(content)
        }
        
        # Add the event with attributes
        span.add_event(event_name, attributes=attributes)
        
        if self.debug:
            logger.debug(f"Added event '{event_name}' to span with content: {content}")
    
    def _format_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Format messages for tracing."""
        formatted = []
        for msg in messages:
            if isinstance(msg, list):
                # Handle nested message lists
                formatted.extend(self._format_messages(msg))
            elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
                msg_dict = {
                    "role": msg.__class__.__name__.replace("Message", "").lower(),
                    "content": msg.content if self.enable_content_recording else "[REDACTED]",
                }
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                if isinstance(msg, ToolMessage):
                    msg_dict["tool_call_id"] = msg.tool_call_id
                    msg_dict["name"] = msg.name
                formatted.append(msg_dict)
        return formatted
    
    def _extract_model_info(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model information from serialized data."""
        kwargs = serialized.get("kwargs", {})
        
        # Get model name from various sources
        model_name = kwargs.get("deployment_name", "")
        if not model_name:
            model_name = kwargs.get("model_name", "")
        if not model_name:
            model_name = kwargs.get("azure_deployment", "")
        if not model_name and "id" in serialized:
            model_name = serialized["id"][0] if isinstance(serialized["id"], list) else serialized["id"]
            
        return {
            "deployment_name": model_name,
            "model_name": model_name,
            "temperature": kwargs.get("temperature", 0.0),
            "azure_endpoint": kwargs.get("azure_endpoint", ""),
            "api_version": kwargs.get("openai_api_version", ""),
        }
    
    def _add_agent_invocation_event(self, span: Any, input_content: Any, output_content: Any = None):
        """Add agent invocation event to span."""
        if not span or not self.enable_content_recording:
            return
            
        event_attrs = {}
        
        # Add input
        if input_content:
            event_attrs["gen_ai.agent.invocation.input"] = self._serialize_json(input_content)
        
        # Add output if available
        if output_content:
            event_attrs["gen_ai.agent.invocation.output"] = self._serialize_json(output_content)
        
        # Add the event
        span.add_event(
            name="gen_ai.agent.invocation",
            attributes=event_attrs
        )
    
    def _handle_tool_calls(self, message: AIMessage, parent_span: Any) -> None:
        """Create execute_tool spans for tool calls in the message."""
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return

        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.get("name", "unknown_tool")
                tool_id = tool_call.get("id", "")
                tool_args = tool_call.get("args", {})

                # Create execute_tool span
                with self.tracer.start_as_current_span(
                    name=f"execute_tool {tool_name}",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        GEN_AI_OPERATION_NAME: "execute_tool",
                        GEN_AI_TOOL_NAME: tool_name,
                        "gen_ai.tool.invocation.id": tool_id,  # Fixed attribute name
                    }
                ) as tool_span:
                    # Add tool arguments if content recording is enabled
                    if self.enable_content_recording:
                        tool_span.set_attribute(
                            "gen_ai.tool.invocation.arguments",  # Fixed attribute name
                            self._serialize_json(tool_args)
                        )
                    
                    # Add event for tool call
                    tool_span.add_event(
                        name="gen_ai.tool.call",
                        attributes={
                            "tool_name": tool_name,
                            "tool_id": tool_id,
                        }
                    )

            except Exception as e:
                logger.error(f"Error creating tool call span: {e}")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Run when Chat Model starts running."""
        run = super().on_chat_model_start(
            serialized, messages, run_id=run_id, tags=tags,
            parent_run_id=parent_run_id, metadata=metadata, name=name, **kwargs
        )
        
        # Check if we should create a root span for the entire execution
        if not self.root_span and tags and "travel-planning-execution" in tags:
            # Create root span for the entire travel planning session
            session_id = metadata.get("session_id", str(UUID(int=0))) if metadata else str(UUID(int=0))
            root_span = self.tracer.start_span("travel_planning_session", kind=SpanKind.SERVER)
            root_span.set_attribute("span_type", "Session")
            root_span.set_attribute(GEN_AI_SYSTEM, "langchain")
            root_span.set_attribute(GEN_AI_OPERATION_NAME, "travel_planning")
            root_span.set_attribute("session.id", session_id)
            if metadata:
                if metadata.get("user_request"):
                    root_span.set_attribute("user.request", metadata["user_request"])
                if metadata.get("timestamp"):
                    root_span.set_attribute("session.start_time", metadata["timestamp"])
            self.root_span = root_span
            self.root_run_id = "00000000-0000-0000-0000-000000000000"  # Special ID for root
            self.spans[self.root_run_id] = root_span
            
            # Log root span info
            self._log_span_info(root_span, "Root Session")
        
        # Extract model info
        model_info = self._extract_model_info(serialized)
        
        # Create span name
        span_name = f"chat.completions {model_info['deployment_name']}"
        
        # Prepare attributes with ALL required and recommended fields
        attributes = {
            # REQUIRED
            "gen_ai.operation.name": "chat.completions",
            "gen_ai.system": "azure_openai",
            "gen_ai.request.model": model_info["deployment_name"],
            
            # CONDITIONALLY REQUIRED (extract from serialized/metadata)
            "gen_ai.request.temperature": model_info.get("temperature", 0.7),
            "gen_ai.request.top_p": serialized.get("kwargs", {}).get("top_p", 1.0),
            "gen_ai.request.max_tokens": serialized.get("kwargs", {}).get("max_tokens", 4096),
            "gen_ai.request.frequency_penalty": serialized.get("kwargs", {}).get("frequency_penalty", 0.0),
            "gen_ai.request.presence_penalty": serialized.get("kwargs", {}).get("presence_penalty", 0.0),
            
            # RECOMMENDED
            "gen_ai.provider.name": "azure_openai",
            "gen_ai.request.api_version": model_info.get("api_version", "2024-02-15-preview"),
            "gen_ai.request.endpoint": model_info.get("azure_endpoint", ""),
            "server.address": urlparse(model_info.get("azure_endpoint", "")).netloc or "",
            "server.port": 443,  # HTTPS default
            
            # Additional context
            "run_id": str(run_id),
            "langchain.run_type": run.run_type,
            "span.kind": "client",  # LLM calls are CLIENT spans
        }
        
        # Add stop sequences if present
        stop_sequences = serialized.get("kwargs", {}).get("stop", [])
        if stop_sequences:
            attributes["gen_ai.request.stop_sequences"] = self._serialize_json(stop_sequences)
        
        # Add parent_run_id only if it's not None
        if parent_run_id is not None:
            attributes["parent_run_id"] = str(parent_run_id)
            
        # Add tags only if they exist
        if tags:
            attributes["tags"] = self._serialize_json(tags)
        
        # Get parent context using new method
        parent_context = self._get_parent_context(parent_run_id)
        
        # Start span
        span = self.tracer.start_span(
            name=span_name,
            context=parent_context,
            attributes=attributes,
            kind=SpanKind.CLIENT
        )
        
        # Store messages as events if content recording is enabled
        if self.enable_content_recording:
            formatted_messages = self._format_messages(messages)
            for i, msg in enumerate(formatted_messages):
                span.add_event(
                    name="gen_ai.content.prompt",
                    attributes={
                        "gen_ai.prompt": self._serialize_json(msg),
                        "message_index": i,
                    },
                )
                
        # Add metadata as span attributes
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"metadata.{key}", value)
                else:
                    span.set_attribute(f"metadata.{key}", self._serialize_json(value))
        
        # Store span
        self.spans[str(run_id)] = span
        
        # Log generation span info
        self._log_span_info(span, "Generation")
        
        return run
    
    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> Run:
        """Run when LLM ends running."""
        run = super().on_llm_end(response, run_id=run_id, **kwargs)
        
        span_key = str(run_id)
        
        if span_key not in self.spans:
            logger.warning(f"No active span found for run_id: {run_id}")
            return run
            
        span = self.spans[span_key]
        
        try:
            # Extract token usage
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})
            
            # Set token usage attributes
            if token_usage:
                span.set_attribute(
                    GEN_AI_USAGE_INPUT_TOKENS,
                    token_usage.get("prompt_tokens", 0),
                )
                span.set_attribute(
                    GEN_AI_USAGE_OUTPUT_TOKENS,
                    token_usage.get("completion_tokens", 0),
                )
                span.set_attribute(
                    GEN_AI_USAGE_TOTAL_TOKENS,
                    token_usage.get("total_tokens", 0)
                )
                
            # Process generations
            if self.enable_content_recording:
                for generation_list in response.generations:
                    for generation in generation_list:
                        if hasattr(generation, "message") and generation.message:
                            message = generation.message
                            
                            # Add completion event
                            completion_attrs = {
                                GEN_AI_EVENT_CONTENT: self._serialize_json(
                                    {
                                        "role": message.__class__.__name__.replace(
                                            "Message", ""
                                        ).lower(),
                                        "content": message.content,
                                    }
                                )
                            }
                            
                            # Add tool calls if present
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                completion_attrs["tool_calls"] = self._serialize_json(
                                    message.tool_calls
                                )
                                # Create execute_tool spans for each tool call
                                self._handle_tool_calls(message, span)
                                
                            span.add_event(
                                name="gen_ai.content.completion",
                                attributes=completion_attrs,
                            )
                            
                            # Set response attributes with proper names
                            if hasattr(message, "response_metadata"):
                                resp_meta = message.response_metadata
                                # Use correct attribute names
                                span.set_attribute("gen_ai.response.model", resp_meta.get("model", "gpt-4.1-2025-04-14"))
                                span.set_attribute("gen_ai.response.id", resp_meta.get("system_fingerprint", ""))
                                span.set_attribute("gen_ai.response.finish_reason", resp_meta.get("finish_reason", "stop"))
                            else:
                                # Set defaults if metadata not available
                                span.set_attribute("gen_ai.response.model", "gpt-4.1-2025-04-14")
                                span.set_attribute("gen_ai.response.finish_reason", "stop")
                                
            # Set status to OK
            span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
        finally:
            # End span and remove from active spans
            span.end()
            del self.spans[span_key]
            
        return run
    
    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> Run:
        """Run when LLM errors."""
        run = super().on_llm_error(error, run_id=run_id, **kwargs)
        
        span_key = str(run_id)
        
        if span_key not in self.spans:
            logger.warning(f"No active span found for run_id: {run_id}")
            return run
            
        span = self.spans[span_key]
        
        try:
            # Record exception
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            
            # Add error details
            span.set_attribute(ERROR_TYPE, type(error).__name__)
            span.set_attribute("error.message", str(error))
            
        finally:
            # End span and remove from active spans
            span.end()
            del self.spans[span_key]
            
        return run
    
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_type: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Run when chain starts running."""
        run = super().on_chain_start(
            serialized, inputs, run_id=run_id, tags=tags,
            parent_run_id=parent_run_id, metadata=metadata,
            run_type=run_type, name=name, **kwargs
        )
    
        # Check if we should create a root span for the entire execution
        if not self.root_span and tags and "travel-planning-execution" in tags and not parent_run_id:
            # This is the top-level chain execution, it should use the externally created root span
            # The root span should already be set by the main function
            pass
        
        # Create span name
        chain_name = kwargs.get("name", "LangGraph")
        
        # Fix span naming for agent invocations
        if metadata and metadata.get("agent_type"):
            agent_type = metadata.get("agent_type")
            agent_name = metadata.get("gen_ai.agent.name", agent_type)
            
            # Use proper naming: "invoke_agent {agent_name}"
            span_name = f"invoke_agent {agent_name}"
            operation_name = "invoke_agent"
            
            # Move metadata attributes to top level for compliance
            attributes = {
                # REQUIRED
                "gen_ai.operation.name": operation_name,
                "gen_ai.system": "langchain",
                "gen_ai.agent.name": agent_name,
                "gen_ai.agent.id": metadata.get("gen_ai.agent.id", f"{agent_name}_{run_id}"),
                
                # CONDITIONALLY REQUIRED
                "gen_ai.request.model": metadata.get("gen_ai.request.model", "gpt-4.1"),
                "gen_ai.request.temperature": metadata.get("gen_ai.request.temperature", 0.7),
                "gen_ai.conversation.id": metadata.get("gen_ai.conversation.id", str(run_id)),
                
                # RECOMMENDED
                "gen_ai.agent.mode": "assistant",  # Child agents are assistants
                "gen_ai.provider.name": metadata.get("gen_ai.provider.name", "azure_openai"),
                "server.address": metadata.get("server.address", ""),
                
                # Additional
                "chain.name": chain_name,
                "run_id": str(run_id),
                "langchain.run_type": run.run_type,
                "span.kind": "internal",
            }
            
            # Only add the task as a separate attribute
            if metadata.get("task"):
                attributes["gen_ai.agent.task"] = metadata["task"]
                
        else:
            # Non-agent chain - keep existing logic but fix attribute placement
            span_name = f"chain.{chain_name}"
            operation_name = f"chain.{chain_name}"
            attributes = {
                "gen_ai.operation.name": operation_name,
                "chain.name": chain_name,
                "run_id": str(run_id),
                "langchain.run_type": run.run_type,
                "span.kind": "internal",
            }
        
        # Add parent_run_id only if it's not None
        if parent_run_id is not None:
            attributes["parent_run_id"] = str(parent_run_id)
            
        # Add tags only if they exist
        if tags:
            attributes["tags"] = self._serialize_json(tags)
            
        # Get parent context using the new method
        parent_context = self._get_parent_context(parent_run_id)
            
        # Start span
        span = self.tracer.start_span(
            name=span_name,
            context=parent_context,
            attributes=attributes,
            kind=SpanKind.INTERNAL
        )
        
        # Add inputs
        if inputs:
            attributes["inputs"] = list(inputs.keys())
                    
        # Log inputs (be careful with sensitive data)
        if self.enable_content_recording and "messages" in inputs:
            formatted_messages = self._format_messages(inputs["messages"])
            span.set_attribute("chain.input.message_count", len(formatted_messages))
            span.set_attribute(
                "chain.input.messages", self._serialize_json(formatted_messages)
            )
            
        # Store span
        self.spans[str(run_id)] = span
        
        if self.debug:
            logger.debug(f"Created {span_name} span (run_id: {run_id})")
            
        return run
    
    def on_chain_end(
        self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any
    ) -> Run:
        """Run when chain ends running."""
        run = super().on_chain_end(outputs, run_id=run_id, **kwargs)
        
        span_key = str(run_id)
        
        if span_key not in self.spans:
            return run
            
        span = self.spans[span_key]
        
        try:
            # Handle string outputs from supervisor chain
            if isinstance(outputs, str):
                outputs = {"output": outputs}
            elif not isinstance(outputs, dict):
                outputs = {"output": str(outputs)}
            
            span.set_attribute(OUTPUTS, list(outputs.keys()))
            
            if self.enable_content_recording:
                span.set_attribute("chain.outputs", self._serialize_json(outputs))
                
            # Check if this is a planning span
            if span.name == "gen_ai.agent.planning":
                # Extract plan details from outputs
                plan_details = {
                    "strategy": "graph_based",
                    "outputs": outputs
                }
                
                # Add planning event
                span.add_event(
                    name="gen_ai.agent.plan",
                    attributes={
                        "gen_ai.agent.plan.description": self._serialize_json(plan_details)
                    }
                )
                
                # Set planning output
                if "messages" in outputs and outputs["messages"]:
                    last_message = outputs["messages"][-1]
                    if hasattr(last_message, "content"):
                        span.set_attribute("gen_ai.agent.planning.output", last_message.content)
                elif "output" in outputs:
                    # Handle string output from supervisor
                    span.set_attribute("gen_ai.agent.planning.output", str(outputs["output"]))
            
            # Check if this is the final travel plan output
            if "messages" in outputs and outputs["messages"]:
                last_message = outputs["messages"][-1]
                if isinstance(last_message, AIMessage):
                    # Check if this contains a final travel plan
                    if "FINAL TRAVEL PLAN" in last_message.content or "comprehensive plan" in last_message.content.lower():
                        span.set_attribute("travel.plan.final", last_message.content)
            elif "output" in outputs:
                # Handle string output that might contain final plan
                output_str = str(outputs["output"])
                if "FINAL TRAVEL PLAN" in output_str or "comprehensive plan" in output_str.lower():
                    span.set_attribute("travel.plan.final", output_str)
            
            # Check if this is the final chain end (root level) - do this BEFORE ending the current span
            if hasattr(run, 'tags') and run.tags and "travel-planning-execution" in run.tags and not run.parent_run_id:
                # Add the final output to root span if available
                if self.root_span:
                    if "messages" in outputs and outputs["messages"]:
                        last_message = outputs["messages"][-1]
                        if isinstance(last_message, AIMessage):
                            # Check for final travel plan
                            if "FINAL TRAVEL PLAN" in last_message.content or "comprehensive plan" in last_message.content.lower():
                                try:
                                    # Check if root span is still active
                                    if self.root_run_id in self.spans:
                                        self.root_span.set_attribute("travel.plan.final", last_message.content)
                                        # Add final output event
                                        self._add_event_to_span(self.root_span, "travel.plan.final", {
                                            "role": "assistant",
                                            "content": last_message.content
                                        })
                                except Exception as e:
                                    logger.warning(f"Could not update root span: {e}")
                    elif "output" in outputs:
                        # Handle string output
                        output_str = str(outputs["output"])
                        if "FINAL TRAVEL PLAN" in output_str or "comprehensive plan" in output_str.lower():
                            try:
                                if self.root_run_id in self.spans:
                                    self.root_span.set_attribute("travel.plan.final", output_str)
                                    # Add final output event
                                    self._add_event_to_span(self.root_span, "travel.plan.final", {
                                        "role": "assistant",
                                        "content": output_str
                                    })
                            except Exception as e:
                                logger.warning(f"Could not update root span: {e}")
            
            span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            logger.error(f"Error in on_chain_end: {e}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            # End the current span
            span.end()
            del self.spans[span_key]
            
            # Now handle root span ending if this was the final chain
            if hasattr(run, 'tags') and run.tags and "travel-planning-execution" in run.tags and not run.parent_run_id:
                # End the root span
                if self.root_run_id and self.root_run_id in self.spans:
                    root_span = self.spans.pop(self.root_run_id, None)
                    if root_span:
                        root_span.set_status(Status(StatusCode.OK))
                        root_span.end()
                        self.root_span = None
                        self.root_run_id = None
                        
        return run

    def on_chain_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> Run:
        """Run when chain errors."""
        run = super().on_chain_error(error, run_id=run_id, **kwargs)
        
        span_key = str(run_id)
        
        if span_key not in self.spans:
            return run
            
        span = self.spans[span_key]
        
        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.set_attribute(ERROR_TYPE, type(error).__name__)
            span.set_attribute("error.message", str(error))
        finally:
            span.end()
            del self.spans[span_key]
            
        # If this is the root chain error, end the root span with error
        if hasattr(run, 'tags') and run.tags and "travel-planning-execution" in run.tags and not run.parent_run_id:
            if self.root_span:
                self.root_span.set_attribute(ERROR_TYPE, type(error).__name__)
                self.root_span.set_attribute("error.message", str(error))
                self.root_span.record_exception(error)
            if self.root_run_id:
                root_span = self.spans.pop(self.root_run_id, None)
                if root_span:
                    root_span.set_status(Status(StatusCode.ERROR, str(error)))
                    root_span.end()
                    self.root_span = None
                    self.root_run_id = None
                    
        return run
    
    def on_tool_start(
    self,
    serialized: dict[str, Any],
    input_str: str,
    *,
    run_id: UUID,
    tags: Optional[list[str]] = None,
    parent_run_id: Optional[UUID] = None,
    metadata: Optional[dict[str, Any]] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Run:
        """Run when tool starts running."""
        run = super().on_tool_start(
            serialized, input_str, run_id=run_id, tags=tags,
            parent_run_id=parent_run_id, metadata=metadata, name=name, **kwargs
        )
        
        # Extract tool name and description
        tool_name = kwargs.get("name", serialized.get("name", "unknown_tool"))
        tool_description = serialized.get("description", "")
        
        # Parse input if it's JSON
        tool_args = input_str
        try:
            tool_args = json.loads(input_str)
        except:
            pass  # Keep as string if not JSON
            
        # Use proper naming format: "tool {tool_name}"
        span_name = f"tool {tool_name}"
        
        attributes = {
            # REQUIRED
            "gen_ai.operation.name": "tool",
            "gen_ai.system": "langchain",
            "gen_ai.tool.name": tool_name,
            
            # CONDITIONALLY REQUIRED
            "gen_ai.tool.id": metadata.get("tool_call_id", str(run_id)) if metadata else str(run_id),
            
            # Tool arguments as top-level attribute
            "gen_ai.tool.arguments": self._serialize_json(tool_args) if isinstance(tool_args, dict) else tool_args,
            
            # Additional
            "run_id": str(run_id),
            "langchain.run_type": run.run_type,
            "span.kind": "internal",
        }
        
        # Add tool description if available
        if tool_description:
            attributes["gen_ai.tool.description"] = tool_description
        
        # Add parent_run_id only if it's not None
        if parent_run_id is not None:
            attributes["parent_run_id"] = str(parent_run_id)
            
        # Add tags only if they exist
        if tags:
            attributes["tags"] = self._serialize_json(tags)
            
        # Get parent context using new method
        parent_context = self._get_parent_context(parent_run_id)
            
        # Create span following proper naming convention
        span = self.tracer.start_span(
            name=span_name,
            context=parent_context,
            kind=SpanKind.INTERNAL,
            attributes=attributes,
        )
        
        # Record tool arguments if content recording is enabled
        if self.enable_content_recording:
            span.set_attribute(
                "gen_ai.tool.invocation.arguments",  # Fixed attribute name
                self._serialize_json(tool_args) if isinstance(tool_args, dict) else tool_args
            )
            
        # Add event for tool call
        span.add_event(
            name="gen_ai.tool.call",
            attributes={
                "tool_name": tool_name,
                "tool_id": metadata.get("tool_call_id", "") if metadata else "",
            }
        )
            
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"metadata.{key}", value)
                else:
                    span.set_attribute(f"metadata.{key}", self._serialize_json(value))
                    
        self.spans[str(run_id)] = span
        
        if self.debug:
            logger.debug(f"Created tool span: execute_tool {tool_name} (run_id: {run_id})")
            
        return run

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> Run:
        """Run when tool ends running."""
        run = super().on_tool_end(output, run_id=run_id, **kwargs)
        
        span_key = str(run_id)
        
        if span_key not in self.spans:
            return run
            
        span = self.spans[span_key]
        
        try:
            # Parse output if it's JSON
            tool_result = output
            try:
                tool_result = json.loads(output)
            except:
                pass  # Keep as string if not JSON
                
            # Record output if content recording is enabled
            if self.enable_content_recording:
                span.set_attribute(
                    "gen_ai.tool.invocation.result",  # Fixed attribute name
                    self._serialize_json(tool_result) if isinstance(tool_result, (dict, list)) else tool_result
                )
                
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            logger.error(f"Error in on_tool_end: {e}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            span.end()
            del self.spans[span_key]
            
        return run    

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> Run:
        """Run when tool errors."""
        run = super().on_tool_error(error, run_id=run_id, **kwargs)
        
        span_key = str(run_id)
        
        if span_key not in self.spans:
            return run
            
        span = self.spans[span_key]
        
        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.set_attribute(ERROR_TYPE, type(error).__name__)
            span.set_attribute("error.message", str(error))
        finally:
            span.end()
            del self.spans[span_key]
            
        return run
    
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Run when LLM starts running (for non-chat models)."""
        # Call parent's on_llm_start to create the run
        run = super().on_llm_start(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            name=name,
            **kwargs
        )
        
        # Convert to chat model format and handle with on_chat_model_start logic
        messages = []
        for prompt in prompts:
            messages.append([HumanMessage(content=prompt)])
        
        # Extract model info
        model_info = self._extract_model_info(serialized)
        
        # Create span name
        span_name = f"chat.completions {model_info['deployment_name']}"
        
        # Prepare attributes
        attributes = {
            # OpenTelemetry Semantic Conventions for GenAI
            GEN_AI_SYSTEM: "azure_openai",
            GEN_AI_REQUEST_MODEL: model_info["deployment_name"],
            GEN_AI_REQUEST_TEMPERATURE: model_info["temperature"],
            
            # Operation name
            GEN_AI_OPERATION_NAME: "chat.completions",
            
            # Azure specific attributes
            "gen_ai.request.api_version": model_info["api_version"],
            "gen_ai.request.endpoint": model_info["azure_endpoint"],
            SERVER_ADDRESS: (
                urlparse(model_info["azure_endpoint"]).netloc
                if model_info["azure_endpoint"]
                else ""
            ),
            
            # Run information
            "run_id": str(run_id),
            "langchain.run_type": run.run_type,
        }
        
        # Add parent_run_id only if it's not None
        if parent_run_id is not None:
            attributes["parent_run_id"] = str(parent_run_id)
            
        # Add tags only if they exist
        if tags:
            attributes[TAGS] = self._serialize_json(tags)
        
        # Get parent context
        parent_context = None
        if parent_run_id:
            parent_span = self.spans.get(str(parent_run_id))
            if parent_span:
                parent_context = trace.set_span_in_context(parent_span)
        elif self.root_span:
            parent_context = trace.set_span_in_context(self.root_span)
        
        # Start span
        span = self.tracer.start_span(
            name=span_name,
            context=parent_context,
            attributes=attributes,
            kind=SpanKind.CLIENT
        )
        
        # Store messages as events if content recording is enabled
        if self.enable_content_recording:
            formatted_messages = self._format_messages(messages)
            for i, msg in enumerate(formatted_messages):
                span.add_event(
                    name="gen_ai.content.prompt",
                    attributes={
                        "gen_ai.prompt": self._serialize_json(msg),
                        "message_index": i,
                    },
                )
        
        # Store span
        self.spans[str(run_id)] = span
        
        if self.debug:
            logger.debug(f"Created LLM span via on_llm_start: {span_name} (run_id: {run_id})")
        
        return run