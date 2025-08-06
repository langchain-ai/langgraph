"""
Travel planning system using LangGraph with OpenTelemetry tracing.
Demonstrates state-based graph orchestration with comprehensive telemetry.
"""
# flake8: noqa
# Standard library imports
import os
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from uuid import uuid4

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure OpenTelemetry before other imports
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
os.environ["OTEL_SERVICE_NAME"] = "travel-planning-system"

# Import custom tracer first
from otel_tracer import OpenTelemetryTracer
from otel_debug import OpenTelemetryTracerDebug

try:
    from opentelemetry import trace
    from opentelemetry import context
    from opentelemetry.trace import SpanKind
except ImportError:
    trace = None
    context = None
    SpanKind = None

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage

# LangChain tracing imports
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.callbacks import CallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain_azure_ai.callbacks.tracers import AzureOpenAITracingCallback

# LLM imports
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

# Initialize tracers
application_insights_connection_string = os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING", "")

# Initialize Azure tracer if connection string exists
tracers = []
if application_insights_connection_string:
    azure_tracer = AzureOpenAITracingCallback(
        connection_string=application_insights_connection_string,
        enable_content_recording=True,
    )
    tracers.append(azure_tracer)
    

# Initialize OpenTelemetry tracer
otel_tracer = OpenTelemetryTracer(
    service_name="travel-planning-system",
    otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
)
tracers.append(otel_tracer)

# Initialize LangSmith tracer if API key exists
# if os.environ.get("LANGCHAIN_API_KEY"):
#     langchain_tracer = LangChainTracer()
#     tracers.append(langchain_tracer)

# Create callback manager with all tracers
callback_manager = CallbackManager(tracers)


# Define the state type
class TravelPlanningState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    destination: str
    origin: str
    budget: float
    travelers: int
    departure_date: str
    return_date: str
    flights: List[Dict]
    hotels: List[Dict]
    current_agent: str
    waiting_for_user: bool  # New field to track if we're waiting for user input
    session_id: str  # Session ID for tracing


# Create LLM with fallback options
def create_llm():
    # Try Azure OpenAI first
    if os.environ.get("AZURE_OPENAI_API_KEY") and AzureChatOpenAI:
        try:
            return AzureChatOpenAI(
                azure_deployment="gpt-4.1",
                api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
                azure_endpoint="https://ai-naarkalgaihub999971652049.openai.azure.com/",
                api_version="2024-02-15-preview",
                temperature=0.7,
                callbacks=callback_manager,  # Automatic tracing
                tags=["travel-planner"],  # Add tags for filtering
                metadata={"system": "travel-planning"}  # Add metadata
            )
        except Exception as e:
            print(f"⚠️  Could not initialize Azure OpenAI: {e}")
    raise Exception("No valid LLM configuration found. Please check your environment variables.")


# Tool functions with OpenTelemetry tracing
@tool
def search_flights(origin: str, destination: str, date: str) -> List[Dict]:
    """Search for available flights between origin and destination on a specific date.
    
    Args:
        origin: The departure city/airport
        destination: The arrival city/airport
        date: The date of travel (YYYY-MM-DD format)
        
    Returns:
        List of available flights with details
    """
    flights = [
        {"flight_id": f"FL{random.randint(100, 999)}", 
         "departure": f"{random.randint(6, 22):02d}:00", 
         "arrival": f"{random.randint(6, 22):02d}:00", 
         "price": random.randint(200, 800),
         "airline": random.choice(["Skyways", "AirConnect", "GlobalJet"])},
        {"flight_id": f"FL{random.randint(100, 999)}", 
         "departure": f"{random.randint(6, 22):02d}:00", 
         "arrival": f"{random.randint(6, 22):02d}:00", 
         "price": random.randint(200, 800),
         "airline": random.choice(["Skyways", "AirConnect", "GlobalJet"])},
        {"flight_id": f"FL{random.randint(100, 999)}", 
         "departure": f"{random.randint(6, 22):02d}:00", 
         "arrival": f"{random.randint(6, 22):02d}:00", 
         "price": random.randint(200, 800),
         "airline": random.choice(["Skyways", "AirConnect", "GlobalJet"])},
    ]
    return flights


@tool
def search_hotels(destination: str, check_in: str, check_out: str, travelers: int) -> List[Dict]:
    """Search for available hotels at the destination for the specified dates and number of travelers.
    
    Args:
        destination: The city/location to search for hotels
        check_in: Check-in date (YYYY-MM-DD format)
        check_out: Check-out date (YYYY-MM-DD format)
        travelers: Number of travelers needing accommodation
        
    Returns:
        List of available hotels with details
    """
    if "chelsea" in destination.lower() or "london" in destination.lower():
        hotels = [
            {"hotel_id": "HT101", 
             "name": "The Chelsea Harbor Hotel", 
             "price_per_night": 250,
             "rating": 4.5,
             "accommodates": travelers,
             "distance_to_stadium": "1.2 miles"},
            {"hotel_id": "HT102", 
             "name": "Stamford Bridge Lodge", 
             "price_per_night": 180,
             "rating": 4.2,
             "accommodates": travelers,
             "distance_to_stadium": "0.5 miles"},
            {"hotel_id": "HT103", 
             "name": "Premier Inn London Kensington", 
             "price_per_night": 120,
             "rating": 4.0,
             "accommodates": travelers,
             "distance_to_stadium": "2.0 miles"},
        ]
    else:
        hotels = [
            {"hotel_id": f"HT{random.randint(100, 999)}", 
             "name": random.choice(["Grand Hotel", "Seaside Resort", "City View Inn", "Mountain Retreat"]), 
             "price_per_night": random.randint(80, 400),
             "rating": round(random.uniform(3.0, 5.0), 1),
             "accommodates": travelers},
            {"hotel_id": f"HT{random.randint(100, 999)}", 
             "name": random.choice(["Grand Hotel", "Seaside Resort", "City View Inn", "Mountain Retreat"]), 
             "price_per_night": random.randint(80, 400),
             "rating": round(random.uniform(3.0, 5.0), 1),
             "accommodates": travelers},
            {"hotel_id": f"HT{random.randint(100, 999)}", 
             "name": random.choice(["Grand Hotel", "Seaside Resort", "City View Inn", "Mountain Retreat"]), 
             "price_per_night": random.randint(80, 400),
             "rating": round(random.uniform(3.0, 5.0), 1),
             "accommodates": travelers},
        ]
    return hotels


@tool
def get_match_schedule(team: str) -> List[Dict]:
    """Get the match schedule for a football team.
    
    Args:
        team: The name of the football team
        
    Returns:
        List of upcoming matches with dates, opponents, and venues
    """
    # Simulate Chelsea match schedule
    if "chelsea" in team.lower():
        matches = [
            {"date": "2024-08-18", "opponent": "Manchester City", "venue": "Stamford Bridge", "competition": "Premier League"},
            {"date": "2024-08-24", "opponent": "Wolves", "venue": "Molineux Stadium", "competition": "Premier League"},
            {"date": "2024-09-01", "opponent": "Crystal Palace", "venue": "Stamford Bridge", "competition": "Premier League"},
        ]
    else:
        matches = [
            {"date": "2024-08-17", "opponent": "Team A", "venue": "Home", "competition": "League"},
            {"date": "2024-08-24", "opponent": "Team B", "venue": "Away", "competition": "League"},
            {"date": "2024-08-31", "opponent": "Team C", "venue": "Home", "competition": "League"},
        ]
    return matches


@tool
def book_flight(flight_id: str) -> str:
    """Book a specific flight by its ID.
    
    Args:
        flight_id: The unique identifier of the flight to book
        
    Returns:
        Booking confirmation message with reference number
    """
    booking_reference = f"BF{random.randint(10000, 99999)}"
    return f"Flight {flight_id} successfully booked. Booking reference: {booking_reference}"


@tool
def book_hotel(hotel_id: str) -> str:
    """Book a specific hotel by its ID.
    
    Args:
        hotel_id: The unique identifier of the hotel to book
        
    Returns:
        Booking confirmation message with reference number
    """
    booking_reference = f"BH{random.randint(10000, 99999)}"
    return f"Hotel {hotel_id} successfully booked. Booking reference: {booking_reference}"


@tool
def get_weather(location: str) -> str:
    """Get current weather information for a location.
    
    Args:
        location: The city or location to get weather for
        
    Returns:
        Weather description with temperature
    """
    weather_conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Stormy"]
    temperature = random.randint(15, 35)
    condition = random.choice(weather_conditions)
    return f"Weather in {location}: {condition}, {temperature}°C"


# Agent definition - simplified without manual tracing
def planner_agent(state: TravelPlanningState) -> Dict:
    """Planner agent that coordinates the overall travel planning."""
    try:
        # Get the messages
        messages = state["messages"]
        
        # Create LLM with tools
        llm = create_llm()
        
        tools = [search_flights, search_hotels, get_weather, get_match_schedule]
        llm_with_tools = llm.bind_tools(tools)
        
        # Build prompt
        system_message = SystemMessage(content="""You are a Travel Planner. You create comprehensive travel plans that include flights, 
        hotels, and activities based on user preferences. You coordinate the overall travel strategy.
        
        Current task: Plan a trip based on user requirements. Use the available tools to search for flights and hotels.
        When planning sports-related trips, first check the match schedule.
        When you have gathered enough information, create a FINAL TRAVEL PLAN summary.
        """)
        
        # Get response - tracing happens automatically through callbacks
        # Add agent metadata to the invocation
        response = llm_with_tools.invoke(
            [system_message] + messages,
            config={
                "metadata": {
                    "agent_name": "Travel Planner",
                    "agent_id": "planner-001",
                    "session_id": state.get("session_id", str(uuid4()))
                }
            }
        )
        
        # Check if the response is asking for more information
        waiting_for_user = False
        if isinstance(response, AIMessage) and response.content and not response.tool_calls:
            # If the AI is asking questions and not calling tools, it's waiting for user input
            if any(q in response.content.lower() for q in ["?", "please provide", "could you", "let me know", "need to know"]):
                waiting_for_user = True
        
        # Return updated state
        return {"messages": [response], "waiting_for_user": waiting_for_user}
        
    except Exception as e:
        print(f"❌ Error in planner agent: {e}")
        error_msg = AIMessage(content=f"I encountered an error: {str(e)}. Please check your configuration.")
        return {"messages": [error_msg], "waiting_for_user": False}

def should_continue(state: TravelPlanningState) -> Literal["tools", "end", "wait_for_user"]:
    """Determine if we should use tools, end the conversation, or wait for user input."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if we're waiting for user input
    if state.get("waiting_for_user", False):
        return "wait_for_user"
    
    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Check for final plan
    if isinstance(last_message, AIMessage) and "FINAL TRAVEL PLAN" in last_message.content.upper():
        return "end"
    
    return "end"


# Build the graph
def build_travel_planning_graph():
    """Build the state graph for travel planning."""
    # Create workflow graph
    workflow = StateGraph(TravelPlanningState)
    
    # Define tools with callback manager for automatic tracing
    tools = [search_flights, search_hotels, book_flight, book_hotel, get_weather, get_match_schedule]
    tool_node = ToolNode(tools)
    
    # Add nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add edges
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "tools": "tools",
            "end": END,
            "wait_for_user": END  # End the current execution to wait for user input
        }
    )
    
    # After tools, always go back to planner
    workflow.add_edge("tools", "planner")
    
    return workflow.compile()



def main():
    print("\n🌍 LangGraph Travel Planning System with Automatic Tracing 🌍")
    print("="*60)
    
    # Build the graph
    print("\nBuilding agent graph...")
    graph = build_travel_planning_graph()
    
    # Get user request
    print("\nExample requests:")
    print("- 'Plan a vacation to watch the first 3 premier league games at chelsea'")
    print("- 'Plan a trip to Tokyo for 2 people with a budget of $4000'")
    print("- 'I need a weekend getaway to Paris'")
    
    user_request = input("\n📝 Please describe your travel plans: ")
    
    # Generate session ID for the entire conversation
    session_id = str(uuid4())
    
    # Initialize state
    state = {
        "messages": [HumanMessage(content=user_request)],
        "destination": "",
        "origin": "",
        "budget": 0.0,
        "travelers": 1,
        "departure_date": "",
        "return_date": "",
        "flights": [],
        "hotels": [],
        "current_agent": "",
        "waiting_for_user": False,
        "session_id": session_id
    }
    
    # Execute the graph with automatic tracing enabled
    print("\n🤖 Starting the travel planning process...\n")
    
    conversation_complete = False
    
    # Create a root span for the entire session if OpenTelemetry is available
    root_span = None
    token = None
    if trace and context:
        tracer = trace.get_tracer(__name__)
        
        # Create invoke_agent span following the semantic conventions
        root_span = tracer.start_span(
            "invoke_agent Travel Planning Agent",  # Span name format: invoke_agent {gen_ai.agent.name}
            kind=SpanKind.CLIENT,  # SHOULD be CLIENT
            attributes={
                # Required attributes
                "gen_ai.operation.name": "invoke_agent",  # Required
                "gen_ai.provider.name": "langchain",  # Required - provider identification
                "gen_ai.system": "azure_openai",
                
                # Conditionally Required attributes
                "gen_ai.agent.name": "Travel Planning Agent",  # Human-readable name
                "gen_ai.agent.id": f"travel-planner-{session_id}",  # Unique identifier
                "gen_ai.agent.description": "AI agent that creates comprehensive travel plans including flights, hotels, and activities based on user preferences",
                "gen_ai.conversation.id": session_id,  # Conversation/session ID
                "gen_ai.request.model": "gpt-4.1",  # Model being used
                
                # Recommended attributes
                "gen_ai.agent.invocation_input": json.dumps([{
                    "role": "user",
                    "body": [{
                        "type": "text",
                        "content": user_request
                    }]
                }]),
                "server.address": "ai-naarkalgaihub999971652049.openai.azure.com",
                "server.port": 443,
                
                # Additional context
                "session.id": session_id,
                "user.request": user_request,
                "service.name": "travel-planning-system",
                "session.start_time": datetime.now().isoformat()
            }
        )
        
        # Set this span as the current span in context
        ctx = trace.set_span_in_context(root_span)
        token = context.attach(ctx)
    
    try:
        while not conversation_complete:
            try:
                # Create session-specific tags
                session_tags = ["travel-planning-execution", f"session-{session_id}"]
                
                # Execute graph with streaming and callbacks
                for step in graph.stream(
                    state, 
                    config={
                        "callbacks": callback_manager,
                        "tags": session_tags,
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "user_request": user_request,
                            "session_id": session_id
                        }
                    }
                ):
                    for node, output in step.items():
                        print(f"\n{'='*60}")
                        print(f"📍 Node: {node}")
                        print(f"{'='*60}")
                        
                        if "messages" in output:
                            for msg in output["messages"]:
                                if isinstance(msg, AIMessage):
                                    print(f"\n🤖 [AI]: {msg.content}")
                                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                                        print(f"\n🔧 [Tool Calls]: {[tc['name'] for tc in msg.tool_calls]}")
                                elif isinstance(msg, HumanMessage):
                                    # Only print if it's not the original request
                                    if msg.content != user_request:
                                        print(f"\n👤 [Human]: {msg.content}")
                                elif isinstance(msg, ToolMessage):
                                    print(f"\n🔨 [Tool - {msg.name}]: {msg.content}")
                        
                        # Update state with the output
                        state.update(output)
                
                # Check if the conversation is complete
                last_message = state["messages"][-1]
                if isinstance(last_message, AIMessage):
                    if "FINAL TRAVEL PLAN" in last_message.content.upper():
                        conversation_complete = True
                        print("\n" + "=" * 80)
                        print("✅ TRAVEL PLANNING COMPLETE")
                        print("=" * 80)
                        
                        # Add final output to root span
                        if root_span:
                            # Update with agent invocation output
                            root_span.set_attribute("gen_ai.agent.invocation_output", json.dumps([{
                                "role": "assistant",
                                "body": [{
                                    "type": "text",
                                    "content": last_message.content
                                }],
                                "finish_reason": "stop"
                            }]))
                            root_span.set_attribute("gen_ai.response.finish_reasons", ["stop"])
                            root_span.set_attribute("travel.plan.final", last_message.content)
                            root_span.add_event(
                                "final_plan_generated",
                                {"content": last_message.content}
                            )
                            
                    elif state.get("waiting_for_user", False):
                        # Get user response
                        user_response = input("\n📝 Your response: ")
                        if user_response.lower() in ["quit", "exit", "done"]:
                            print("\n👋 Ending conversation. Thank you!")
                            conversation_complete = True
                        else:
                            # Add user response to messages and continue
                            state["messages"].append(HumanMessage(content=user_response))
                            state["waiting_for_user"] = False
                            
                            # Add user interaction event to root span
                            if root_span:
                                root_span.add_event(
                                    "user_interaction",
                                    {"user_response": user_response}
                                )
                    else:
                        # If we get here and no tool calls, assume conversation is done
                        conversation_complete = True
                        
            except Exception as e:
                print(f"\n❌ Error during execution: {e}")
                print("\nTroubleshooting tips:")
                print("1. Check your .env file has valid Azure OpenAI credentials")
                print("2. Ensure Jaeger is running (docker-compose up)")
                print("3. Check OTEL_EXPORTER_OTLP_ENDPOINT is correct")
                print("4. For Azure tracing, verify APPLICATION_INSIGHTS_CONNECTION_STRING")
                print("5. For LangSmith tracing, set LANGCHAIN_API_KEY")
                
                # Record error in root span following the spec
                if root_span:
                    root_span.record_exception(e)
                    root_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    root_span.set_attribute("error.type", type(e).__name__)  # Required on error
                    
                conversation_complete = True
                
    finally:
        # End the root span when conversation is complete
        if root_span:
            root_span.set_status(trace.Status(trace.StatusCode.OK))
            root_span.end()
        
        # Detach the context
        if context and token:
            context.detach(token)



if __name__ == "__main__":
    main()