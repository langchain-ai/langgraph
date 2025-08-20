"""
Enhanced Multi-Agent Travel Planning System with Real Data Sources
Uses LangGraph for orchestration and integrates with real APIs for data.
"""

import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from uuid import uuid4
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment and configuration
from config import config
from data_sources import (
    search_flights_sync, search_hotels_sync, search_activities_sync, 
    get_weather_sync, DataSourceManager
)

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# Tracing imports
try:
    from otel_tracer import OpenTelemetryTracer
    from langchain_azure_ai.callbacks.tracers import AzureOpenAITracingCallback
except ImportError as e:
    logger.warning(f"Tracing imports failed: {e}")
    OpenTelemetryTracer = None
    AzureOpenAITracingCallback = None

try:
    from opentelemetry import trace, context
    from opentelemetry.trace import SpanKind
except ImportError:
    trace = None
    context = None
    SpanKind = None

# State definition
class TravelPlanState(TypedDict):
    """Enhanced state for travel planning with LangGraph."""
    messages: Annotated[List[AnyMessage], add_messages]
    user_request: str
    destination: Optional[str]
    origin: Optional[str]
    budget: Optional[float]
    travelers: int
    departure_date: Optional[str]
    return_date: Optional[str]
    trip_type: Optional[str]  # business, leisure, family, romantic
    preferences: Dict[str, Any]  # user preferences
    
    # Agent results
    flight_results: List[Dict]
    hotel_results: List[Dict]
    activity_results: List[Dict]
    weather_results: List[Dict]
    budget_analysis: Optional[str]
    
    # Planning state
    current_step: str
    completed_steps: List[str]
    final_plan: Optional[str]
    
    # Session management
    session_id: str
    agent_handoffs: List[Dict]

# Agent types
class AgentType(Enum):
    COORDINATOR = "coordinator"
    FLIGHT_SPECIALIST = "flight_specialist"
    HOTEL_SPECIALIST = "hotel_specialist"
    ACTIVITY_SPECIALIST = "activity_specialist"
    BUDGET_ANALYST = "budget_analyst"
    PLAN_SYNTHESIZER = "plan_synthesizer"

# Initialize tracing
def setup_tracing():
    """Set up tracing infrastructure."""
    tracers = []
    
    # Azure Application Insights
    if config.tracing.application_insights_connection_string and AzureOpenAITracingCallback:
        azure_tracer = AzureOpenAITracingCallback(
            connection_string=config.tracing.application_insights_connection_string,
            enable_content_recording=config.tracing.enable_content_recording
        )
        tracers.append(azure_tracer)
        logger.info("Azure tracing enabled")
    
    # OpenTelemetry
    if OpenTelemetryTracer:
        otel_tracer = OpenTelemetryTracer(
            service_name=config.tracing.otel_service_name,
            otlp_endpoint=config.tracing.otel_exporter_otlp_endpoint,
            enable_content_recording=config.tracing.enable_content_recording
        )
        tracers.append(otel_tracer)
        logger.info("OpenTelemetry tracing enabled")
    
    return tracers

# LLM setup
def create_llm(agent_type: str = "default") -> Any:
    """Create LLM instance with proper configuration."""
    tracers = setup_tracing()
    
    # Temperature based on agent type
    temperature_map = {
        "coordinator": config.llm.supervisor_temperature,
        "flight_specialist": config.llm.flight_specialist_temperature,
        "hotel_specialist": config.llm.hotel_specialist_temperature,
        "activity_specialist": config.llm.activity_specialist_temperature,
        "budget_analyst": config.llm.budget_analyst_temperature,
        "plan_synthesizer": config.llm.supervisor_temperature
    }
    temperature = temperature_map.get(agent_type, 0.7)
    
    # Try Azure OpenAI first
    if config.llm.azure_openai_api_key and config.llm.azure_openai_endpoint:
        return AzureChatOpenAI(
            azure_deployment=config.llm.azure_deployment_name,
            api_key=config.llm.azure_openai_api_key,
            azure_endpoint=config.llm.azure_openai_endpoint,
            api_version=config.llm.azure_openai_api_version,
            temperature=temperature,
            callbacks=tracers,
            tags=[agent_type, "enhanced-travel-planning"],
            metadata={
                "agent_type": agent_type,
                "system": "enhanced-travel-planning",
                "provider": "azure_openai"
            }
        )
    
    # Fallback to OpenAI
    elif config.llm.openai_api_key:
        return ChatOpenAI(
            model=config.llm.openai_model,
            api_key=config.llm.openai_api_key,
            temperature=temperature,
            callbacks=tracers,
            tags=[agent_type, "enhanced-travel-planning"],
            metadata={
                "agent_type": agent_type,
                "system": "enhanced-travel-planning",
                "provider": "openai"
            }
        )
    
    else:
        raise ValueError("No LLM configuration found. Please set Azure OpenAI or OpenAI credentials.")

# Enhanced Tools with Real Data
@tool
def search_flights(origin: str, destination: str, departure_date: str, travelers: int = 2) -> str:
    """Search for flights with real data from APIs or realistic fallbacks."""
    try:
        flights = search_flights_sync(origin, destination, departure_date, travelers)
        
        if not flights:
            return f"No flights found from {origin} to {destination} on {departure_date}"
        
        result = f"Found {len(flights)} flights from {origin} to {destination} on {departure_date}:\n\n"
        
        for i, flight in enumerate(flights, 1):
            result += f"{i}. {flight.airline} Flight {flight.flight_id}\n"
            result += f"   Departure: {flight.departure_time} | Arrival: {flight.arrival_time}\n"
            result += f"   Duration: {flight.duration} | Price: ${flight.price} {flight.currency}\n"
            result += f"   Available seats: {flight.available_seats}\n\n"
        
        return result.strip()
        
    except Exception as e:
        logger.error(f"Error searching flights: {e}")
        return f"Error searching flights: {str(e)}"

@tool
def search_hotels(destination: str, check_in: str, check_out: str, travelers: int = 2) -> str:
    """Search for hotels with real data from APIs or realistic fallbacks."""
    try:
        hotels = search_hotels_sync(destination, check_in, check_out, travelers)
        
        if not hotels:
            return f"No hotels found in {destination} for {check_in} to {check_out}"
        
        result = f"Found {len(hotels)} hotels in {destination} from {check_in} to {check_out}:\n\n"
        
        for i, hotel in enumerate(hotels, 1):
            result += f"{i}. {hotel.name} ({hotel.rating}⭐)\n"
            result += f"   Price: ${hotel.price_per_night} {hotel.currency}/night\n"
            result += f"   Location: {hotel.location}\n"
            if hotel.distance_to_center:
                result += f"   Distance to center: {hotel.distance_to_center}\n"
            result += f"   Amenities: {', '.join(hotel.amenities[:5])}\n\n"
        
        return result.strip()
        
    except Exception as e:
        logger.error(f"Error searching hotels: {e}")
        return f"Error searching hotels: {str(e)}"

@tool
def search_activities(destination: str, activity_type: Optional[str] = None) -> str:
    """Search for activities and attractions with real data from APIs."""
    try:
        activities = search_activities_sync(destination, activity_type)
        
        if not activities:
            return f"No activities found in {destination}"
        
        type_filter = f" ({activity_type})" if activity_type else ""
        result = f"Found {len(activities)} activities{type_filter} in {destination}:\n\n"
        
        for i, activity in enumerate(activities, 1):
            result += f"{i}. {activity.name}\n"
            result += f"   Category: {activity.category} | Duration: {activity.duration}\n"
            result += f"   Price: ${activity.price} {activity.currency} | Rating: {activity.rating}⭐\n"
            if activity.description:
                result += f"   Description: {activity.description[:100]}...\n"
            result += "\n"
        
        return result.strip()
        
    except Exception as e:
        logger.error(f"Error searching activities: {e}")
        return f"Error searching activities: {str(e)}"

@tool
def get_weather_forecast(location: str, date: str) -> str:
    """Get weather forecast for a location and date."""
    try:
        weather = get_weather_sync(location, date)
        
        result = f"Weather forecast for {weather.location} on {weather.date}:\n"
        result += f"Condition: {weather.condition}\n"
        result += f"Temperature: {weather.temperature_high}°C / {weather.temperature_low}°C\n"
        result += f"Humidity: {weather.humidity}%\n"
        result += f"Description: {weather.description}"
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting weather: {e}")
        return f"Error getting weather forecast: {str(e)}"

@tool
def analyze_budget(
    flight_cost: float, 
    hotel_cost_per_night: float, 
    nights: int, 
    activity_costs: List[float], 
    travelers: int,
    destination_currency: str = "USD"
) -> str:
    """Analyze trip budget and provide cost breakdown."""
    try:
        # Calculate totals
        total_flight_cost = flight_cost * travelers
        total_hotel_cost = hotel_cost_per_night * nights
        total_activity_cost = sum(activity_costs) * travelers
        
        # Add estimated meals and miscellaneous (varies by destination)
        meal_cost_per_day = {
            'London': 60, 'Paris': 55, 'New York': 70, 'Tokyo': 50, 'Rome': 45
        }.get(destination_currency, 50)
        
        total_meals = meal_cost_per_day * nights * travelers
        miscellaneous = (total_flight_cost + total_hotel_cost + total_activity_cost) * 0.15
        
        grand_total = total_flight_cost + total_hotel_cost + total_activity_cost + total_meals + miscellaneous
        per_person = grand_total / travelers
        
        result = f"💰 Trip Budget Analysis:\n\n"
        result += f"✈️  Flights (round trip): ${total_flight_cost:,.2f}\n"
        result += f"🏨 Hotels ({nights} nights): ${total_hotel_cost:,.2f}\n"
        result += f"🎯 Activities: ${total_activity_cost:,.2f}\n"
        result += f"🍽️  Meals ({nights} days): ${total_meals:,.2f}\n"
        result += f"💼 Miscellaneous (15%): ${miscellaneous:,.2f}\n"
        result += f"{'='*40}\n"
        result += f"💵 Total Cost: ${grand_total:,.2f}\n"
        result += f"👤 Per Person: ${per_person:,.2f}\n\n"
        
        # Budget recommendations
        if per_person < 1000:
            result += "💡 Budget-friendly trip with great value!"
        elif per_person < 2000:
            result += "💡 Moderate budget with good balance of cost and experience."
        elif per_person < 3500:
            result += "💡 Premium trip with luxury accommodations and experiences."
        else:
            result += "💡 Luxury travel experience with top-tier services."
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing budget: {e}")
        return f"Error analyzing budget: {str(e)}"

@tool
def extract_trip_details(user_request: str) -> str:
    """Extract and structure trip details from user request."""
    import re
    from datetime import datetime, timedelta
    
    details = {
        "destination": None,
        "origin": None, 
        "travelers": 2,
        "duration": 7,
        "trip_type": "leisure",
        "budget": None,
        "preferences": []
    }
    
    request_lower = user_request.lower()
    
    # Extract destination
    destinations = ['london', 'paris', 'new york', 'tokyo', 'rome', 'sydney', 'dubai']
    for dest in destinations:
        if dest in request_lower:
            details["destination"] = dest.title()
            break
    
    # Extract origin
    origins = ['seattle', 'san francisco', 'los angeles', 'chicago', 'miami']
    for orig in origins:
        if f"from {orig}" in request_lower:
            details["origin"] = orig.title()
            break
    
    # Extract number of travelers
    traveler_match = re.search(r'(\d+)\s*(?:people|person|traveler|guest)', request_lower)
    if traveler_match:
        details["travelers"] = int(traveler_match.group(1))
    
    # Extract trip type
    if any(word in request_lower for word in ['business', 'work', 'conference']):
        details["trip_type"] = "business"
    elif any(word in request_lower for word in ['family', 'kids', 'children']):
        details["trip_type"] = "family"
    elif any(word in request_lower for word in ['romantic', 'honeymoon', 'anniversary']):
        details["trip_type"] = "romantic"
    elif any(word in request_lower for word in ['luxury', 'premium', 'upscale']):
        details["trip_type"] = "luxury"
    
    # Extract budget
    budget_match = re.search(r'\$(\d+(?:,\d+)?)', user_request)
    if budget_match:
        details["budget"] = int(budget_match.group(1).replace(',', ''))
    
    # Extract preferences
    if 'luxury' in request_lower or 'upscale' in request_lower:
        details["preferences"].append("luxury_accommodations")
    if 'adventure' in request_lower:
        details["preferences"].append("adventure_activities")
    if 'culture' in request_lower or 'museum' in request_lower:
        details["preferences"].append("cultural_activities")
    if 'food' in request_lower or 'culinary' in request_lower:
        details["preferences"].append("culinary_experiences")
    
    return json.dumps(details, indent=2)

# Agent nodes for LangGraph
def coordinator_node(state: TravelPlanState) -> TravelPlanState:
    """Coordinator agent that manages the overall travel planning process."""
    llm = create_llm("coordinator")
    
    system_message = """You are the Travel Planning Coordinator. Your role is to:
    1. Understand the user's travel requirements
    2. Extract key details (destination, dates, travelers, preferences)
    3. Coordinate with specialist agents
    4. Ensure all aspects are covered
    
    Analyze the user request and determine the next steps needed."""
    
    messages = [SystemMessage(content=system_message)] + state["messages"]
    response = llm.invoke(messages)
    
    # Extract trip details if not already done
    if not state.get("destination"):
        tools = [extract_trip_details]
        tool_llm = llm.bind_tools(tools)
        tool_response = tool_llm.invoke([
            SystemMessage(content="Extract trip details from the user request."),
            HumanMessage(content=state["user_request"])
        ])
        
        if tool_response.tool_calls:
            details_str = extract_trip_details.invoke({"user_request": state["user_request"]})
            details = json.loads(details_str)
            
            # Update state with extracted details
            state.update({
                "destination": details.get("destination", "London"),
                "origin": details.get("origin", "Seattle"),
                "travelers": details.get("travelers", 2),
                "trip_type": details.get("trip_type", "leisure"),
                "preferences": details.get("preferences", [])
            })
    
    # Set default dates if not provided
    if not state.get("departure_date"):
        departure = datetime.now() + timedelta(days=30)
        return_date = departure + timedelta(days=7)
        state["departure_date"] = departure.strftime("%Y-%m-%d")
        state["return_date"] = return_date.strftime("%Y-%m-%d")
    
    state["current_step"] = "flight_search"
    state["messages"].append(response)
    
    return state

def flight_specialist_node(state: TravelPlanState) -> TravelPlanState:
    """Flight specialist agent with real flight search capabilities."""
    llm = create_llm("flight_specialist")
    tools = [search_flights, get_weather_forecast]
    
    agent = create_react_agent(llm, tools)
    
    task = f"""Search for flights from {state['origin']} to {state['destination']} 
    departing on {state['departure_date']} for {state['travelers']} travelers.
    
    Also check weather conditions for the travel dates to provide timing recommendations.
    
    Return detailed flight options with prices and recommendations."""
    
    result = agent.invoke({
        "messages": [HumanMessage(content=task)]
    })
    
    # Store results
    state["flight_results"] = result.get("messages", [])
    state["completed_steps"].append("flight_search")
    state["current_step"] = "hotel_search"
    
    # Add handoff information
    state["agent_handoffs"].append({
        "from": "coordinator",
        "to": "flight_specialist", 
        "timestamp": datetime.now().isoformat(),
        "task": "flight_search_and_weather"
    })
    
    return state

def hotel_specialist_node(state: TravelPlanState) -> TravelPlanState:
    """Hotel specialist agent with real hotel search capabilities."""
    llm = create_llm("hotel_specialist")
    tools = [search_hotels]
    
    agent = create_react_agent(llm, tools)
    
    # Customize search based on trip type and preferences
    hotel_requirements = []
    if state.get("trip_type") == "luxury" or "luxury_accommodations" in state.get("preferences", []):
        hotel_requirements.append("luxury hotels with premium amenities")
    elif state.get("trip_type") == "business":
        hotel_requirements.append("business-friendly hotels with meeting facilities")
    elif state.get("trip_type") == "family":
        hotel_requirements.append("family-friendly hotels with connecting rooms")
    elif state.get("trip_type") == "romantic":
        hotel_requirements.append("romantic hotels with couples amenities")
    
    requirements_text = " Focus on " + ", ".join(hotel_requirements) if hotel_requirements else ""
    
    task = f"""Search for hotels in {state['destination']} from {state['departure_date']} 
    to {state['return_date']} for {state['travelers']} travelers.{requirements_text}
    
    Consider location, amenities, and price. Provide detailed recommendations."""
    
    result = agent.invoke({
        "messages": [HumanMessage(content=task)]
    })
    
    state["hotel_results"] = result.get("messages", [])
    state["completed_steps"].append("hotel_search")
    state["current_step"] = "activity_search"
    
    state["agent_handoffs"].append({
        "from": "flight_specialist",
        "to": "hotel_specialist",
        "timestamp": datetime.now().isoformat(),
        "task": "hotel_search_and_recommendations"
    })
    
    return state

def activity_specialist_node(state: TravelPlanState) -> TravelPlanState:
    """Activity specialist agent with real activity search capabilities."""
    llm = create_llm("activity_specialist")
    tools = [search_activities, get_weather_forecast]
    
    agent = create_react_agent(llm, tools)
    
    # Customize activity search based on preferences
    activity_focus = []
    preferences = state.get("preferences", [])
    
    if "cultural_activities" in preferences:
        activity_focus.append("cultural and historical attractions")
    if "adventure_activities" in preferences:
        activity_focus.append("adventure and outdoor activities")
    if "culinary_experiences" in preferences:
        activity_focus.append("culinary experiences and food tours")
    
    if state.get("trip_type") == "business":
        activity_focus.append("networking events and professional attractions")
    elif state.get("trip_type") == "family":
        activity_focus.append("family-friendly activities and entertainment")
    elif state.get("trip_type") == "romantic":
        activity_focus.append("romantic experiences and couples activities")
    
    focus_text = " Focus on " + ", ".join(activity_focus) if activity_focus else ""
    
    task = f"""Search for activities and attractions in {state['destination']}.{focus_text}
    
    Trip duration: {state['departure_date']} to {state['return_date']}
    Travelers: {state['travelers']}
    
    Check weather conditions and create a balanced itinerary with indoor and outdoor options."""
    
    result = agent.invoke({
        "messages": [HumanMessage(content=task)]
    })
    
    state["activity_results"] = result.get("messages", [])
    state["completed_steps"].append("activity_search")
    state["current_step"] = "budget_analysis"
    
    state["agent_handoffs"].append({
        "from": "hotel_specialist",
        "to": "activity_specialist",
        "timestamp": datetime.now().isoformat(),
        "task": "activity_search_and_itinerary"
    })
    
    return state

def budget_analyst_node(state: TravelPlanState) -> TravelPlanState:
    """Budget analyst agent with comprehensive cost analysis."""
    llm = create_llm("budget_analyst")
    tools = [analyze_budget]
    
    agent = create_react_agent(llm, tools)
    
    # Calculate trip duration
    from datetime import datetime
    dep_date = datetime.strptime(state["departure_date"], "%Y-%m-%d")
    ret_date = datetime.strptime(state["return_date"], "%Y-%m-%d")
    nights = (ret_date - dep_date).days
    
    task = f"""Analyze the budget for this trip:
    
    Destination: {state['destination']}
    Travelers: {state['travelers']}
    Duration: {nights} nights
    Trip Type: {state.get('trip_type', 'leisure')}
    
    Based on the flight and hotel search results, calculate total costs including:
    - Round-trip flights
    - Hotel accommodations
    - Activities and attractions
    - Meals and dining
    - Miscellaneous expenses
    
    Provide a detailed breakdown and recommendations for cost optimization."""
    
    result = agent.invoke({
        "messages": [HumanMessage(content=task)]
    })
    
    state["budget_analysis"] = result.get("messages", [])
    state["completed_steps"].append("budget_analysis")
    state["current_step"] = "plan_synthesis"
    
    state["agent_handoffs"].append({
        "from": "activity_specialist",
        "to": "budget_analyst",
        "timestamp": datetime.now().isoformat(),
        "task": "budget_analysis_and_optimization"
    })
    
    return state

def plan_synthesizer_node(state: TravelPlanState) -> TravelPlanState:
    """Plan synthesizer that creates the final comprehensive travel plan."""
    llm = create_llm("plan_synthesizer")
    
    system_message = """You are the Plan Synthesizer. Create a comprehensive, well-structured travel plan based on all the specialist reports.

    Include:
    1. Executive Summary
    2. Trip Overview (dates, destination, travelers)
    3. Flight Recommendations
    4. Hotel Recommendations  
    5. Day-by-Day Itinerary
    6. Budget Summary
    7. Important Notes and Tips
    8. Emergency Contacts and Backup Plans
    
    Make it detailed, actionable, and easy to follow."""
    
    # Compile all agent results
    all_results = []
    
    if state.get("flight_results"):
        all_results.append("FLIGHT SPECIALIST FINDINGS:")
        for msg in state["flight_results"]:
            if hasattr(msg, 'content'):
                all_results.append(msg.content)
    
    if state.get("hotel_results"):
        all_results.append("\nHOTEL SPECIALIST FINDINGS:")
        for msg in state["hotel_results"]:
            if hasattr(msg, 'content'):
                all_results.append(msg.content)
    
    if state.get("activity_results"):
        all_results.append("\nACTIVITY SPECIALIST FINDINGS:")
        for msg in state["activity_results"]:
            if hasattr(msg, 'content'):
                all_results.append(msg.content)
    
    if state.get("budget_analysis"):
        all_results.append("\nBUDGET ANALYST FINDINGS:")
        for msg in state["budget_analysis"]:
            if hasattr(msg, 'content'):
                all_results.append(msg.content)
    
    compilation = "\n".join(all_results)
    
    task = f"""Create a comprehensive travel plan based on the following specialist reports:

    ORIGINAL REQUEST: {state['user_request']}
    
    DESTINATION: {state['destination']}
    ORIGIN: {state['origin']}
    DATES: {state['departure_date']} to {state['return_date']}
    TRAVELERS: {state['travelers']}
    TRIP TYPE: {state.get('trip_type', 'leisure')}
    
    SPECIALIST REPORTS:
    {compilation}
    
    Create a final, comprehensive travel plan that integrates all findings."""
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=task)
    ]
    
    response = llm.invoke(messages)
    
    state["final_plan"] = response.content
    state["completed_steps"].append("plan_synthesis")
    state["current_step"] = "completed"
    state["messages"].append(response)
    
    state["agent_handoffs"].append({
        "from": "budget_analyst",
        "to": "plan_synthesizer",
        "timestamp": datetime.now().isoformat(),
        "task": "final_plan_synthesis"
    })
    
    return state

# Conditional routing
def should_continue(state: TravelPlanState) -> str:
    """Determine next step in the planning process."""
    current_step = state.get("current_step", "start")
    
    if current_step == "start":
        return "coordinator"
    elif current_step == "flight_search":
        return "flight_specialist"
    elif current_step == "hotel_search":
        return "hotel_specialist"
    elif current_step == "activity_search":
        return "activity_specialist"
    elif current_step == "budget_analysis":
        return "budget_analyst"
    elif current_step == "plan_synthesis":
        return "plan_synthesizer"
    else:
        return END

# Create the enhanced travel planning graph
def create_travel_planning_graph():
    """Create the LangGraph workflow for travel planning."""
    
    # Create graph
    workflow = StateGraph(TravelPlanState)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("flight_specialist", flight_specialist_node)
    workflow.add_node("hotel_specialist", hotel_specialist_node)
    workflow.add_node("activity_specialist", activity_specialist_node)
    workflow.add_node("budget_analyst", budget_analyst_node)
    workflow.add_node("plan_synthesizer", plan_synthesizer_node)
    
    # Add edges
    workflow.add_conditional_edges(START, should_continue)
    workflow.add_conditional_edges("coordinator", should_continue)
    workflow.add_conditional_edges("flight_specialist", should_continue)
    workflow.add_conditional_edges("hotel_specialist", should_continue)
    workflow.add_conditional_edges("activity_specialist", should_continue)
    workflow.add_conditional_edges("budget_analyst", should_continue)
    workflow.add_conditional_edges("plan_synthesizer", should_continue)
    
    # Set up checkpointer
    checkpointer = MemorySaver()
    
    # Compile graph
    app = workflow.compile(checkpointer=checkpointer)
    
    return app

# Sample inputs for testing
SAMPLE_INPUTS = {
    "business_trip": {
        "request": "I need to plan a business trip to London for 3 days. I'll be traveling from Seattle, need a hotel near the financial district, and prefer efficient flights. Budget is around $3000.",
        "description": "Short business trip with specific location and budget requirements"
    },
    
    "family_vacation": {
        "request": "Plan a family vacation to Paris for 2 adults and 2 children for 10 days in summer. We love museums, parks, and family-friendly activities. Budget is flexible but around $8000 total.",
        "description": "Extended family trip with children, cultural focus"
    },
    
    "romantic_getaway": {
        "request": "Plan a romantic anniversary trip to Rome for 2 people for a week. We want luxury accommodations, fine dining, and romantic experiences. Budget up to $6000.",
        "description": "Luxury romantic trip with premium experiences"
    },
    
    "group_adventure": {
        "request": "Organize a group trip to Tokyo for 6 friends for 2 weeks. We're interested in both traditional culture and modern experiences, food tours, and adventure activities. Budget $4000 per person.",
        "description": "Large group with diverse interests and longer duration"
    },
    
    "cultural_exploration": {
        "request": "I want to explore New York for cultural experiences - museums, theaters, art galleries, and culinary tours. 5 days for 2 people, mid-range budget around $3500 total.",
        "description": "Culture-focused trip with specific activity preferences"
    }
}

def main():
    """Main execution function."""
    print("🌍 Enhanced Multi-Agent Travel Planning System 🌍")
    print("="*70)
    
    # Check configuration
    validation = config.validate_required_config()
    if not validation["llm_configured"]:
        print("❌ LLM configuration required!")
        print(config.get_missing_config_instructions())
        return
    
    missing_apis = [k for k, v in validation.items() if not v and k != "llm_configured"]
    if missing_apis:
        print("⚠️  Some APIs are not configured - will use realistic fallback data")
        print("For real data, configure these APIs:")
        for api in missing_apis:
            print(f"  - {api}")
        print()
    
    # Show sample inputs
    print("📝 Sample inputs you can try:")
    for key, sample in SAMPLE_INPUTS.items():
        print(f"\n{key.replace('_', ' ').title()}:")
        print(f"  \"{sample['request']}\"")
        print(f"  → {sample['description']}")
    
    print("\n" + "="*70)
    
    # Get user input
    user_input = input("\n✈️ Describe your travel plans (or press Enter for sample): ").strip()
    
    if not user_input:
        # Use first sample
        sample_key = list(SAMPLE_INPUTS.keys())[0]
        user_input = SAMPLE_INPUTS[sample_key]["request"]
        print(f"\n🎯 Using sample: {user_input}")
    
    # Create graph
    print("\n🔧 Initializing multi-agent system...")
    app = create_travel_planning_graph()
    
    # Initialize state
    session_id = str(uuid4())
    initial_state = TravelPlanState(
        messages=[HumanMessage(content=user_input)],
        user_request=user_input,
        destination=None,
        origin=None,
        budget=None,
        travelers=2,
        departure_date=None,
        return_date=None,
        trip_type=None,
        preferences={},
        flight_results=[],
        hotel_results=[],
        activity_results=[],
        weather_results=[],
        budget_analysis=None,
        current_step="start",
        completed_steps=[],
        final_plan=None,
        session_id=session_id,
        agent_handoffs=[]
    )
    
    # Configure execution
    config_dict = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 20
    }
    
    print("\n🚀 Starting travel planning process...\n")
    
    # Execute with tracing
    root_span = None
    if trace and SpanKind:
        tracer = trace.get_tracer(__name__)
        root_span = tracer.start_span(
            "enhanced_travel_planning_session",
            kind=SpanKind.SERVER,
            attributes={
                "session_id": session_id,
                "user_request": user_input,
                "system": "enhanced-multi-agent-travel-planning"
            }
        )
    
    try:
        # Stream the execution
        for step in app.stream(initial_state, config_dict):
            agent_name = list(step.keys())[0]
            agent_state = step[agent_name]
            
            print(f"🤖 {agent_name.replace('_', ' ').title()} Agent")
            print("="*50)
            
            if agent_state.get("messages"):
                last_message = agent_state["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    # Truncate very long outputs for readability
                    content = last_message.content
                    if len(content) > 1000:
                        content = content[:1000] + "... [truncated]"
                    print(content)
            
            print(f"\n✅ Completed: {', '.join(agent_state.get('completed_steps', []))}")
            print(f"🎯 Next: {agent_state.get('current_step', 'unknown')}")
            print()
        
        # Get final state
        final_state = app.get_state(config_dict)
        
        if final_state.values.get("final_plan"):
            print("\n" + "="*70)
            print("🎉 FINAL TRAVEL PLAN")
            print("="*70)
            print(final_state.values["final_plan"])
            print("="*70)
        
        # Show agent handoffs
        if final_state.values.get("agent_handoffs"):
            print("\n🔄 Agent Collaboration Summary:")
            for handoff in final_state.values["agent_handoffs"]:
                print(f"  {handoff['from']} → {handoff['to']}: {handoff['task']}")
        
        if root_span:
            root_span.set_attribute("planning_completed", True)
            root_span.set_attribute("agents_used", len(final_state.values.get("completed_steps", [])))
            
    except Exception as e:
        print(f"\n❌ Error during planning: {e}")
        traceback.print_exc()
        if root_span:
            root_span.record_exception(e)
    finally:
        if root_span:
            root_span.end()
    
    print("\n🎊 Travel planning completed!")
    print("Thank you for using the Enhanced Multi-Agent Travel Planning System!")

if __name__ == "__main__":
    main()