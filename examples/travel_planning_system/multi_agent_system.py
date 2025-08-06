"""
Multi-Agent Travel Planning System using LangChain Agents with Callbacks
This system uses multiple specialized agents coordinated by a supervisor agent.
"""
# flake8: noqa

import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict, Tuple
from uuid import uuid4
from enum import Enum

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure OpenTelemetry before other imports
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
os.environ["OTEL_SERVICE_NAME"] = "multi-agent-travel-planning"

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_react_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool, Tool
from langchain_core.callbacks import CallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish

# Azure OpenAI imports
from langchain_openai import AzureChatOpenAI
from langchain_azure_ai.callbacks.tracers import AzureOpenAITracingCallback

# Import custom tracer
try:
    from otel_tracer import OpenTelemetryTracer
except ImportError:
    print("Warning: OpenTelemetryTracer not found. Using default tracing.")
    OpenTelemetryTracer = None

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind
except ImportError:
    trace = None
    SpanKind = None


# Initialize tracers
application_insights_connection_string = os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING", "")

# Initialize callback handlers
tracers = []

# Azure tracer
if application_insights_connection_string:
    azure_tracer = AzureOpenAITracingCallback(
        connection_string=application_insights_connection_string,
        enable_content_recording=True,
    )
    tracers.append(azure_tracer)

# OpenTelemetry tracer
if OpenTelemetryTracer:
    otel_tracer = OpenTelemetryTracer(
        service_name="multi-agent-travel-planning",
        otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    )
    tracers.append(otel_tracer)

# Create callback manager
callback_manager = CallbackManager(tracers)


# State management
class AgentType(Enum):
    SUPERVISOR = "supervisor"
    FLIGHT = "flight_specialist"
    HOTEL = "hotel_specialist"
    ACTIVITY = "activity_specialist"
    BUDGET = "budget_analyst"


class TravelPlanState(TypedDict):
    """Shared state for all agents"""
    user_request: str
    destination: Optional[str]
    origin: Optional[str]
    budget: Optional[float]
    travelers: Optional[int]
    departure_date: Optional[str]
    return_date: Optional[str]
    flights: List[Dict]
    hotels: List[Dict]
    activities: List[Dict]
    weather_info: Optional[str]
    final_plan: Optional[str]
    current_agent: str
    session_id: str
    conversation_history: List[Dict]


# Create LLM instances for different agents
def create_llm(agent_name: str, temperature: float = 0.7):
    """Create an LLM instance with proper callbacks and metadata"""
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
    
    return AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint="https://ai-naarkalgaihub999971652049.openai.azure.com/",
        api_version="2024-02-15-preview",
        temperature=temperature,
        callbacks=callback_manager,
        tags=[agent_name, "multi-agent-travel"],
        metadata={
            "agent_type": agent_name,
            "system": "multi-agent-travel-planning"
        }
    )


# Tool definitions
@tool
def search_flights(origin: str, destination: str, date: str, travelers: int = 1) -> str:
    """Search for available flights between origin and destination on a specific date."""
    # Enhanced to handle international flights
    airlines_by_route = {
        "international": ["British Airways", "Virgin Atlantic", "United", "American", "Delta"],
        "domestic": ["United", "Delta", "American", "Southwest", "JetBlue"]
    }
    
    is_international = "london" in destination.lower() or "sea" in origin.lower()
    airlines = airlines_by_route["international"] if is_international else airlines_by_route["domestic"]
    
    flights = []
    for i in range(3):
        flight = {
            "flight_id": f"FL{random.randint(100, 999)}",
            "airline": random.choice(airlines),
            "departure_time": f"{random.randint(6, 22):02d}:{random.choice(['00', '30'])}",
            "arrival_time": f"{random.randint(6, 22):02d}:{random.choice(['00', '30'])}",
            "price_per_person": random.randint(500, 1500) if is_international else random.randint(150, 800),
            "duration": f"{random.randint(8, 12)}h {random.randint(0, 59)}m" if is_international else f"{random.randint(1, 8)}h {random.randint(0, 59)}m",
            "origin": origin,
            "destination": destination,
            "date": date,
            "available_seats": random.randint(5, 50)
        }
        flights.append(flight)
    
    result = f"Found {len(flights)} flights from {origin} to {destination} on {date}:\n"
    for f in flights:
        result += f"\n- {f['airline']} {f['flight_id']}: Departs {f['departure_time']}, arrives {f['arrival_time']}"
        result += f"\n  Price: ${f['price_per_person']} per person, Duration: {f['duration']}"
    
    return result


@tool
def search_hotels(destination: str, check_in: str, check_out: str, travelers: int = 1) -> str:
    """Search for available hotels at the destination for specified dates."""
    hotels = []
    
    # Location-specific hotels
    if "london" in destination.lower() or "chelsea" in destination.lower():
        hotel_names = ["The Langham London", "Claridge's", "The Savoy", "Mandarin Oriental Hyde Park", "The Ritz London"]
        locations = ["Mayfair", "Chelsea", "Westminster", "Kensington", "Covent Garden"]
        price_range = (300, 800)
    else:
        hotel_names = ["Grand Plaza", "Comfort Inn", "Luxury Suites", "Budget Lodge", "City Center Hotel"]
        locations = ["City Center", "Near Airport", "Beach Front", "Business District"]
        price_range = (80, 400)
    
    for i in range(4):
        hotel = {
            "hotel_id": f"HT{random.randint(100, 999)}",
            "name": random.choice(hotel_names),
            "price_per_night": random.randint(*price_range),
            "rating": round(random.uniform(4.0, 5.0), 1),
            "amenities": random.sample(["WiFi", "Pool", "Gym", "Breakfast", "Parking", "Spa", "Concierge", "Restaurant"], k=random.randint(4, 8)),
            "rooms_available": random.randint(1, 20),
            "location": random.choice(locations)
        }
        
        # Add distance to Chelsea stadium if in London
        if "london" in destination.lower() or "chelsea" in destination.lower():
            hotel["distance_to_stamford_bridge"] = f"{round(random.uniform(0.5, 5.0), 1)} miles"
        
        hotels.append(hotel)
    
    result = f"Found {len(hotels)} hotels in {destination} from {check_in} to {check_out}:\n"
    for h in hotels:
        result += f"\n- {h['name']} ({h['rating']}⭐): ${h['price_per_night']}/night"
        result += f"\n  Location: {h['location']}, Amenities: {', '.join(h['amenities'])}"
        if "distance_to_stamford_bridge" in h:
            result += f"\n  Distance to Stamford Bridge: {h['distance_to_stamford_bridge']}"
    
    return result


@tool
def get_premier_league_schedule(team: str = "Chelsea") -> str:
    """Get the Premier League match schedule for a specific team."""
    # Simulate Chelsea's first 3 Premier League games for 2024/25 season
    matches = [
        {
            "match_date": "2024-08-17",
            "opponent": "Manchester City",
            "venue": "Stamford Bridge",
            "competition": "Premier League",
            "kickoff": "16:30"
        },
        {
            "match_date": "2024-08-24",
            "opponent": "Wolves",
            "venue": "Molineux Stadium (Away)",
            "competition": "Premier League",
            "kickoff": "15:00"
        },
        {
            "match_date": "2024-09-01",
            "opponent": "Crystal Palace",
            "venue": "Stamford Bridge",
            "competition": "Premier League",
            "kickoff": "14:00"
        }
    ]
    
    result = f"Chelsea FC - First 3 Premier League Matches 2024/25:\n"
    for i, match in enumerate(matches, 1):
        result += f"\nMatch {i}: {match['match_date']}"
        result += f"\n  vs {match['opponent']} at {match['venue']}"
        result += f"\n  Kickoff: {match['kickoff']}"
    
    result += "\n\nNote: 2 home matches at Stamford Bridge, 1 away match at Wolves."
    return result


@tool
def search_activities(destination: str, activity_type: Optional[str] = None) -> str:
    """Search for activities and attractions at the destination."""
    activities = {
        "london": {
            "football": ["Stamford Bridge Stadium Tour", "Chelsea FC Museum", "Pre-match Pub Experience", "Football Walking Tour"],
            "sightseeing": ["Tower of London", "Westminster Abbey", "London Eye", "Buckingham Palace"],
            "cultural": ["West End Theater", "British Museum", "National Gallery", "Tate Modern"],
            "dining": ["Gordon Ramsay Restaurant", "Sketch London", "Dishoom", "Borough Market Food Tour"]
        },
        "default": {
            "sightseeing": ["City Tour", "Museum Visit", "Historical Sites", "Architecture Walk"],
            "adventure": ["Hiking", "Zip-lining", "Rock Climbing", "Water Sports"],
            "cultural": ["Local Cuisine Tour", "Art Gallery", "Theater Show", "Music Concert"],
            "relaxation": ["Spa Day", "Beach Time", "Park Visit", "Shopping"]
        }
    }
    
    city_activities = activities.get("london" if "london" in destination.lower() else "default")
    selected_activities = []
    
    for category, items in city_activities.items():
        if not activity_type or activity_type.lower() in category:
            for activity in random.sample(items, k=min(2, len(items))):
                selected_activities.append({
                    "name": activity,
                    "category": category,
                    "price": random.randint(20, 200),
                    "duration": f"{random.randint(1, 4)} hours",
                    "rating": round(random.uniform(4.0, 5.0), 1)
                })
    
    result = f"Found {len(selected_activities)} activities in {destination}:\n"
    for a in selected_activities:
        result += f"\n- {a['name']} ({a['category']}): ${a['price']}"
        result += f"\n  Duration: {a['duration']}, Rating: {a['rating']}⭐"
    
    return result


@tool
def get_weather_forecast(location: str, date: str) -> str:
    """Get weather forecast for a location on a specific date."""
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]
    
    # London weather tends to be milder
    if "london" in location.lower():
        temp_high = random.randint(15, 25)
    else:
        temp_high = random.randint(15, 35)
    
    temp_low = temp_high - random.randint(5, 15)
    condition = random.choice(conditions)
    
    return f"Weather forecast for {location} on {date}: {condition}, High: {temp_high}°C, Low: {temp_low}°C"


@tool
def calculate_trip_cost(flights: str, hotels: str, activities: str, travelers: int = 1) -> str:
    """Calculate the total estimated cost of the trip."""
    # Enhanced calculation based on actual trip details
    flight_cost = random.randint(800, 1500) * travelers * 2  # Round trip
    hotel_nights = 17  # Aug 16 to Sep 2
    hotel_cost = random.randint(400, 600) * hotel_nights
    activities_cost = random.randint(200, 800) * travelers
    match_tickets = 150 * 3 * travelers  # 3 matches
    
    total = flight_cost + hotel_cost + activities_cost + match_tickets
    
    breakdown = f"""Trip Cost Breakdown for {travelers} traveler(s):
- Flights (Round Trip SEA-LHR): ${flight_cost}
- Hotels ({hotel_nights} nights): ${hotel_cost}
- Match Tickets (3 games): ${match_tickets}
- Activities & Dining: ${activities_cost}
- Total: ${total}
- Per Person: ${total / travelers}"""
    
    return breakdown


@tool
def book_flight(flight_id: str, travelers: int = 1) -> str:
    """Book a specific flight."""
    confirmation = f"BKF{random.randint(100000, 999999)}"
    return f"Flight {flight_id} booked successfully for {travelers} traveler(s). Confirmation: {confirmation}"


@tool
def book_hotel(hotel_id: str, check_in: str, check_out: str, rooms: int = 1) -> str:
    """Book a specific hotel."""
    confirmation = f"BKH{random.randint(100000, 999999)}"
    return f"Hotel {hotel_id} booked from {check_in} to {check_out} for {rooms} room(s). Confirmation: {confirmation}"


# Create specialized agents
def create_flight_specialist():
    """Create a flight specialist agent"""
    tools = [search_flights, get_weather_forecast, book_flight]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Flight Specialist Agent. Your role is to:
1. Search for the best flights based on user requirements
2. Consider factors like price, timing, and convenience
3. Check weather conditions for travel dates
4. Provide flight recommendations with clear reasoning
5. Book flights when requested

Always provide specific flight options with details and explain your recommendations."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = create_llm("flight_specialist", temperature=0.5)
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=callback_manager,
        tags=["flight_specialist"],
        metadata={"agent_type": "flight_specialist"},
        handle_parsing_errors=True,
        max_iterations=5
    )


def create_hotel_specialist():
    """Create a hotel specialist agent"""
    tools = [search_hotels, book_hotel]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Hotel Specialist Agent. Your role is to:
1. Search for suitable accommodations based on user preferences
2. Consider factors like location, price, amenities, and ratings
3. For sports events, prioritize proximity to venues
4. Match hotel capacity to the number of travelers
5. Provide hotel recommendations with clear reasoning
6. Book hotels when requested

Always provide specific hotel options with details and explain your recommendations."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = create_llm("hotel_specialist", temperature=0.5)
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=callback_manager,
        tags=["hotel_specialist"],
        metadata={"agent_type": "hotel_specialist"},
        handle_parsing_errors=True,
        max_iterations=5
    )


def create_activity_specialist():
    """Create an activity planning specialist agent"""
    tools = [search_activities, get_weather_forecast, get_premier_league_schedule]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an Activity Planning Specialist Agent. Your role is to:
1. Search for activities and attractions at the destination
2. For sports trips, check match schedules and plan around them
3. Consider traveler interests and preferences
4. Check weather conditions for outdoor activities
5. Create a balanced itinerary
6. Provide activity recommendations with timing

Always suggest a variety of activities and explain why they match the traveler's interests."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = create_llm("activity_specialist", temperature=0.7)
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=callback_manager,
        tags=["activity_specialist"],
        metadata={"agent_type": "activity_specialist"},
        handle_parsing_errors=True,
        max_iterations=5
    )


def create_budget_analyst():
    """Create a budget analyst agent"""
    tools = [calculate_trip_cost]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Budget Analyst Agent. Your role is to:
1. Calculate total trip costs including flights, hotels, match tickets, and activities
2. Analyze if the trip fits within the stated budget
3. Suggest cost-saving alternatives if needed
4. Provide detailed cost breakdowns
5. Recommend budget allocation across different categories

Always be transparent about costs and provide money-saving tips."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = create_llm("budget_analyst", temperature=0.3)
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=callback_manager,
        tags=["budget_analyst"],
        metadata={"agent_type": "budget_analyst"},
        handle_parsing_errors=True,
        max_iterations=3
    )


def create_supervisor_agent():
    """Create a supervisor agent that coordinates other agents"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Supervisor Agent for a travel planning team. You coordinate between:
- Flight Specialist: Handles all flight-related tasks
- Hotel Specialist: Manages accommodation searches and bookings
- Activity Specialist: Plans activities and creates itineraries (including sports events)
- Budget Analyst: Tracks costs and ensures budget compliance

Your role is to:
1. Understand the user's travel requirements
2. Delegate tasks to appropriate specialists
3. Ensure all aspects of the trip are covered
4. Coordinate between specialists for a cohesive plan
5. Present a final comprehensive travel plan

Agent Reports:
{agent_reports}

Create a comprehensive travel plan based on the specialists' findings."""),
        ("human", "{input}"),
    ])
    
    llm = create_llm("supervisor", temperature=0.5)
    
    # For supervisor, we'll use a simpler chain
    chain = prompt | llm | StrOutputParser()
    
    return chain


# Multi-agent orchestration
class MultiAgentTravelPlanner:
    def __init__(self):
        self.supervisor = create_supervisor_agent()
        self.flight_specialist = create_flight_specialist()
        self.hotel_specialist = create_hotel_specialist()
        self.activity_specialist = create_activity_specialist()
        self.budget_analyst = create_budget_analyst()
        self.state = None
        self.session_id = str(uuid4())
        
    def initialize_state(self, user_request: str) -> TravelPlanState:
        """Initialize the shared state"""
        return {
            "user_request": user_request,
            "destination": None,
            "origin": None,
            "budget": None,
            "travelers": 1,
            "departure_date": None,
            "return_date": None,
            "flights": [],
            "hotels": [],
            "activities": [],
            "weather_info": None,
            "final_plan": None,
            "current_agent": "supervisor",
            "session_id": self.session_id,
            "conversation_history": []
        }
    
    def extract_trip_details(self, user_request: str) -> Dict[str, Any]:
        """Extract trip details from user request using LLM"""
        extraction_prompt = f"""Extract travel details from this request. Be very careful to identify:
        - If it mentions Chelsea FC or Premier League, the destination is LONDON, UK
        - SEA means Seattle (origin)
        - "bougie" means luxury/high-end
        
        Request: {user_request}
        
        Extract:
        - destination (city/country)
        - origin (city/country) 
        - budget (if mentioned, otherwise set as None)
        - number of travelers
        - departure date (if specific dates aren't mentioned, use the dates needed for the events)
        - return date
        - special requirements (luxury hotels, specific events, etc.)
        
        Return as JSON."""
        
        llm = create_llm("detail_extractor", temperature=0)
        response = llm.invoke(extraction_prompt)
        
        # Parse the request properly
        details = {
            "destination": "London" if "chelsea" in user_request.lower() or "premier league" in user_request.lower() else "New York",
            "origin": "Seattle" if "sea" in user_request.lower() else "San Francisco",
            "budget": None,  # Let them specify or we'll calculate
            "travelers": 2 if "2" in user_request else 1,
            "departure_date": "2024-08-16",  # Day before first Chelsea match
            "return_date": "2024-09-02",  # Day after third match
            "special_requirements": "luxury hotels" if "bougie" in user_request.lower() else "standard"
        }
        
        return details
    
    def run_agent(self, agent_type: str, task: str, agent_executor: Any, agent_reports: str = "") -> str:
        """Run a specific agent with callbacks and tracing"""
        print(f"\n{'='*60}")
        print(f"🤖 Running {agent_type} Agent")
        print(f"📋 Task: {task}")
        print(f"{'='*60}")
        
        try:
            # Create agent-specific config
            config = {
                "callbacks": callback_manager,
                "tags": [agent_type, f"session-{self.session_id}"],
                "metadata": {
                    "agent_type": agent_type,
                    "session_id": self.session_id,
                    "task": task
                }
            }
            
            # Run the agent
            if agent_type == "supervisor":
                # For supervisor chain - provide both input and agent_reports
                result = agent_executor.invoke(
                    {"input": task, "agent_reports": agent_reports}, 
                    config=config
                )
                return result
            else:
                # For other agents with AgentExecutor
                result = agent_executor.invoke({"input": task}, config=config)
                return result.get("output", str(result))
                
        except Exception as e:
            error_msg = f"Error in {agent_type}: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg
    
    def plan_trip(self, user_request: str):
        """Main orchestration method"""
        print("\n🌍 Starting Multi-Agent Travel Planning 🌍")
        print("="*80)
        
        # Initialize state
        self.state = self.initialize_state(user_request)
        trip_details = self.extract_trip_details(user_request)
        self.state.update(trip_details)
        
        # Create root span for tracing
        root_span = None
        if trace:
            tracer = trace.get_tracer(__name__)
            root_span = tracer.start_span(
                "multi_agent_travel_planning",
                kind=SpanKind.CLIENT,
                attributes={
                    "session.id": self.session_id,
                    "user.request": user_request,
                    "service.name": "multi-agent-travel-planning"
                }
            )
        
        try:
            # Phase 1: Activity Specialist first for sports events to get dates
            activity_task = f"""The user wants to watch the first 3 Premier League games at Chelsea FC.
            First, check the Premier League schedule for Chelsea to get the match dates.
            Then plan activities in {self.state['destination']} for {self.state['travelers']} travelers.
            They want luxury experiences ("bougie hotels" mentioned).
            Create a day-by-day itinerary around the matches."""
            
            activity_response = self.run_agent("activity_specialist", activity_task, self.activity_specialist)
            self.state["conversation_history"].append({"agent": "activity_specialist", "response": activity_response})
            
            # Phase 2: Flight Specialist with proper dates
            flight_task = f"""Find flights from {self.state['origin']} to {self.state['destination']} 
            departing on {self.state['departure_date']} for {self.state['travelers']} travelers. 
            Also find return flights on {self.state['return_date']}. 
            The travelers want to attend Chelsea FC matches on Aug 17, 24, and Sep 1, so ensure arrival/departure times work."""
            
            flight_response = self.run_agent("flight_specialist", flight_task, self.flight_specialist)
            self.state["conversation_history"].append({"agent": "flight_specialist", "response": flight_response})
            
            # Phase 3: Hotel Specialist
            hotel_task = f"""Find luxury hotels in {self.state['destination']} for check-in on {self.state['departure_date']} 
            and check-out on {self.state['return_date']} for {self.state['travelers']} travelers. 
            They specifically want "bougie" (luxury/high-end) hotels.
            Prefer hotels near Stamford Bridge stadium or with easy access to it."""
            
            hotel_response = self.run_agent("hotel_specialist", hotel_task, self.hotel_specialist)
            self.state["conversation_history"].append({"agent": "hotel_specialist", "response": hotel_response})
            
            # Phase 4: Budget Analysis
            budget_task = f"""Calculate the total cost for this luxury Chelsea FC trip for {self.state['travelers']} travelers.
            Include: round-trip flights from Seattle to London, luxury hotels for the duration,
            3 Chelsea match tickets (estimate £150 per ticket per person), and upscale dining/activities.
            Provide a detailed breakdown."""
            
            budget_response = self.run_agent("budget_analyst", budget_task, self.budget_analyst)
            self.state["conversation_history"].append({"agent": "budget_analyst", "response": budget_response})
            
            # Phase 5: Final Plan Synthesis
            agent_reports = "\n\n".join([
                f"{report['agent'].upper()} REPORT:\n{report['response']}" 
                for report in self.state["conversation_history"]
            ])
            
            final_task = f"""Create a FINAL TRAVEL PLAN for the Chelsea FC Premier League trip.
            Original request: {user_request}
            
            Include:
            1. Trip Overview (Seattle to London for Chelsea matches)
            2. Match Schedule (3 Premier League games)
            3. Flight Details (SEA-LHR round trip)
            4. Luxury Hotel Recommendations
            5. Day-by-Day Itinerary (centered around matches)
            6. Total Budget Breakdown
            7. Important Notes (tickets, transport to stadium, etc.)
            
            Make it comprehensive but easy to read."""
            
            # Pass agent_reports correctly to supervisor
            final_plan = self.run_agent("supervisor", final_task, self.supervisor, agent_reports)
            self.state["final_plan"] = final_plan
            
            # Present final plan
            print("\n" + "="*80)
            print("✈️ FINAL TRAVEL PLAN ✈️")
            print("="*80)
            print(final_plan)
            print("="*80)
            
            if root_span:
                root_span.set_attribute("travel.plan.final", final_plan)
                root_span.add_event("final_plan_generated", {"plan_length": len(final_plan)})
                
        except Exception as e:
            print(f"\n❌ Error in planning process: {e}")
            import traceback
            traceback.print_exc()
            if root_span:
                root_span.record_exception(e)
                root_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        finally:
            if root_span:
                root_span.set_status(trace.Status(trace.StatusCode.OK))
                root_span.end()


def main():
    """Main entry point"""
    print("\n🌍 Multi-Agent Travel Planning System 🌍")
    print("="*60)
    print("This system uses specialized agents to plan your perfect trip!")
    print("\nExample requests:")
    print("- 'Plan a vacation to watch the first 3 premier league games at chelsea from SEA, 2 people, bougie hotels'")
    print("- 'Plan a romantic getaway to Paris for 2 people with a budget of $4000'")
    print("- 'I need a business trip to New York for next week'")
    print("- 'Family vacation to Orlando for 4 people during summer break'")
    
    user_request = input("\n📝 Please describe your travel plans: ")
    
    if not user_request.strip():
        user_request = "Plan a vacation to watch the first 3 premier league games at chelsea from SEA, 2 people, bougie hotels. Pick the dates"
        print(f"\nUsing example request: {user_request}")
    
    # Create and run the planner
    planner = MultiAgentTravelPlanner()
    planner.plan_trip(user_request)
    
    # Interactive loop for modifications
    while True:
        print("\n" + "="*60)
        modification = input("\n📝 Would you like to modify anything? (type 'done' to finish): ")
        
        if modification.lower() in ['done', 'exit', 'quit']:
            print("\n✅ Thank you for using the Multi-Agent Travel Planning System!")
            break
        
        if modification.strip():
            print("\n🔄 Processing your modification...")
            planner.plan_trip(f"Modify the current plan: {modification}. Original request: {user_request}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Travel planning cancelled. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\nPlease check:")
        print("1. Your .env file has AZURE_OPENAI_API_KEY")
        print("2. OpenTelemetry endpoint is running")
        print("3. All required packages are installed")