"""
Multi-Agent Travel Planning System (LangGraph Version)
=====================================================


Features:
- LangGraph state machine orchestrating 6 agent roles:
  * coordinator
  * flight_specialist
  * hotel_specialist
  * activity_specialist
  * budget_analyst
  * plan_synthesizer
- Structured TravelPlanState with message history, agent handoffs, intermediate results
- Tool functions with graceful fallbacks (simulated realistic data if real APIs absent)
- Trip detail extraction
- Budget analysis + synthesized final plan
- Optional Azure OpenAI or OpenAI support (auto-detect via env vars)
- Optional tracing:
  * Azure Application Insights via langchain_azure_ai callbacks (if installed & configured)
  * OpenTelemetry (if libraries and OTLP endpoint available)
- Sample inputs & interactive CLI
- Recursion limit protection
- Rich metadata tags for observability

Environment Variables (auto-detected):
- Azure: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT
- OpenAI: OPENAI_API_KEY
- Tracing: APPLICATION_INSIGHTS_CONNECTION_STRING, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME
"""

from __future__ import annotations

import os
import json
import random
import logging
from uuid import uuid4
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env if present

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("multi_agent_travel_graph")

# ---------------------------------------------------------------------------
# Optional tracing imports
# ---------------------------------------------------------------------------
AzureOpenAITracingCallback = None
OpenTelemetryTracer = None
try:
    from langchain_azure_ai.callbacks.tracers import AzureOpenAITracingCallback
except Exception as e:
    logger.debug(f"Azure tracing callback unavailable: {e}")

try:
    from otel_tracer import OpenTelemetryTracer  # custom tracer if present
except Exception:
    # Fallback to no custom tracer
    OpenTelemetryTracer = None

try:
    from opentelemetry import trace, context
    from opentelemetry.trace import SpanKind
except Exception:
    trace = None
    context = None
    SpanKind = None

# ---------------------------------------------------------------------------
# LangChain / LangGraph Imports
# ---------------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# If user lacks dependencies, provide guidance
missing_core = []
for dep in ["langgraph", "langchain_openai"]:
    # Light check, rely on actual import success above
    pass

# ---------------------------------------------------------------------------
# Agent Descriptions (for metadata / tracing enrichment)
# ---------------------------------------------------------------------------
AGENT_DESCRIPTIONS: Dict[str, str] = {
    "coordinator": "Coordinates planning, extracts details, and orchestrates specialist agents",
    "flight_specialist": "Searches flights and checks weather; returns options and timing recommendations",
    "hotel_specialist": "Finds hotels tailored to trip type, preferences, and location",
    "activity_specialist": "Finds activities and builds a balanced itinerary considering weather",
    "budget_analyst": "Calculates and optimizes budget with detailed cost breakdown",
    "plan_synthesizer": "Synthesizes a final plan from all agent results",
}

# ---------------------------------------------------------------------------
# Tracing Setup
# ---------------------------------------------------------------------------
_TRACERS: Optional[List[Any]] = None


def setup_tracing() -> List[Any]:
    """Idempotently set up tracing callbacks."""
    global _TRACERS
    if _TRACERS is not None:
        return _TRACERS

    tracers: List[Any] = []

    app_insights_conn = os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING", "")
    if app_insights_conn and AzureOpenAITracingCallback:
        try:
            azure_tracer = AzureOpenAITracingCallback(
                connection_string=app_insights_conn,
                enable_content_recording=True,
                name="multi_agent_travel_planner",
                id="multi_agent_travel_planner",
                endpoint="travel_planning_system",
                scope="multi_agent_travel",
            )
            tracers.append(azure_tracer)
            logger.info("Azure Application Insights tracing enabled.")
        except Exception as e:
            logger.warning(f"Failed to init Azure tracing: {e}")

    if OpenTelemetryTracer:
        try:
            otlp_endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"
            )
            service_name = os.environ.get(
                "OTEL_SERVICE_NAME", "multi-agent-travel-planning"
            )
            otel_tracer = OpenTelemetryTracer(
                service_name=service_name,
                otlp_endpoint=otlp_endpoint,
                enable_content_recording=True,
            )
            tracers.append(otel_tracer)
            logger.info("Custom OpenTelemetry tracer enabled.")
        except Exception as e:
            logger.warning(f"Failed to init OpenTelemetry tracer: {e}")

    _TRACERS = tracers
    return tracers


def _trace_config_for_agent(agent_name: str, session_id: Optional[str]) -> Dict[str, Any]:
    """Return an invoke config dict with callbacks and metadata for the LLM/tool run."""
    tracers = setup_tracing()
    return {
        "callbacks": tracers,
        "tags": [f"agent:{agent_name}", agent_name, "multi_agent_travel"],
        "metadata": {
            "agent_name": agent_name,
            "agent_type": agent_name,
            "session_id": session_id,
            "thread_id": session_id,
            "system": "multi_agent_travel_planning",
            "agent_description": AGENT_DESCRIPTIONS.get(agent_name, agent_name),
        },
    }


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class TravelPlanState(TypedDict):
    """Graph state shared among nodes."""
    messages: Annotated[List[AnyMessage], add_messages]
    user_request: str
    destination: Optional[str]
    origin: Optional[str]
    budget: Optional[float]
    travelers: int
    departure_date: Optional[str]
    return_date: Optional[str]
    trip_type: Optional[str]
    preferences: Dict[str, Any]

    # Specialist outputs
    flight_results: List[Dict]
    hotel_results: List[Dict]
    activity_results: List[Dict]
    weather_results: List[Dict]
    budget_analysis: Optional[str]

    # Process control
    current_step: str
    completed_steps: List[str]
    final_plan: Optional[str]

    # Session & observability
    session_id: str
    agent_handoffs: List[Dict]


class AgentType(str, Enum):
    COORDINATOR = "coordinator"
    FLIGHT_SPECIALIST = "flight_specialist"
    HOTEL_SPECIALIST = "hotel_specialist"
    ACTIVITY_SPECIALIST = "activity_specialist"
    BUDGET_ANALYST = "budget_analyst"
    PLAN_SYNTHESIZER = "plan_synthesizer"


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------
def create_llm(
    agent_type: str,
    *,
    system_instructions: Optional[str],
    session_id: Optional[str],
) -> Any:
    """Create an LLM instance for an agent, preferring Azure OpenAI then OpenAI."""
    tracers = setup_tracing()

    # Temperature mapping; override or expand as needed
    temperature_map = {
        "coordinator": 0.3,
        "flight_specialist": 0.4,
        "hotel_specialist": 0.4,
        "activity_specialist": 0.7,
        "budget_analyst": 0.2,
        "plan_synthesizer": 0.45,
    }
    temperature = temperature_map.get(agent_type, 0.5)

    # Azure configuration
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    openai_key = os.environ.get("OPENAI_API_KEY")

    meta = {
        "agent_type": agent_type,
        "agent_name": agent_type,
        "agent_description": AGENT_DESCRIPTIONS.get(agent_type, agent_type),
        "system": "multi_agent_travel_planning",
        "system_instructions": system_instructions,
        "provider": None,
        "ls_provider": None,
        "ls_model_name": None,
        "ls_model_type": "chat",
        "ls_temperature": temperature,
        "thread_id": session_id,
        "session_id": session_id,
    }

    if azure_key and azure_endpoint:
        meta.update(
            {
                "provider": "azure_openai",
                "ls_provider": "azure",
                "ls_model_name": azure_deployment,
            }
        )
        return AzureChatOpenAI(
            azure_deployment=azure_deployment,
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_version,
            temperature=temperature,
            callbacks=tracers,
            tags=[agent_type, "multi_agent_travel"],
            metadata=meta,
        )
    elif openai_key:
        # Fallback to OpenAI
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        meta.update(
            {
                "provider": "openai",
                "ls_provider": "openai",
                "ls_model_name": model,
            }
        )
        return ChatOpenAI(
            model=model,
            api_key=openai_key,
            temperature=temperature,
            callbacks=tracers,
            tags=[agent_type, "multi_agent_travel"],
            metadata=meta,
        )
    else:
        raise ValueError(
            "No LLM configuration found. Set Azure OpenAI (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT) "
            "or OpenAI (OPENAI_API_KEY)."
        )


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def _safe_float_extract(text: str) -> Optional[float]:
    try:
        digits = "".join(ch for ch in text if ch.isdigit() or ch in ".")
        if digits:
            return float(digits)
    except Exception:
        return None
    return None


@tool
def extract_trip_details(user_request: str) -> str:
    """
    Extract structured trip details from a free-form user request.

    Returns JSON string with:
    destination, origin, travelers, duration (days), trip_type, budget, preferences
    """
    request_lower = user_request.lower()
    details = {
        "destination": None,
        "origin": None,
        "travelers": 2,
        "duration": 7,
        "trip_type": "leisure",
        "budget": None,
        "preferences": [],
    }

    # Destinations (simplified list)
    known_destinations = [
        "london",
        "paris",
        "new york",
        "tokyo",
        "rome",
        "sydney",
        "dubai",
        "orlando",
        "barcelona",
    ]
    for d in known_destinations:
        if d in request_lower:
            details["destination"] = d.title()
            break

    # Origin heuristic (look for "from <City>")
    if " from " in request_lower:
        snippet = request_lower.split(" from ")[1][:40]
        origin_tokens = snippet.split()
        if origin_tokens:
            candidate = origin_tokens[0].strip(",.")
            if candidate and candidate.isalpha():
                details["origin"] = candidate.title()

    # Travelers
    for token in request_lower.split():
        if token.isdigit():
            n = int(token)
            if 1 <= n <= 12:
                details["travelers"] = n
                break

    # Budget
    if "$" in request_lower:
        try:
            after = request_lower.split("$", 1)[1].split()[0]
            amount = _safe_float_extract(after)
            if amount:
                details["budget"] = amount
        except Exception:
            pass

    # Trip type inference
    if any(k in request_lower for k in ["business", "conference", "work"]):
        details["trip_type"] = "business"
    elif any(k in request_lower for k in ["family", "kids", "children"]):
        details["trip_type"] = "family"
    elif any(k in request_lower for k in ["romantic", "anniversary", "honeymoon"]):
        details["trip_type"] = "romantic"
    elif any(k in request_lower for k in ["adventure", "hiking", "extreme"]):
        details["trip_type"] = "adventure"

    # Preferences
    pref_map = {
        "luxury": "luxury_accommodations",
        "museum": "cultural_activities",
        "museums": "cultural_activities",
        "art": "cultural_activities",
        "food": "culinary_experiences",
        "dining": "culinary_experiences",
        "adventure": "adventure_activities",
        "hiking": "adventure_activities",
        "family": "family_friendly",
        "romantic": "romantic_experiences",
        "park": "parks_and_nature",
        "sports": "sports_events",
    }
    for k, tag in pref_map.items():
        if k in request_lower and tag not in details["preferences"]:
            details["preferences"].append(tag)

    # Duration (heuristic)
    if "week" in request_lower and "weeks" not in request_lower:
        details["duration"] = 7
    if "2 weeks" in request_lower or "two weeks" in request_lower:
        details["duration"] = 14
    if "3 days" in request_lower:
        details["duration"] = 3
    if "10 days" in request_lower:
        details["duration"] = 10

    # Ensure some defaults
    if not details["destination"]:
        details["destination"] = "London"
    if not details["origin"]:
        details["origin"] = "Seattle"

    return json.dumps(details)


@tool
def search_flights(
    origin: str, destination: str, departure_date: str, travelers: int = 2
) -> str:
    """
    Search for flights (simulated realistic data).
    Returns a JSON string with a list of flight dicts.
    """
    try:
        airlines_international = [
            "British Airways",
            "Virgin Atlantic",
            "United",
            "Delta",
            "American",
            "Lufthansa",
            "Air France",
        ]
        airlines_domestic = ["United", "Delta", "American", "Southwest", "JetBlue"]

        is_international = any(
            kw in destination.lower()
            for kw in ["london", "paris", "tokyo", "rome", "dubai", "sydney"]
        )

        airlines = airlines_international if is_international else airlines_domestic

        flights = []
        for _ in range(4):
            base_price = (
                random.randint(550, 1400)
                if is_international
                else random.randint(120, 550)
            )
            duration_hours = (
                random.randint(7, 12) if is_international else random.randint(1, 6)
            )
            flight = {
                "flight_id": f"FL{random.randint(1000,9999)}",
                "airline": random.choice(airlines),
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "departure_time": f"{random.randint(5, 21):02d}:{random.choice(['00','30'])}",
                "arrival_time": f"{(random.randint(6, 23))%24:02d}:{random.choice(['05','35'])}",
                "duration_hours": duration_hours,
                "price_per_person": base_price,
                "total_price": base_price * travelers,
                "seats_left": random.randint(4, 60),
            }
            flights.append(flight)

        return json.dumps({"flights": flights})
    except Exception as e:
        return json.dumps({"error": f"flight_search_failed: {e}"})


@tool
def search_hotels(
    destination: str, check_in: str, check_out: str, travelers: int = 2
) -> str:
    """
    Search for hotels (simulated).
    Returns JSON string with 'hotels': [...]
    """
    try:
        upscale = [
            "The Grand Palace",
            "Imperial Suites",
            "Regency Plaza",
            "Royal Crescent",
            "Skyline Retreat",
        ]
        mid = [
            "Central Comfort",
            "Cityscape Inn",
            "Traveler's Hub",
            "Urban Stay",
            "Metro Lodge",
        ]
        economy = [
            "Budget Stay",
            "Econo Lodge",
            "Value Inn",
            "Savings Suites",
            "Hostel Central",
        ]

        all_hotels = upscale + mid + economy
        hotels = []
        for _ in range(5):
            name = random.choice(all_hotels)
            price = random.randint(90, 750)
            rating = round(random.uniform(3.8, 5.0), 1)
            amenities_pool = [
                "WiFi",
                "Breakfast",
                "Gym",
                "Pool",
                "Spa",
                "Concierge",
                "Airport Shuttle",
                "Restaurant",
                "Bar",
            ]
            hotel = {
                "hotel_id": f"HT{random.randint(1000,9999)}",
                "name": name,
                "destination": destination,
                "check_in": check_in,
                "check_out": check_out,
                "price_per_night": price,
                "rating": rating,
                "amenities": random.sample(
                    amenities_pool, k=random.randint(4, min(8, len(amenities_pool)))
                ),
                "rooms_left": random.randint(2, 25),
                "neighborhood": random.choice(
                    ["Central", "Riverside", "Historic Quarter", "Financial District"]
                ),
            }
            hotels.append(hotel)
        return json.dumps({"hotels": hotels})
    except Exception as e:
        return json.dumps({"error": f"hotel_search_failed: {e}"})


@tool
def search_activities(destination: str, activity_type: Optional[str] = None) -> str:
    """
    Search for activities (simulated).
    Returns JSON with 'activities': [...]
    """
    try:
        base_catalog = {
            "cultural": [
                "Museum Tour",
                "Historic Walking Tour",
                "Art Gallery Visit",
                "Cultural Performance",
            ],
            "adventure": [
                "Guided Hike",
                "Kayaking Excursion",
                "Rock Climbing Session",
                "Zipline Adventure",
            ],
            "culinary": [
                "Food Market Tour",
                "Wine Tasting",
                "Cooking Class",
                "Chef's Table Experience",
            ],
            "family": [
                "Interactive Science Center",
                "Theme Park",
                "Aquarium",
                "Zoo Visit",
            ],
            "romantic": [
                "Sunset Cruise",
                "Fine Dining Experience",
                "Couples Spa",
                "Scenic Overlook Picnic",
            ],
        }

        chosen_categories = (
            [activity_type] if activity_type and activity_type in base_catalog else list(base_catalog.keys())
        )

        acts = []
        for cat in chosen_categories:
            for item in random.sample(base_catalog[cat], k=min(2, len(base_catalog[cat]))):
                acts.append(
                    {
                        "activity_id": f"AC{random.randint(1000,9999)}",
                        "category": cat,
                        "name": item,
                        "destination": destination,
                        "estimated_cost": random.randint(20, 180),
                        "duration_hours": random.choice([2, 3, 4, 6]),
                        "indoor": random.choice([True, False]),
                    }
                )

        return json.dumps({"activities": acts})
    except Exception as e:
        return json.dumps({"error": f"activity_search_failed: {e}"})


@tool
def get_weather_forecast(location: str, date: str) -> str:
    """
    Returns simple weather forecast JSON for a given location and date.
    """
    try:
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear", "Showers"]
        high = random.randint(8, 30)
        low = high - random.randint(3, 10)
        data = {
            "location": location,
            "date": date,
            "condition": random.choice(conditions),
            "temp_high_c": high,
            "temp_low_c": low,
            "precip_chance_pct": random.randint(0, 80),
        }
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": f"weather_failed: {e}"})


@tool
def analyze_budget(
    flight_cost: float,
    hotel_cost_per_night: float,
    nights: int,
    activity_costs: List[float],
    travelers: int,
    destination_currency: str = "USD",
) -> str:
    """
    Aggregate a budget breakdown.
    Returns JSON with detailed cost structure & optimization tips.
    """
    try:
        subtotal_flights = flight_cost
        subtotal_hotel = hotel_cost_per_night * nights
        subtotal_activities = sum(activity_costs)
        meals = 60 * travelers * nights  # heuristic
        misc = 0.1 * (subtotal_flights + subtotal_hotel + subtotal_activities)
        total = (
            subtotal_flights + subtotal_hotel + subtotal_activities + meals + misc
        )

        tips = []
        if subtotal_hotel / nights > 300:
            tips.append("Consider mid-range hotels for savings.")
        if subtotal_activities / max(1, travelers) > 400:
            tips.append("Choose a mix of free/low-cost cultural sites.")
        if meals / travelers > 70 * nights:
            tips.append("Plan some self-catered meals.")

        result = {
            "currency": destination_currency,
            "nights": nights,
            "travelers": travelers,
            "flights": subtotal_flights,
            "hotel": subtotal_hotel,
            "activities": subtotal_activities,
            "meals": meals,
            "miscellaneous": misc,
            "total": total,
            "per_person": total / max(1, travelers),
            "tips": tips,
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"budget_failed: {e}"})


# ---------------------------------------------------------------------------
# Graph Node Implementations
# ---------------------------------------------------------------------------
def coordinator_node(state: TravelPlanState) -> TravelPlanState:
    """Extract trip details and initialize next step."""
    system_msg = (
        "You are the Coordinator. Understand user needs, fill missing trip details, "
        "and set up the workflow."
    )
    llm = create_llm(
        "coordinator", system_instructions=system_msg, session_id=state["session_id"]
    )
    messages = [SystemMessage(content=system_msg)] + state["messages"]

    # Quick pass to see if we extracted details yet
    if not state.get("destination") or not state.get("departure_date"):
        # Use the extract_trip_details tool directly (non-agent call)
        extraction_raw = extract_trip_details.invoke({"user_request": state["user_request"]})
        try:
            details = json.loads(extraction_raw)
        except Exception:
            details = {}

        state["destination"] = (
            details.get("destination") or state.get("destination") or "London"
        )
        state["origin"] = details.get("origin") or state.get("origin") or "Seattle"
        state["travelers"] = details.get("travelers") or state.get("travelers") or 2
        state["trip_type"] = details.get("trip_type") or state.get("trip_type") or "leisure"
        state["preferences"] = details.get("preferences") or state.get("preferences") or {}

        # Dates: choose next month if missing
        if not state.get("departure_date"):
            today = datetime.utcnow().date()
            dep = today + timedelta(days=30)
            duration = details.get("duration") or 7
            ret = dep + timedelta(days=duration)
            state["departure_date"] = dep.isoformat()
            state["return_date"] = ret.isoformat()

        # Budget
        if details.get("budget"):
            state["budget"] = details["budget"]

    response = llm.invoke(
        messages + [HumanMessage(content="Acknowledge trip details and proceed.")],
        config=_trace_config_for_agent("coordinator", state["session_id"]),
    )
    state["messages"].append(response)

    state["current_step"] = "flight_search"
    state["completed_steps"].append("coordinator")
    return state


def flight_specialist_node(state: TravelPlanState) -> TravelPlanState:
    llm = create_llm(
        "flight_specialist",
        system_instructions="Find optimal flights and summarize key options.",
        session_id=state["session_id"],
    )

    # Invoke tools directly
    flights_raw = search_flights.invoke(
        {
            "origin": state["origin"],
            "destination": state["destination"],
            "departure_date": state["departure_date"],
            "travelers": state["travelers"],
        }
    )
    weather_raw = get_weather_forecast.invoke(
        {"location": state["destination"], "date": state["departure_date"]}
    )

    try:
        flights = json.loads(flights_raw).get("flights", [])
    except Exception:
        flights = []
    try:
        weather = json.loads(weather_raw)
    except Exception:
        weather = {}

    summary_prompt = f"""
Summarize these flight options and weather for planning:

Flights JSON:
{flights_raw}

Weather JSON:
{weather_raw}

Provide:
- 2-3 recommended flights
- Reasoning (timing, price, duration)
- Weather note
Return a concise structured summary.
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are a flight analyst."),
            HumanMessage(content=summary_prompt),
        ],
        config=_trace_config_for_agent("flight_specialist", state["session_id"]),
    )

    state["flight_results"] = flights
    state["weather_results"].append(weather)
    state["messages"].append(response)
    state["completed_steps"].append("flight_search")
    state["current_step"] = "hotel_search"
    state["agent_handoffs"].append(
        {
            "from": "coordinator",
            "to": "flight_specialist",
            "timestamp": datetime.utcnow().isoformat(),
            "task": "flight_search",
        }
    )
    return state


def hotel_specialist_node(state: TravelPlanState) -> TravelPlanState:
    llm = create_llm(
        "hotel_specialist",
        system_instructions="Recommend hotels aligned with trip type/preferences.",
        session_id=state["session_id"],
    )

    hotels_raw = search_hotels.invoke(
        {
            "destination": state["destination"],
            "check_in": state["departure_date"],
            "check_out": state["return_date"],
            "travelers": state["travelers"],
        }
    )
    try:
        hotels = json.loads(hotels_raw).get("hotels", [])
    except Exception:
        hotels = []

    pref_text = json.dumps(state.get("preferences") or {})
    prompt = f"""
Trip Type: {state.get('trip_type')}
Preferences: {pref_text}

Hotels JSON:
{hotels_raw}

Select 3 recommended hotels with justification (location, price, amenities).
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are a hotel specialist."),
            HumanMessage(content=prompt),
        ],
        config=_trace_config_for_agent("hotel_specialist", state["session_id"]),
    )

    state["hotel_results"] = hotels
    state["messages"].append(response)
    state["completed_steps"].append("hotel_search")
    state["current_step"] = "activity_search"
    state["agent_handoffs"].append(
        {
            "from": "flight_specialist",
            "to": "hotel_specialist",
            "timestamp": datetime.utcnow().isoformat(),
            "task": "hotel_search",
        }
    )
    return state


def activity_specialist_node(state: TravelPlanState) -> TravelPlanState:
    llm = create_llm(
        "activity_specialist",
        system_instructions="Curate activities & preliminary itinerary.",
        session_id=state["session_id"],
    )
    activities_raw = search_activities.invoke(
        {"destination": state["destination"], "activity_type": None}
    )
    try:
        acts = json.loads(activities_raw).get("activities", [])
    except Exception:
        acts = []

    weather_note = ""
    if state["weather_results"]:
        wr = state["weather_results"][-1]
        weather_note = f"Weather on departure date: {wr.get('condition')} (High {wr.get('temp_high_c')}C)."

    prompt = f"""
Destination: {state['destination']}
Trip Type: {state.get('trip_type')}
Travelers: {state['travelers']}
Dates: {state['departure_date']} to {state['return_date']}
Preferences: {json.dumps(state.get('preferences') or {})}
Weather: {weather_note}

Activities JSON:
{activities_raw}

Create:
1. Categorized activity recommendations
2. A sample day-by-day high-level itinerary (avoid exact times)
3. Notes for indoor vs outdoor options
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are an activity specialist."),
            HumanMessage(content=prompt),
        ],
        config=_trace_config_for_agent("activity_specialist", state["session_id"]),
    )

    state["activity_results"] = acts
    state["messages"].append(response)
    state["completed_steps"].append("activity_search")
    state["current_step"] = "budget_analysis"
    state["agent_handoffs"].append(
        {
            "from": "hotel_specialist",
            "to": "activity_specialist",
            "timestamp": datetime.utcnow().isoformat(),
            "task": "activity_recommendations",
        }
    )
    return state


def budget_analyst_node(state: TravelPlanState) -> TravelPlanState:
    llm = create_llm(
        "budget_analyst",
        system_instructions="Analyze and optimize budget.",
        session_id=state["session_id"],
    )

    # Roughly pick average flight + hotel + activities
    avg_flight_total = 0.0
    if state["flight_results"]:
        avg_flight_total = sum(f["total_price"] for f in state["flight_results"]) / max(
            1, len(state["flight_results"])
        )

    avg_hotel_ppn = 0.0
    if state["hotel_results"]:
        avg_hotel_ppn = sum(h["price_per_night"] for h in state["hotel_results"]) / max(
            1, len(state["hotel_results"])
        )

    activity_costs = [a["estimated_cost"] for a in state["activity_results"][:5]]

    # Nights calculation
    nights = 0
    try:
        d1 = datetime.fromisoformat(state["departure_date"])
        d2 = datetime.fromisoformat(state["return_date"])
        nights = max(1, (d2 - d1).days)
    except Exception:
        nights = 7

    budget_raw = analyze_budget.invoke(
        {
            "flight_cost": avg_flight_total,
            "hotel_cost_per_night": avg_hotel_ppn,
            "nights": nights,
            "activity_costs": activity_costs,
            "travelers": state["travelers"],
            "destination_currency": "USD",
        }
    )
    try:
        budget_json = json.loads(budget_raw)
    except Exception:
        budget_json = {"error": "parse_failure"}

    prompt = f"""
User Stated Budget: {state.get('budget')}
Budget Calculation JSON:
{budget_raw}

Provide:
1. Budget summary vs target
2. Category breakdown explanation
3. 3-5 optimization tips

If user stated budget is None, propose a reasonable target range.
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are a budget analyst."),
            HumanMessage(content=prompt),
        ],
        config=_trace_config_for_agent("budget_analyst", state["session_id"]),
    )

    state["budget_analysis"] = budget_raw
    state["messages"].append(response)
    state["completed_steps"].append("budget_analysis")
    state["current_step"] = "plan_synthesis"
    state["agent_handoffs"].append(
        {
            "from": "activity_specialist",
            "to": "budget_analyst",
            "timestamp": datetime.utcnow().isoformat(),
            "task": "budget_analysis",
        }
    )
    return state


def plan_synthesizer_node(state: TravelPlanState) -> TravelPlanState:
    llm = create_llm(
        "plan_synthesizer",
        system_instructions="Synthesize final comprehensive travel plan.",
        session_id=state["session_id"],
    )

    flights_excerpt = json.dumps(state["flight_results"][:3], indent=2)
    hotels_excerpt = json.dumps(state["hotel_results"][:3], indent=2)
    activities_excerpt = json.dumps(state["activity_results"][:6], indent=2)
    budget_summary = state["budget_analysis"]

    prompt = f"""
ORIGINAL REQUEST:
{state['user_request']}

TRIP DETAILS:
Destination: {state['destination']}
Origin: {state['origin']}
Dates: {state['departure_date']} to {state['return_date']}
Travelers: {state['travelers']}
Trip Type: {state.get('trip_type')}
Preferences: {json.dumps(state.get('preferences') or {})}

SPECIALIST DATA (JSON EXCERPTS):
Flights: {flights_excerpt}
Hotels: {hotels_excerpt}
Activities: {activities_excerpt}
Budget: {budget_summary}

Create FINAL TRAVEL PLAN with sections:
1. Executive Summary
2. Trip Overview
3. Recommended Flights
4. Lodging Recommendations
5. Day-by-Day High-Level Itinerary
6. Budget Overview & Optimization Tips
7. Key Notes (Weather, Safety, Local Transport)
8. Optional Enhancements / Upsells
Return a polished, structured plan.
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are the final plan synthesizer."),
            HumanMessage(content=prompt),
        ],
        config=_trace_config_for_agent("plan_synthesizer", state["session_id"]),
    )

    state["final_plan"] = response.content
    state["messages"].append(response)
    state["completed_steps"].append("plan_synthesis")
    state["current_step"] = "completed"
    state["agent_handoffs"].append(
        {
            "from": "budget_analyst",
            "to": "plan_synthesizer",
            "timestamp": datetime.utcnow().isoformat(),
            "task": "plan_synthesis",
        }
    )
    return state


# ---------------------------------------------------------------------------
# Conditional Routing
# ---------------------------------------------------------------------------
def should_continue(state: TravelPlanState) -> str:
    step = state.get("current_step", "start")
    if step == "start":
        return "coordinator"
    if step == "flight_search":
        return "flight_specialist"
    if step == "hotel_search":
        return "hotel_specialist"
    if step == "activity_search":
        return "activity_specialist"
    if step == "budget_analysis":
        return "budget_analyst"
    if step == "plan_synthesis":
        return "plan_synthesizer"
    if step == "completed":
        return END
    # Fallback
    logger.warning(f"Unknown step '{step}', ending.")
    return END


# ---------------------------------------------------------------------------
# Graph Factory
# ---------------------------------------------------------------------------
def create_travel_planning_graph():
    workflow = StateGraph(TravelPlanState)

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("flight_specialist", flight_specialist_node)
    workflow.add_node("hotel_specialist", hotel_specialist_node)
    workflow.add_node("activity_specialist", activity_specialist_node)
    workflow.add_node("budget_analyst", budget_analyst_node)
    workflow.add_node("plan_synthesizer", plan_synthesizer_node)

    workflow.add_conditional_edges(START, should_continue)
    workflow.add_conditional_edges("coordinator", should_continue)
    workflow.add_conditional_edges("flight_specialist", should_continue)
    workflow.add_conditional_edges("hotel_specialist", should_continue)
    workflow.add_conditional_edges("activity_specialist", should_continue)
    workflow.add_conditional_edges("budget_analyst", should_continue)
    workflow.add_conditional_edges("plan_synthesizer", should_continue)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app


# ---------------------------------------------------------------------------
# Sample Inputs
# ---------------------------------------------------------------------------
SAMPLE_INPUTS = {
    "business_trip": "I need a 3-day business trip to London from Seattle next month, near the financial district. Budget about $3000.",
    "family_vacation": "Plan a family vacation to Paris for 2 adults and 2 kids for 10 days in summer. Museums and parks. Around $8000.",
    "romantic_getaway": "Romantic anniversary trip to Rome for a week. Luxury hotel, fine dining, budget up to $6000.",
    "group_adventure": "Group trip to Tokyo for 6 friends for 2 weeks. Culture, food tours, and some adventure activities. Budget $4000 each.",
    "cultural_exploration": "Explore New York for 5 days for 2 people. Museums, theater, art galleries, culinary tours. About $3500.",
}


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    print("🌍 Multi-Agent Travel Planning (LangGraph) 🌍")
    print("=" * 72)
    print("Available sample keys:", ", ".join(SAMPLE_INPUTS.keys()))
    user_input = input("\nDescribe your travel plans (or enter a sample key): ").strip()

    if not user_input:
        user_input = "romantic_getaway"
        print(f"Using sample key: {user_input}")

    if user_input in SAMPLE_INPUTS:
        user_request = SAMPLE_INPUTS[user_input]
        print(f"\nSample request loaded:\n{user_request}\n")
    else:
        user_request = user_input

    # Basic LLM config validation
    if not (
        (os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"))
        or os.environ.get("OPENAI_API_KEY")
    ):
        print(
            "\n❌ No LLM credentials detected.\n"
            "Set Azure (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT) or OPENAI_API_KEY.\n"
        )
        return

    app = create_travel_planning_graph()
    session_id = str(uuid4())

    initial_state: TravelPlanState = TravelPlanState(
        messages=[HumanMessage(content=user_request)],
        user_request=user_request,
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
        agent_handoffs=[],
    )

    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 30}

    # Root span for entire planning (if OTEL available)
    root_span = None
    token = None
    if trace and SpanKind:
        try:
            tracer = trace.get_tracer("multi_agent_travel_planning")
            root_span = tracer.start_span(
                "travel_planning_session",
                kind=SpanKind.SERVER,
                attributes={
                    "session_id": session_id,
                    "user_request.length": len(user_request),
                    "system": "multi_agent_travel_planning",
                },
            )
            ctx = trace.set_span_in_context(root_span)
            token = context.attach(ctx)
        except Exception as e:
            logger.warning(f"Failed to start root span: {e}")

    print("\n🚀 Running multi-agent workflow...\n")
    final_state = None
    try:
        for event in app.stream(initial_state, config, stream_mode="values"):
            # event yields partial state per node; we can display progress
            step = event.get("current_step")
            completed = event.get("completed_steps", [])
            print(f"Progress: completed={completed} next={step}")

        # Fetch final (END) state
        final_state = app.get_state(config).values
    finally:
        if root_span:
            try:
                root_span.set_attribute(
                    "final_plan.generated", bool(final_state and final_state.get("final_plan"))
                )
                root_span.end()
            except Exception:
                pass
        if token:
            try:
                context.detach(token)
            except Exception:
                pass

    if not final_state:
        print("\n❌ No final state produced. Aborting.")
        return

    print("\n🎊 Travel Planning Completed!")
    print("=" * 72)
    print(final_state.get("final_plan") or "No final plan produced.")
    print("=" * 72)

    # Optional: show agent handoffs
    if final_state.get("agent_handoffs"):
        print("\n🔄 Agent Handoffs:")
        for h in final_state["agent_handoffs"]:
            print(
                f"- {h['from']} -> {h['to']} @ {h['timestamp']} ({h['task']})"
            )

    print("\n✅ Done. Have a great trip!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user.")
    except Exception as e:
        logger.exception("Fatal error running travel planner.")
        print(f"\n❌ Fatal error: {e}")
        print(
            "\nTroubleshooting:\n"
            "1. Check LLM credentials (Azure or OpenAI)\n"
            "2. Install dependencies: pip install langgraph langchain-openai langchain_azure_ai\n"
            "3. (Optional) Set tracing variables\n"
        )