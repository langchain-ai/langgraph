"""
Single-file Azure LangGraph travel planning sample with tracing.

Run with:
    python examples/travel_planning_system/travel_planning_system.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()


try:
    from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer
except ImportError:  # pragma: no cover - optional dependency
    AzureAIOpenTelemetryTracer = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError:  # pragma: no cover
    trace = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    OTLPSpanExporter = None  # type: ignore


DEFAULT_PROMPT = (
    "Plan a five-day culinary getaway to Lisbon for two adults in September "
    "with a budget around $4,000. We are flying from New York, enjoy boutique "
    "hotels, and want at least one day trip outside the city."
)

DISPLAY_FIELDS = {
    "analyze_requirements": "requirements_summary",
    "plan_flights": "flight_plan",
    "plan_stays": "stay_plan",
    "plan_activities": "activity_plan",
    "build_budget": "budget_summary",
    "author_itinerary": "final_plan",
}

_TRACING_CACHE: Optional[List[BaseCallbackHandler]] = None
_OTLP_CONFIGURED = False


class TravelPlannerState(TypedDict, total=False):
    """Shared state carried across LangGraph nodes."""

    user_input: str
    session_id: str
    preferences: Dict[str, Any]
    requirements_summary: str
    flight_plan: str
    stay_plan: str
    activity_plan: str
    budget_summary: str
    final_plan: str


def _build_tracing_callbacks(service_name: str) -> List[BaseCallbackHandler]:
    """Create tracing callbacks for Azure Application Insights / OTLP."""
    callbacks: List[BaseCallbackHandler] = []

    connection_string = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")
    if connection_string:
        if AzureAIOpenTelemetryTracer:
            tracer_name = os.getenv("TRAVEL_TRACER_NAME", "travel-planning-system")
            tracer_id = os.getenv("TRAVEL_TRACER_ID", tracer_name)
            callbacks.append(
                AzureAIOpenTelemetryTracer(
                    connection_string=connection_string,
                    enable_content_recording=True,
                    redact=False,
                    include_legacy_keys=True,
                    provider_name="langchain-azure-ai",
                    name=tracer_name,
                    id=tracer_id,
                )
            )
            print(
                "🔭 Azure Application Insights tracing enabled via AzureAIOpenTelemetryTracer."
            )
        else:  # pragma: no cover - dependency missing
            print(
                "⚠️ APPLICATION_INSIGHTS_CONNECTION_STRING is set but "
                "`langchain-azure-ai` is not installed; Azure tracing is disabled."
            )
    else:
        print("ℹ️ APPLICATION_INSIGHTS_CONNECTION_STRING not set. Azure tracing disabled.")

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if (
        endpoint
        and trace is not None
        and TracerProvider is not None
        and BatchSpanProcessor is not None
        and OTLPSpanExporter is not None
    ):
        _configure_otlp(service_name=service_name, endpoint=endpoint)
        print(f"📡 OTLP exporter configured for endpoint: {endpoint}")

    return callbacks


def _configure_otlp(*, service_name: str, endpoint: str) -> None:
    """Configure a minimal OTLP exporter once per process."""
    global _OTLP_CONFIGURED
    if _OTLP_CONFIGURED or trace is None or TracerProvider is None:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _OTLP_CONFIGURED = True


def _callbacks() -> Optional[List[BaseCallbackHandler]]:
    """Return tracing callbacks or None to disable callbacks."""
    global _TRACING_CACHE
    if _TRACING_CACHE is None:
        _TRACING_CACHE = _build_tracing_callbacks(service_name="travel-planning-system")
    return _TRACING_CACHE or None


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(
            f"Environment variable '{var_name}' is required to run this sample. "
            "See the README for instructions."
        )
    return value


def _create_agent_llm(agent_name: str, *, temperature: float = 0.3) -> AzureChatOpenAI:
    """Create an AzureChatOpenAI instance configured for a specific agent."""
    deployment = _require_env("AZURE_OPENAI_DEPLOYMENT")
    endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
    api_key = _require_env("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=temperature,
        callbacks=_callbacks(),
        tags=["travel_planning", agent_name],
        metadata={"agent": agent_name, "system": "travel_planning", "role": agent_name},
    )


ANALYST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a travel requirements analyst. Extract what the traveler "
                "is asking for and return a compact JSON object with these keys: "
                "destination (string or null), origin (string or null), "
                "travelers (integer or null), departure_date (string or null), "
                "return_date (string or null), budget (string or null), "
                "trip_style (string or null), must_do (list of strings, can be empty). "
                "Only output valid JSON with those fields."
            ),
        ),
        ("human", "{user_input}"),
    ]
)

FLIGHT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a flight specialist. Recommend 2-3 flight strategies that "
                "match the travel request. When curated flight data is provided use "
                "it as authoritative. Include airline, departure/arrival airports, "
                "approximate duration, cabin suggestion, and estimated price. "
                "Close with practical guidance."
            ),
        ),
        (
            "human",
            (
                "Traveler request:\n{user_input}\n\n"
                "Normalized preferences:\n{preference_json}\n\n"
                "Curated flight data:\n{curated_data}"
            ),
        ),
    ]
)

STAY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a lodging specialist. Suggest 2-3 places to stay that fit "
                "the traveler's style and budget. Prefer curated stay data when "
                "available. Report each option with neighborhood, vibe, nightly "
                "estimate, and why it fits."
            ),
        ),
        (
            "human",
            (
                "Traveler request:\n{user_input}\n\n"
                "Normalized preferences:\n{preference_json}\n\n"
                "Curated lodging data:\n{curated_data}"
            ),
        ),
    ]
)

ACTIVITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You curate memorable activities. Craft a balanced mix of signature "
                "experiences, local food, and downtime aligned with the traveler's "
                "style. Use curated options when present or trusted travel knowledge."
            ),
        ),
        (
            "human",
            (
                "Traveler request:\n{user_input}\n\n"
                "Normalized preferences:\n{preference_json}\n\n"
                "Curated activity data:\n{curated_data}"
            ),
        ),
    ]
)

BUDGET_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a travel budget analyst. Combine the specialist findings "
                "into an estimated budget. Break down costs by category, include a "
                "total, and suggest one savings lever and one splurge idea."
            ),
        ),
        (
            "human",
            (
                "Traveler request:\n{user_input}\n\n"
                "Normalized preferences:\n{preference_json}\n\n"
                "Flight options:\n{flight_plan}\n\n"
                "Stay options:\n{stay_plan}\n\n"
                "Activity ideas:\n{activity_plan}"
            ),
        ),
    ]
)

ITINERARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are the lead travel planner. Combine all specialist outputs "
                "into a cohesive itinerary with Markdown headings. Include sections "
                "for Trip Overview, Flight Plan, Lodging, Experiences, Budget "
                "Snapshot, and Next Steps."
            ),
        ),
        (
            "human",
            (
                "Original request:\n{user_input}\n\n"
                "Normalized preferences:\n{preference_json}\n\n"
                "Requirements summary:\n{requirements_summary}\n\n"
                "Flight options:\n{flight_plan}\n\n"
                "Stay options:\n{stay_plan}\n\n"
                "Activity ideas:\n{activity_plan}\n\n"
                "Budget summary:\n{budget_summary}"
            ),
        ),
    ]
)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _safe_parse_json(raw: str) -> Dict[str, Any]:
    candidate = _strip_code_fences(raw)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {"notes": raw}
    except json.JSONDecodeError:
        return {"notes": raw}


def _summarize_preferences(preferences: Dict[str, Any]) -> str:
    parts = []
    destination = preferences.get("destination")
    if destination:
        parts.append(f"- Destination: {destination}")
    origin = preferences.get("origin")
    if origin:
        parts.append(f"- Origin: {origin}")
    travelers = preferences.get("travelers")
    if travelers:
        parts.append(f"- Travelers: {travelers}")
    departure = preferences.get("departure_date")
    if departure:
        parts.append(f"- Departure date: {departure}")
    return_date = preferences.get("return_date")
    if return_date:
        parts.append(f"- Return date: {return_date}")
    budget = preferences.get("budget")
    if budget:
        parts.append(f"- Budget guidance: {budget}")
    trip_style = preferences.get("trip_style")
    if trip_style:
        parts.append(f"- Trip style: {trip_style}")
    must_do = preferences.get("must_do") or []
    if isinstance(must_do, list) and must_do:
        bullet_items = "\n  ".join(f"- {item}" for item in must_do)
        parts.append(f"- Must do:\n  {bullet_items}")
    if not parts:
        return "No structured preferences detected."
    return "\n".join(parts)


def _curated_section(preferences: Dict[str, Any], key: str) -> str:
    destination = (preferences.get("destination") or "").lower()
    if not destination:
        return "No curated data because destination was not detected."
    options = CURATED_DATA.get(destination, {}).get(key)
    if not options:
        return f"No curated data for '{destination}'."
    return json.dumps(options, indent=2)


def analyze_requirements(state: TravelPlannerState) -> Dict[str, Any]:
    llm = _create_agent_llm("requirements_analyst", temperature=0.0)
    result = llm.invoke(ANALYST_PROMPT.format_messages(user_input=state["user_input"]))
    if not isinstance(result, AIMessage):
        raise RuntimeError("Unexpected response type from requirements analyst.")
    preferences = _safe_parse_json(result.content)
    summary = _summarize_preferences(preferences)
    return {"preferences": preferences, "requirements_summary": summary}


def plan_flights(state: TravelPlannerState) -> Dict[str, Any]:
    preferences = state["preferences"]
    llm = _create_agent_llm("flight_specialist", temperature=0.2)
    rendered = FLIGHT_PROMPT.format_messages(
        user_input=state["user_input"],
        preference_json=json.dumps(preferences, indent=2),
        curated_data=_curated_section(preferences, "flights"),
    )
    result = llm.invoke(rendered)
    if not isinstance(result, AIMessage):
        raise RuntimeError("Unexpected response type from flight specialist.")
    return {"flight_plan": result.content}


def plan_stays(state: TravelPlannerState) -> Dict[str, Any]:
    preferences = state["preferences"]
    llm = _create_agent_llm("stay_specialist", temperature=0.4)
    rendered = STAY_PROMPT.format_messages(
        user_input=state["user_input"],
        preference_json=json.dumps(preferences, indent=2),
        curated_data=_curated_section(preferences, "stays"),
    )
    result = llm.invoke(rendered)
    if not isinstance(result, AIMessage):
        raise RuntimeError("Unexpected response type from stay specialist.")
    return {"stay_plan": result.content}


def plan_activities(state: TravelPlannerState) -> Dict[str, Any]:
    preferences = state["preferences"]
    llm = _create_agent_llm("activities_curator", temperature=0.5)
    rendered = ACTIVITY_PROMPT.format_messages(
        user_input=state["user_input"],
        preference_json=json.dumps(preferences, indent=2),
        curated_data=_curated_section(preferences, "activities"),
    )
    result = llm.invoke(rendered)
    if not isinstance(result, AIMessage):
        raise RuntimeError("Unexpected response type from activities curator.")
    return {"activity_plan": result.content}


def build_budget(state: TravelPlannerState) -> Dict[str, Any]:
    preferences = state["preferences"]
    llm = _create_agent_llm("budget_analyst", temperature=0.2)
    rendered = BUDGET_PROMPT.format_messages(
        user_input=state["user_input"],
        preference_json=json.dumps(preferences, indent=2),
        flight_plan=state["flight_plan"],
        stay_plan=state["stay_plan"],
        activity_plan=state["activity_plan"],
    )
    result = llm.invoke(rendered)
    if not isinstance(result, AIMessage):
        raise RuntimeError("Unexpected response type from budget analyst.")
    return {"budget_summary": result.content}


def author_itinerary(state: TravelPlannerState) -> Dict[str, Any]:
    preferences = state["preferences"]
    llm = _create_agent_llm("itinerary_author", temperature=0.25)
    rendered = ITINERARY_PROMPT.format_messages(
        user_input=state["user_input"],
        preference_json=json.dumps(preferences, indent=2),
        requirements_summary=state["requirements_summary"],
        flight_plan=state["flight_plan"],
        stay_plan=state["stay_plan"],
        activity_plan=state["activity_plan"],
        budget_summary=state["budget_summary"],
    )
    result = llm.invoke(rendered)
    if not isinstance(result, AIMessage):
        raise RuntimeError("Unexpected response type from itinerary author.")
    return {"final_plan": result.content}


def build_travel_planner_graph() -> StateGraph:
    graph = StateGraph(TravelPlannerState)
    graph.add_node("analyze_requirements", analyze_requirements)
    graph.add_node("plan_flights", plan_flights)
    graph.add_node("plan_stays", plan_stays)
    graph.add_node("plan_activities", plan_activities)
    graph.add_node("build_budget", build_budget)
    graph.add_node("author_itinerary", author_itinerary)

    graph.set_entry_point("analyze_requirements")
    graph.add_edge("analyze_requirements", "plan_flights")
    graph.add_edge("plan_flights", "plan_stays")
    graph.add_edge("plan_stays", "plan_activities")
    graph.add_edge("plan_activities", "build_budget")
    graph.add_edge("build_budget", "author_itinerary")
    graph.add_edge("author_itinerary", END)
    return graph.compile()


def _print_header() -> None:
    print("\n🌍  LangGraph Multi-Agent Travel Planner (Azure edition)\n")
    print("This single-file sample runs specialist Azure OpenAI agents and streams their outputs.")
    print("Tracing is automatically enabled when APPLICATION_INSIGHTS_CONNECTION_STRING is set.\n")


def _print_step(title: str, content: str) -> None:
    bar = "=" * 80
    print(f"{bar}\n{title}\n{bar}\n{content.strip()}\n")


def run_planner(user_request: str, *, session_id: str | None = None) -> Dict[str, Any]:
    graph = build_travel_planner_graph()
    session = session_id or str(uuid4())
    state_snapshot: Dict[str, Any] = {
        "user_input": user_request,
        "session_id": session,
    }

    for step in graph.stream(
        state_snapshot,
        config={
            "configurable": {"thread_id": session},
        },
    ):
        for node, update in step.items():
            if not isinstance(update, dict):
                continue
            state_snapshot.update(update)
            if node == "__end__":
                continue
            pretty_title = node.replace("_", " ").title()
            key = DISPLAY_FIELDS.get(node)
            content = update.get(key) if key else update
            if isinstance(content, (dict, list)):
                content = json.dumps(content, indent=2)
            content = str(content or "").strip()
            _print_step(pretty_title, content)

    plan = state_snapshot.get("final_plan", "")
    if not plan:
        raise RuntimeError("Planner did not produce a final plan. Check Azure credentials.")

    _print_step("Traveler-Friendly Plan", plan)
    return state_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph Azure travel planner")
    parser.add_argument(
        "--prompt",
        help="Optional travel brief to plan. If omitted an example is used.",
    )
    args = parser.parse_args()

    _print_header()

    user_prompt = args.prompt
    if not user_prompt:
        user_prompt = input("Describe your trip (press Enter to use the sample prompt): ").strip()
        if not user_prompt:
            print("\nUsing sample prompt:")
            print(DEFAULT_PROMPT)
            user_prompt = DEFAULT_PROMPT

    try:
        run_planner(user_prompt)
    except RuntimeError as exc:
        print(f"\n❌ Planner failed: {exc}")
        print(
            "\nTroubleshooting:\n"
            "- Ensure Azure OpenAI variables are set in .env\n"
            "- Verify the deployment name matches your Azure resource\n"
            "- Check network connectivity to the Azure endpoint\n"
        )


CURATED_DATA: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "paris": {
        "flights": [
            {
                "airline": "Air France",
                "from": "JFK",
                "to": "CDG",
                "duration": "7h 30m",
                "price_estimate_usd": 950,
                "notes": "Non-stop, includes checked bag, best for premium economy upgrades.",
            },
            {
                "airline": "Delta",
                "from": "ATL",
                "to": "CDG",
                "duration": "8h 5m",
                "price_estimate_usd": 870,
                "notes": "Evening departure, easy SkyTeam mileage upgrades.",
            },
            {
                "airline": "United",
                "from": "EWR",
                "to": "CDG",
                "duration": "7h 25m",
                "price_estimate_usd": 910,
                "notes": "Morning arrival, great for maximizing first-day sightseeing.",
            },
        ],
        "stays": [
            {
                "name": "Hotel des Arts Montmartre",
                "neighborhood": "Montmartre",
                "nightly_estimate_usd": 240,
                "vibe": "Boutique bohemian with rooftop views.",
            },
            {
                "name": "Maison Breguet",
                "neighborhood": "Le Marais",
                "nightly_estimate_usd": 310,
                "vibe": "Design-forward, walkable to cafes and galleries.",
            },
            {
                "name": "Pullman Paris Tour Eiffel",
                "neighborhood": "7th arrondissement",
                "nightly_estimate_usd": 360,
                "vibe": "Modern rooms with Eiffel Tower balcony options.",
            },
        ],
        "activities": [
            {
                "title": "After-hours Louvre highlights tour",
                "ideal_day": "Arrival evening",
                "estimate_usd": 110,
            },
            {
                "title": "Day trip to Champagne with cellar tastings",
                "ideal_day": "Day 3",
                "estimate_usd": 220,
            },
            {
                "title": "Seine dinner cruise with live jazz",
                "ideal_day": "Final night",
                "estimate_usd": 150,
            },
        ],
    },
    "tokyo": {
        "flights": [
            {
                "airline": "Japan Airlines",
                "from": "SFO",
                "to": "HND",
                "duration": "11h 15m",
                "price_estimate_usd": 1180,
                "notes": "Direct, excellent premium economy, lands close to city center.",
            },
            {
                "airline": "ANA",
                "from": "LAX",
                "to": "NRT",
                "duration": "11h 30m",
                "price_estimate_usd": 1040,
                "notes": "Morning departure with Star Alliance mileage earning.",
            },
            {
                "airline": "United",
                "from": "ORD",
                "to": "HND",
                "duration": "13h 0m",
                "price_estimate_usd": 980,
                "notes": "One-stop via San Francisco with shortest layover.",
            },
        ],
        "stays": [
            {
                "name": "Hotel Niwa Tokyo",
                "neighborhood": "Chiyoda",
                "nightly_estimate_usd": 210,
                "vibe": "Relaxed luxury with Japanese garden courtyard.",
            },
            {
                "name": "Trunk Hotel",
                "neighborhood": "Shibuya",
                "nightly_estimate_usd": 320,
                "vibe": "Trendy boutique with local artisan collaborations.",
            },
            {
                "name": "Park Hyatt Tokyo",
                "neighborhood": "Shinjuku",
                "nightly_estimate_usd": 520,
                "vibe": "Iconic skyline views from rooms and New York Grill.",
            },
        ],
        "activities": [
            {
                "title": "Sushi making class with market tour",
                "ideal_day": "Day 2 morning",
                "estimate_usd": 130,
            },
            {
                "title": "TeamLab Planets immersive art experience",
                "ideal_day": "Day 3 afternoon",
                "estimate_usd": 60,
            },
            {
                "title": "Day trip to Hakone with onsen visit",
                "ideal_day": "Day 4",
                "estimate_usd": 190,
            },
        ],
    },
    "lisbon": {
        "flights": [
            {
                "airline": "TAP Air Portugal",
                "from": "JFK",
                "to": "LIS",
                "duration": "6h 45m",
                "price_estimate_usd": 620,
                "notes": "Often cheapest nonstop; includes free stopover options.",
            },
            {
                "airline": "United",
                "from": "EWR",
                "to": "LIS",
                "duration": "6h 55m",
                "price_estimate_usd": 700,
                "notes": "Evening departure with Polaris upgrade availability.",
            },
            {
                "airline": "Iberia",
                "from": "ORD",
                "to": "LIS",
                "duration": "10h 20m",
                "price_estimate_usd": 640,
                "notes": "One-stop via Madrid with smooth Schengen connection.",
            },
        ],
        "stays": [
            {
                "name": "The Lumiares Hotel",
                "neighborhood": "Bairro Alto",
                "nightly_estimate_usd": 260,
                "vibe": "Apartment-style suites with rooftop bar and city views.",
            },
            {
                "name": "Memmo Alfama",
                "neighborhood": "Alfama",
                "nightly_estimate_usd": 280,
                "vibe": "Adults-only hideaway with infinity pool overlooking Tagus River.",
            },
            {
                "name": "The Vintage Lisbon",
                "neighborhood": "Avenida da Liberdade",
                "nightly_estimate_usd": 230,
                "vibe": "Mid-century flair, complimentary city discovery kits.",
            },
        ],
        "activities": [
            {
                "title": "Private tram tour through the historic hills",
                "ideal_day": "Arrival afternoon",
                "estimate_usd": 140,
            },
            {
                "title": "Day trip to Sintra with Quinta da Regaleira entry",
                "ideal_day": "Day 3",
                "estimate_usd": 95,
            },
            {
                "title": "Sunset sailing cruise on the Tagus River",
                "ideal_day": "Final evening",
                "estimate_usd": 85,
            },
        ],
    },
}


if __name__ == "__main__":
    main()
