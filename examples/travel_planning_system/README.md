# Enhanced Multi-Agent Travel Planning System

A sophisticated travel planning system built with LangGraph that uses multiple specialized AI agents to create comprehensive travel plans. The system integrates with real APIs for authentic data while providing intelligent fallbacks.

## 🌟 Features

### Multi-Agent Architecture
- **Coordinator Agent**: Manages overall planning process and extracts requirements
- **Flight Specialist**: Searches real flight data using Amadeus API
- **Hotel Specialist**: Finds accommodations using Foursquare Places API  
- **Activity Specialist**: Discovers attractions and events using multiple APIs
- **Budget Analyst**: Provides detailed cost analysis and optimization
- **Plan Synthesizer**: Creates comprehensive final travel plans

### Real Data Integration
- **Weather**: OpenWeatherMap API for accurate forecasts
- **Flights**: Amadeus API for real flight search and pricing
- **Hotels**: Foursquare Places API for accommodation discovery
- **Activities**: Foursquare + Eventbrite APIs for attractions and events
- **Currency**: ExchangeRate API for international trip budgeting
- **Fallback Data**: Realistic generated data when APIs unavailable

### Advanced Orchestration
- **LangGraph Workflow**: Sophisticated state management and agent coordination
- **Smart Handoffs**: Agents pass context and collaborate seamlessly  
- **Error Handling**: Graceful degradation with fallback data sources
- **Tracing Support**: Full observability with OpenTelemetry and Azure insights

## 🚀 Quick Start

### 1. Setup
```bash
# Clone and navigate to the travel planning system
cd examples/travel_planning_system

# Run the setup script
python setup.py
```

### 2. Configuration
Edit the `.env` file with your API keys:

```bash
# Required: LLM Configuration
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Optional: Real Data APIs (free tiers available)
OPENWEATHER_API_KEY=your_key_here
AMADEUS_API_KEY=your_key_here
AMADEUS_API_SECRET=your_secret_here
EXCHANGERATE_API_KEY=your_key_here
FOURSQUARE_API_KEY=your_key_here
EVENTBRITE_TOKEN=your_token_here
```

### 3. Run
```bash
python enhanced_multi_agent_system.py
```

## 📝 Sample Inputs

The system handles various complex travel scenarios with detailed requirements. Each sample demonstrates how comprehensive inputs trigger all six specialized agents and their tools:

### Business Trip
```
"I need to plan a comprehensive business trip to London, UK for 3 days from March 15-17, 2026, traveling from Seattle (SEA) with arrival by 9 AM London time on March 15th for an 11 AM board meeting and departure by 6 PM on March 17th to catch a connecting domestic flight. I prefer business class or premium economy flights with major airlines like British Airways, Virgin Atlantic, or Delta. For accommodations, I require a 4-star or 5-star hotel in the City of London financial district within walking distance of Bank or Monument stations, featuring business center facilities, high-speed WiFi, room service until midnight, concierge service, quiet rooms with workspace, express check-in/out, 24-hour fitness center, laundry services, and ideally a business lounge. My itinerary includes visiting the London Stock Exchange area on March 15th, Canary Wharf on March 16th morning, Covent Garden on March 16th evening, and a morning in Mayfair on March 17th before departure. My total budget is approximately $3,000 USD."
```

### Family Vacation
```
"Please plan a comprehensive family vacation to Paris, France for our family of four - 2 adults (ages 38 and 42) and 2 children (ages 8 and 12) - for 10 days from July 20-29, 2026, flying from Los Angeles (LAX) with preference for direct flights, using major carriers with good family services and option to select seats together. We need family-friendly accommodations that comfortably sleep four people - in the 6th or 7th arrondissement or near Latin Quarter with easy metro access to major attractions, reliable WiFi, air conditioning for July weather, and laundry facilities for our 10-day stay. Our family loves museums and cultural experiences with age-appropriate activities engaging both kids, Eiffel Tower, Louvre, Notre-Dame area, Versailles day trip, Luxembourg Gardens and Tuileries. Food is important - need kid-friendly options since our 8-year-old is picky, so we need family-catering restaurants, information about French foods kids enjoy. Our budget is flexible at approximately $8,000 total."
```

### Romantic Getaway
```
"I'm planning a special romantic anniversary trip to Rome, Italy for my partner and me (both in our early 30s) to celebrate our 5th wedding anniversary from September 10-16, 2026, departing from Chicago (ORD). We want luxury accommodations - a 5-star boutique hotel or romantic bed & breakfast in the heart of Rome near Spanish Steps, Trastevere for romantic atmosphere, or close to Pantheon area, featuring historic character and charm with room views (city or charming courtyard), luxury amenities like marble bathroom, high-quality linens, room service for romantic in-room breakfasts, and ideally rooftop terrace or balcony for sunset aperitivos. Our romantic wishlist includes classic Rome experiences with elevated, intimate touches. Our budget is up to $6,000 for this special celebration, willing to splurge on unique experiences, exceptional dining, and luxury accommodations, and we need current weather forecasts for September in Rome, recommendations for romantic photography spots, tips for avoiding crowds."
```

### Group Adventure
```
"I'm organizing an epic group adventure trip to Tokyo, Japan for 6 close friends (ages 25-32, mix of 4 males and 2 females) for 2 weeks from October 5-18, 2024, departing from New York City (JFK) looking for best value flights while remaining comfortable for the long journey. We need accommodations for 6 people through creative solutions like multiple hotel rooms (preferably 3 double rooms close together), large vacation rental, or traditional ryokan experience for part of our stay, wanting to experience different Tokyo neighborhoods like Shibuya or Harajuku for modern energetic vibes, Asakusa for traditional culture, and maybe one night in a ryokan in nearby Hakone or Nikko for authentic Japanese experience. Our group is fascinated by Japanese culture wanting deep, authentic experiences including traditional tea ceremony, visiting ancient temples and shrines (Senso-ji, Meiji Shrine), experiencing traditional onsen, learning about samurai culture, visiting sumo training stable if possible, trying traditional crafts like pottery or calligraphy, and exploring traditional gardens like East Gardens of Imperial Palace, while simultaneously being excited about modern Japan including crazy energy of Shibuya crossing, gaming and anime culture in Akihabara, teamLab Borderless or similar digital art experience, Robot Restaurant or similar unique Tokyo entertainment, and exploring fashion and youth culture in Harajuku. Food is a HUGE priority as we're all serious foodies wanting full spectrum of Japanese cuisine including authentic sushi at Tsukiji Outer Market, traditional kaiseki dinner, best ramen shops (different styles in different neighborhoods), authentic yakitori, and if budget allows maybe splurge on Michelin-starred restaurant meal, plus taking cooking class together focusing on Japanese techniques, visiting sake brewery or comprehensive sake tasting, and exploring food markets and department store food courts. We're interested in adventure activities like hiking Mount Fuji if weather permits in October, visiting snow monkeys in Jigokudani if we can fit day trip, exploring Nikko for natural beauty and temples, or taking day trip to Kamakura for giant Buddha and hiking trails, plus experiencing Tokyo's unique nightlife including tiny bars in Golden Gai, multiple karaoke sessions, and nightlife districts like Roppongi or Shinjuku with recommendations for foreigner-welcoming bars where we can interact with locals. Each person has approximately $4,000 budget so our total group budget is around $24,000 covering flights, accommodations, local transportation including day trips, meals, activities, and shopping, wanting to experience efficient Japanese transportation including rides on different train types and maybe short shinkansen bullet train ride to nearby city. Please provide current weather forecasts for October in Tokyo, comprehensive navigation and transportation information for group of 6 including JR Pass recommendations, cultural etiquette and customs we should know, language basics for essential interactions, group activity recommendations perfect for 6 friends traveling together, information about shopping districts, Japanese gift ideas to bring home, and any seasonal activities or festivals happening during our October visit."
```

### Cultural Exploration
```
"My partner and I (both culture enthusiasts in our late 40s with graduate degrees in art history and literature) want to plan an intensive cultural exploration of New York City for 5 days from November 8-12, 2024, flying from Denver as serious cultural travelers who prefer depth over breadth, spending substantial time at fewer institutions rather than rushing through many attractions. We need accommodations in Manhattan within easy reach of the major cultural district - ideally Midtown near Museum Mile or Upper West Side near Lincoln Center - preferring boutique hotels with character, history, sophisticated decor, and quintessentially New York feel, ideally with good library or reading area for evening unwinding. Our primary focus is New York's world-class museums approached strategically with expert guidance: at Metropolitan Museum spending entire day with private or semi-private expert guide focusing on Egyptian collection, European paintings, and new modern wing; visiting Museum of Modern Art (MoMA) with emphasis on understanding modern art movements evolution with insider access or curator-led experience if possible; Guggenheim for current exhibitions and understanding Frank Lloyd Wright's architectural vision; Frick Collection for intimate setting and exceptional Old Masters; and if time permits, specialized museums like Morgan Library & Museum for literary manuscripts and medieval artifacts. Theater is essential with both Broadway production (preferably critically acclaimed rather than just popular) and off-Broadway show representing innovative contemporary theater based on November productions, willing to pay premium for excellent seats. We're serious food enthusiasts wanting New York's culinary culture at highest level including tasting menu at Michelin-starred restaurant, authentic ethnic cuisine in neighborhoods like Chinatown or Little Italy, and iconic New York food experiences like proper New York deli, classic steakhouse, or authentic New York pizza locals would recommend. For art galleries, we want to explore Chelsea's gallery district and SoHo ideally with guidance about current exhibitions and emerging artists to understand New York's contemporary art scene and cutting-edge work. Architecture is another passion requiring guided architectural walking tour covering different periods of New York's development from historic landmarks to modern skyscrapers, particularly Art Deco buildings, High Line's integration of urban planning and art, and understanding how architecture reflects cultural evolution. We're interested in literary New York including places associated with famous writers, historic literary venues, independent bookstores with character and knowledgeable staff, and perhaps literary walking tour beyond surface-level tourist information. Our budget is approximately $3,500 total for both of us covering flights, accommodation, meals, theater tickets, museum admissions, tours, and cultural activities, preferring spending more on fewer, higher-quality experiences rather than seeing everything, and we'd appreciate information about museum memberships or cultural passes providing value and special access, recommendations for cultural events or lectures during our visit, guidance on timing museum visits to avoid crowds while maximizing cultural immersion, and suggestions for sophisticated evening activities beyond theater like classical music performances at Lincoln Center, jazz clubs with authentic atmosphere, or cultural salons and intellectual gatherings accessible to visitors."
```

## 🔧 API Setup Instructions

### Free API Keys (No Credit Card Required)

#### 1. OpenWeatherMap (Weather Data)
- Visit: https://openweathermap.org/api
- Sign up for free account
- Get API key (1000 calls/day free)
- Add to `.env`: `OPENWEATHER_API_KEY=your_key`

#### 2. Amadeus (Flight Data)
- Visit: https://developers.amadeus.com
- Create free developer account
- Get API key and secret (2000 calls/month free)
- Add to `.env`: 
  ```
  AMADEUS_API_KEY=your_key
  AMADEUS_API_SECRET=your_secret
  ```

#### 3. ExchangeRate API (Currency Data)
- Visit: https://exchangerate-api.com
- Sign up for free account
- Get API key (1500 calls/month free)
- Add to `.env`: `EXCHANGERATE_API_KEY=your_key`

#### 4. Foursquare Places (Hotels & Activities)
- Visit: https://developer.foursquare.com
- Create developer account
- Get API key (100 calls/day free)
- Add to `.env`: `FOURSQUARE_API_KEY=your_key`

#### 5. Eventbrite (Events Data)
- Visit: https://www.eventbrite.com/platform/api
- Create account and app
- Get private token (free)
- Add to `.env`: `EVENTBRITE_TOKEN=your_token`

### LLM Configuration

#### Azure OpenAI (Recommended)
```bash
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_DEPLOYMENT_NAME=gpt-4
```

#### OpenAI (Alternative)
```bash
OPENAI_API_KEY=your_key
```

## 🏗️ Architecture

### LangGraph Workflow
```
Start → Coordinator → Flight Specialist → Hotel Specialist → Activity Specialist → Budget Analyst → Plan Synthesizer → End
```

### Runtime State Schema
The `TravelPlanState` (see `enhanced_multi_agent_system.py`) persists across nodes:
- messages: conversation history (LangChain message objects)
- user_request: original free‑form request
- destination / origin: city names or IATA-origin city
- budget: numeric (total trip budget – may be None if unspecified)
- travelers: integer (defaults 2)
- departure_date / return_date: ISO `YYYY-MM-DD` (auto-assigned if missing)
- trip_type: business | leisure | family | romantic | luxury (inferred)
- preferences: dict / structured tags (e.g. adventure_activities, luxury_accommodations)
- flight_results / hotel_results / activity_results: lists of tool or agent messages
- weather_results: list of weather lookups
- budget_analysis: generated analysis text / messages
- current_step: routing cursor for conditional edges
- completed_steps: ordered list of finished phases
- final_plan: synthesized comprehensive plan (string)
- session_id: UUID for checkpoint isolation
- agent_handoffs: audit log of inter-agent transitions

### Execution Flow (Happy Path)

1. Coordinator extracts & normalizes request details (may invoke `extract_trip_details` tool)
2. Flight Specialist (flights + optional weather) → stores flight options
3. Hotel Specialist chooses accommodation recommendations
4. Activity Specialist compiles itinerary options (uses weather & preference cues)
5. Budget Analyst aggregates estimated costs & optimization guidance
6. Plan Synthesizer produces structured final plan
7. Graph ends; `final_plan` available in state

If any step sets `current_step` unexpectedly, routing fallback returns `END`.

### Tools (Implemented / Placeholders)

The following tool functions are scaffolded in `enhanced_multi_agent_system.py`:

- search_flights
- search_hotels
- search_activities
- get_weather_forecast
- analyze_budget
- extract_trip_details (partially implemented; performs lightweight regex + keyword extraction)

At present several tool bodies are placeholders awaiting integration with concrete implementations from `data_sources.py` (e.g. `search_flights_sync`, `search_hotels_sync`). Fill each `try/except` block with calls to synchronous wrappers, serialize structured JSON (string return required by LangChain tool contract), and provide graceful fallbacks when external APIs are not configured.

### Tracing & Observability

The system optionally wires:

- Azure Application Insights (via `AzureOpenAITracingCallback`)
- OpenTelemetry exporter (`OpenTelemetryTracer`) emitting spans for: agent node execution, tool calls, LLM invocations

Enable by setting in `.env` (names may map through `config.py`):

```bash
APPLICATION_INSIGHTS_CONNECTION_STRING=...
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
OTEL_SERVICE_NAME=enhanced-travel-planner
```
During execution, each agent could emit spans once `_get_otel_tracer` helper is fully implemented (currently partially stubbed). To finalize:
1. Populate `_get_otel_tracer()` loop to return the cached `OpenTelemetryTracer` instance.
2. Wrap main graph invocation in a root span (already scaffolded in `main()` with TODO comments).
3. For each node, optionally create child spans annotating input parameters and summary of outputs (avoid full PII unless redaction configured).

### Completing the Remaining TODOs

Open `enhanced_multi_agent_system.py` and implement the following:

1. Import missing standard libs at top (`import logging, json, re`).
2. Fill in `_get_otel_tracer()` try/except (identify tracer by `isinstance(t, OpenTelemetryTracer)`).
3. Implement each tool body:
  * Call corresponding `*_sync` function from `data_sources` (e.g., `search_flights_sync`).
  * On success: return `json.dumps({"status": "ok", "data": results})`.
  * On failure: log error and return fallback structure with `status=error` and message.
4. In `extract_trip_details` loops, set fields when keywords match; add budget & preference extraction logic (currently placeholders).
5. In coordinator node, after tool response, parse tool JSON to update `destination`, `origin`, `travelers`, `trip_type`, `budget`, `preferences`.
6. In synthesis node, iterate messages (they may be LangChain message objects) and append `.content` safely.
7. Finish tracing root span creation and ensure closing in `finally` to flush exporters.

### Sample Output Structure (Excerpt)

```text
EXECUTIVE SUMMARY
Trip: 2 travelers, Seattle → London (Mar 15–17 2026)
Primary Goals: Board meeting, site visits (LSE, Canary Wharf), evening networking

FLIGHTS
- Option 1: SEA → LHR (BA52) Business Class ...
...

HOTELS (City / Bank District)
- Hotel A (5★) ...

ITINERARY (Condensed)
Day 1 AM: Arrival + Transfer; Midday: Board Meeting...
...

BUDGET SUMMARY
Flights: $X
Hotels: $Y (2 nights)
Per Diem / Meals: $Z
Total vs Budget: $Total / $Budget (Δ)
```

### Command Line Usage Patterns

Interactive (prompts for request if none provided):

```bash
python enhanced_multi_agent_system.py
```

Programmatic (embedding in another script):

```python
from enhanced_multi_agent_system import create_travel_planning_graph, TravelPlanState
from langchain_core.messages import HumanMessage

graph = create_travel_planning_graph()
state = TravelPlanState(
  messages=[HumanMessage(content="Plan a 5 day art trip to NYC in November for 2 people, budget $3500")],
  user_request="Plan a 5 day art trip to NYC in November for 2 people, budget $3500",
  destination=None, origin=None, budget=None, travelers=2,
  departure_date=None, return_date=None, trip_type=None, preferences={},
  flight_results=[], hotel_results=[], activity_results=[], weather_results=[],
  budget_analysis=None, current_step="start", completed_steps=[], final_plan=None,
  session_id="dev-session", agent_handoffs=[]
)
for event in graph.stream(state, {"configurable": {"thread_id": "dev-session"}}):
  pass
```

### Adding a New Agent (Example: Visa Specialist)

1. Extend `AgentType` enum and temperature map in `create_llm`.
2. Add fields to `TravelPlanState` if persistent outputs required (e.g. `visa_requirements`).
3. Implement node function (e.g. `visa_specialist_node`).
4. Insert conditional routing branch in `should_continue` and graph construction.
5. Provide tools (e.g. `lookup_visa_requirements`) referencing new or existing data sources.
6. Update README and tests.

### Testing Recommendations

- Unit style: create lightweight tests calling each tool with mocked/fallback responses.
- Integration: run the graph end‑to‑end with a deterministic seed or mocked LLM (LangChain Fake LLM) for CI stability.
- Snapshot: store golden plan outlines to detect structural regressions.

### Troubleshooting Quick Reference

| Symptom | Likely Cause | Resolution |
| ------- | ------------ | ---------- |
| ValueError: No LLM configuration found | Missing Azure/OpenAI keys | Populate `.env` then re-run `python setup.py` |
| Empty flight/hotel/activity results | Tool bodies still placeholders | Implement tool logic calling `data_sources` helpers |
| Tracing not emitting spans | `_get_otel_tracer` incomplete or env vars absent | Finish helper & set OTEL env vars |
| JSON decode errors in coordinator | Tool returned non-JSON string | Ensure tools return serialized JSON |
| Plan synthesizer missing sections | Upstream lists empty or not iterated | Confirm earlier nodes append `.content` |

### Roadmap (Proposed Enhancements)

- Deterministic test harness with mock LLM & fixtures
- Caching layer between repeated API calls across sessions
- Cost model normalization (different currencies → base currency via ExchangeRate API)
- Additional agents: Visa, Insurance, Safety Advisory
- Web UI (Next.js) integration & streaming plan updates
- Persistent vector store for previously generated itineraries

---
This README reflects the enhanced system structure including current implementation gaps (placeholders) and outlines concrete steps to complete and extend the platform.

### State Management

The system maintains a comprehensive state object that tracks:

- User requirements and preferences
- Search results from each specialist
- Agent handoffs and collaboration history
- Final plan synthesis

### Agent Collaboration

Each agent:

1. Receives context from previous agents
2. Performs specialized tasks using real APIs
3. Updates shared state with findings
4. Hands off to next agent with recommendations

## 🔍 Tracing and Observability

### OpenTelemetry Support

- Comprehensive span tracking for all agent activities
- Tool usage and API call monitoring
- Performance metrics and error tracking

### Azure Application Insights Integration

- Production-ready observability
- GenAI semantic conventions compliance
- Content recording with PII protection

### Configuration

```bash
# Azure Application Insights
APPLICATION_INSIGHTS_CONNECTION_STRING=your_connection_string

# OpenTelemetry (local development)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
```

## 🛠️ Development

### Project Structure

```text
travel_planning_system/
├── enhanced_multi_agent_system.py  # Main system
├── config.py                       # Configuration management
├── data_sources.py                 # API integrations
├── otel_tracer.py                  # OpenTelemetry tracer
├── setup.py                        # Setup and validation
├── requirements.txt                # Dependencies
├── .env.template                   # Configuration template
└── README.md                       # This file
```

### Key Components

#### Configuration (`config.py`)

- Centralized configuration management
- API key validation
- Environment-based settings
- Missing configuration detection

#### Data Sources (`data_sources.py`)

- Async API clients for all external services
- Intelligent fallback mechanisms
- Rate limiting and error handling
- Realistic data generation

#### Enhanced System (`enhanced_multi_agent_system.py`)

- LangGraph workflow implementation
- Specialized agent nodes
- State management and routing
- Comprehensive error handling

### Adding New APIs

1. Add configuration to `config.py`
2. Implement client in `data_sources.py`
3. Create fallback data generation
4. Update agent tools as needed
5. Test with and without API keys

## 🧪 Testing

### Manual Testing

```bash
# Test with sample inputs
python enhanced_multi_agent_system.py

# Test configuration
python setup.py
```

### System Validation

The setup script validates:

- Python version compatibility
- Required dependencies
- Configuration completeness
- Basic system functionality

## 📊 Performance

### API Usage Optimization

- Intelligent caching for repeated requests
- Batch API calls where possible
- Graceful degradation to fallbacks
- Rate limiting respect

### Resource Management

- Async HTTP client pooling
- Memory-efficient state management
- Configurable timeout handling
- Clean error recovery

## 🔒 Security and Privacy

### API Key Management

- Environment variable storage
- No hardcoded credentials
- Template-based configuration
- Validation warnings

### Content Recording

- Configurable PII protection
- Tracing content redaction
- Development vs production modes
- GDPR compliance considerations

## 🤝 Contributing

### Enhancement Ideas

- Additional destination coverage
- More specialized agents (visa, insurance)
- Integration with booking platforms
- Mobile app interface
- Voice interaction support

### Code Contributions

1. Follow existing patterns
2. Add comprehensive error handling
3. Include fallback mechanisms
4. Update documentation
5. Test with and without API keys

## 📄 License

This project is part of the LangGraph repository and follows the same licensing terms.

## 🆘 Support

### Common Issues

#### "No LLM configuration found"

- Add Azure OpenAI or OpenAI credentials to `.env`
- Run `python setup.py` to validate configuration

#### "API key missing" warnings

- Optional - system works with fallback data
- Add desired API keys for real data
- See API setup instructions above

#### Import errors

- Run `python setup.py` to install dependencies
- Ensure Python 3.8+ is installed

### Getting Help

1. Check the setup script output
2. Validate `.env` configuration
3. Review error messages for missing dependencies
4. Test with sample inputs first

---

**🎯 Goal**: Demonstrate sophisticated multi-agent orchestration with real data integration while maintaining usability and reliability.
