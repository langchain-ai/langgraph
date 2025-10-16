# Enhanced Azure Multi-Agent Travel Planner

This sample mirrors the behaviour of `enhanced_multi_agent_system.py` that you
liked: a LangGraph supervisor coordinates six Azure OpenAI specialists, each
tagged for tracing with `AzureAIOpenTelemetryTracer`. The graph enriches its
answers with curated fallback data and optional live API calls (flights, hotels,
weather, activities, currency).

All code for the workflow lives across three files in this directory:

- `enhanced_multi_agent_system.py` – orchestrates the multi-agent LangGraph,
  manages tracing, and exposes a CLI entry point.
- `config.py` – loads environment variables for LLMs, tracing, and optional
  external APIs.
- `data_sources.py` – fetches data from real services with graceful fallbacks.

Everything else from the old sample has been removed to keep things focused.

---

## Prerequisites

- Python 3.9+.
- An [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/)
  resource with a chat-completion deployment (for example `gpt-4o` or `gpt-4.1`).
- Optional: an [Application Insights](https://learn.microsoft.com/azure/azure-monitor/app/app-insights-overview)
  instance (or any OTLP collector) if you want to inspect traces.

---

## 1. Set up a virtual environment

```bash
cd examples/travel_planning_system
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Configure environment variables

Copy `.env.sample` to `.env` and populate the values that apply to you:

```
cp .env.sample .env
```

At minimum you must set:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_DEPLOYMENT_NAME=<your-chat-deployment>
```

Optional but recommended:

- `APPLICATION_INSIGHTS_CONNECTION_STRING` – enables `AzureAIOpenTelemetryTracer`
  so every agent/tool call shows up in Azure Monitor.
- `OTEL_EXPORTER_OTLP_ENDPOINT` – forward spans to another collector.
- API keys for OpenWeather, Amadeus, ExchangeRate, Foursquare, Eventbrite to
  replace the curated fallback data with live results.

`config.py` documents every environment variable that can be supplied.

---

## 3. Run the sample

```bash
python enhanced_multi_agent_system.py
```

You will be prompted for a travel request. Press <kbd>Enter</kbd> to use the
interactive defaults or provide your own scenario. The coordinator agent will
hand off to the flight, hotel, activity, budget, and plan synthesis agents. The
console prints each step and finishes with a full itinerary.

### Streaming with tracing

With `APPLICATION_INSIGHTS_CONNECTION_STRING` set, you will see console output:

```
AzureAIOpenTelemetryTracer(...)
azure tracer initialized
```

Each Azure OpenAI call is tagged with `agent:<name>` and `enhanced-travel-planning`
so you can filter spans per specialist. If you also set
`OTEL_EXPORTER_OTLP_ENDPOINT`, spans are exported via OTLP in parallel.

---

## 4. Inspect traces (optional)

1. Open Application Insights and use **Transaction search** to locate requests
   tagged with `enhanced-travel-planning`.
2. Expand a request to view the timeline of agent/tool calls, prompts, responses,
   and token usage captured by `AzureAIOpenTelemetryTracer`.
3. If you configured OTLP export, use your collector’s UI (Jaeger, Grafana, etc.)
   to view the same spans.

---

## Customisation ideas

- Adjust prompt templates, temperatures, or tool wiring directly in
  `enhanced_multi_agent_system.py`.
- Update `data_sources.py` with organisation-specific integrations.
- Extend `config.py` with additional service credentials and expose them through
  new agents or tools.

Enjoy experimenting with LangGraph + Azure!
