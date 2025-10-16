# Azure Multi-Agent Travel Planning Sample

This example shows how to orchestrate a small team of Azure OpenAI agents with
LangGraph to produce a complete travel itinerary. Every model invocation is
automatically traced with `langchain-azure-ai`, so you can follow the agent
handoffs live inside Azure Application Insights.

The graph coordinates six specialists:

- **requirements_analyst** – normalises the travel request into structured
  preferences.
- **flight_specialist** – suggests routes and cabin strategies.
- **stay_specialist** – recommends lodging matched to the brief.
- **activities_curator** – curates experiences and day trips.
- **budget_analyst** – estimates costs and savings ideas.
- **itinerary_author** – composes a traveller-facing plan.

Each role runs on the same Azure OpenAI deployment but receives its own prompt,
metadata, and tracing tags. The curated travel data bundled with this sample
keeps outputs consistent even when the trip brief is vague. Everything lives in
the single `travel_planning_system.py` file so you can tweak the graph quickly.

---

## Prerequisites

- Python 3.9 or newer.
- An [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/) resource
  with a chat completion deployment (for example `gpt-4o` or `gpt-4.1`).
- Optional but recommended: an [Azure Application Insights](https://learn.microsoft.com/azure/azure-monitor/app/app-insights-overview)
  instance to capture traces. The sample falls back to console-only if the
  connection string is not supplied.

---

## 1. Bootstrap the environment

```bash
cd examples/travel_planning_system
python -m venv .venv
source .venv/bin/activate            # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The requirements pin `langgraph`, `langchain-openai`, `langchain-azure-ai`,
> `opentelemetry-sdk`, and `python-dotenv`. If you prefer to use the packages
> from this monorepo, adjust the requirements accordingly.

---

## 2. Configure credentials

Create a `.env` file next to this README (or copy `.env.sample`) and fill in the
values that apply to your Azure resources:

```
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
AZURE_OPENAI_API_VERSION=2024-05-01-preview  # optional override

# Optional: Application Insights tracing
APPLICATION_INSIGHTS_CONNECTION_STRING=InstrumentationKey=...;IngestionEndpoint=...
OTEL_EXPORTER_OTLP_ENDPOINT=https://<region>.in.applicationinsights.azure.com/v2/track
```

Only the first three variables are mandatory. When the Application Insights
connection string is present, `langchain-azure-ai` forwards every specialist
invocation (prompt, tool usage, completion metadata, token counts) to your
workspace. Setting `OTEL_EXPORTER_OTLP_ENDPOINT` enables additional OpenTelemetry
export if you have the Python OTEL SDK installed.

---

## 3. Run the planner

```bash
python travel_planning_system.py
```

The CLI will prompt for a travel brief. Press <kbd>Enter</kbd> to reuse the
sample prompt:

```
Plan a five-day culinary getaway to Lisbon for two adults in September with a budget around $4,000. ...
```

You can also provide a one-off brief via `--prompt`:

```bash
python travel_planning_system.py --prompt "Family trip to Tokyo for four people this April; we love food and anime."
```

While the graph runs, the console prints each specialist’s contribution and ends
with a polished itinerary.

---

## 4. Inspect traces (optional)

1. Open the **Application Insights** resource linked to your connection string.
2. Navigate to **Transaction search** or **Performance** to locate spans tagged
   with `travel_planning`. Each agent uses its own `agent` metadata and tag so
   you can filter by specialist.
3. Expand the trace to view inputs, outputs, token usage, and tool metadata
   captured by `AzureOpenAITracingCallback`.

If you enabled OTLP export in `.env`, you can forward the traces to any collector
compatible with the OpenTelemetry protocol.

---

## Customising the sample

- Update the `CURATED_DATA` dictionary inside `travel_planning_system.py` to add
  destinations or swap in company-specific recommendations.
- Adjust the prompts or add extra nodes in `build_travel_planner_graph()` to
  experiment with more elaborate coordination patterns (e.g., conditional
  hand-offs or additional tooling).

---

## Troubleshooting

- **Missing environment variable** – ensure `.env` is loaded before running the
  script. The CLI raises a clear error when an Azure variable is absent.
- **`ModuleNotFoundError: langchain_openai`** – install dependencies from
  `requirements.txt` inside an activated virtual environment.
- **No traces in Application Insights** – double check the connection string,
  then rerun the script. Traces can take a minute to appear in the portal.

Enjoy building with LangGraph and Azure! We’d love feedback or PRs with further
enhancements.
