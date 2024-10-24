# LangGraph Deploy

With LangGraph Deploy, you define a simple configuration file and it generates a Docker image with:

🎁 Automatic HTTP prediction server: Your agent definition is used to dynamically generate a RESTful HTTP API.

🥞 Automatic queue worker. Long-running agents or batch processing is best architected with a queue. LangGraph Deploy manages this out of the box.

☁️ Persistence storage. Optimized Postgres Checkpointers and Stores are spun up and attached to your agent, giving it persistence. This enables memory and human-in-the-loop interation patterns.

🚀 Ready for production. Deploy your model anywhere that Docker images run. Your own infrastructure, or LangGraph Cloud.

## How it works

### Define the Docker environment your agent runs in with `langgraph.json`:

```
{
  "dependencies": ["./my_agent"],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "env": ".env"
}
```
### Define your graph in `agent.py`:

# TODO: this should be more self contained
```
# my_agent/agent.py
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from my_agent.utils.nodes import call_model, should_continue, tool_node # import nodes
from my_agent.utils.state import AgentState # import state

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

graph = workflow.compile()
```

To build the docker image, you first need to install the Cli:

```shell
pip install langgraph-cli
```

You can then use:

```
langgraph build -t my-image
```

This will build a docker image with the LangGraph Deploy server. The `-t my-image` is used to tag the image with a name.

When running this server, you need to pass three environment variables:

# TODO: change DATABASE_URL name to POSTGRES_URI?

- `REDIS_URI`: Connection details to a Redis instance. Redis will be used as a pub-sub broker to enable streaming real time output from background runs.
- `DATABASE_URI`: Postgres connection details. Postgres will be used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics.
- `LANGSMITH_API_KEY`: LangSmith API key. This will be used to authenticate ONCE at server start up.

```shell
docker run  -e REDIS_URI="foo" -e DATABASE_URI="bar" -e LANGSMITH_API_KEY="baz" my-image
```

If you want to run this quickly without setting up a separate Redis and Postgres instance, you can use this docker compose file. 

**NOTE #1:** notice the points below where you will need to put your own `IMAGE_NAME` (from `langgraph build` step above) and your `LANGSMITH_API_KEY` 

**NOTE #2:** if your graph requires other environment variables, you will need to add them in the langgraph-api service.

```text
volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-postgres:
        image: postgres:16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-api:
        image: {IMAGE_NAME}
        ports:
            - "8123:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            LANGSMITH_API_KEY: {LANGSMITH_API_KEY}
            POSTGRES_URI: postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable
```

You can then run `docker compose up` with this Docker compose file in the same folder.

This will spin up LangGraph Deploy on port 8123 (if you want to change this, you can change this by changing the ports in the `langgraph-api` volume).

You can test that this up by trying to create a thread:

```shell
curl --request POST \
  --url 0.0.0.0:8123/threads \
  --header 'Content-Type: application/json' \
  --data '{}'
```

Assuming everything is running correctly, you should see a response like:

```shell
{"thread_id":"166177a7-ad9f-4e5e-ac74-2645072cb434","created_at":"2024-10-24T02:29:28.710696+00:00","updated_at":"2024-10-24T02:29:28.710696+00:00","metadata":{},"status":"idle","config":{},"values":null}%
```

