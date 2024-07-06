# Environment Variables

The LangGraph Cloud API supports specific environment variables for configuring a deployment.

## `LANGGRAPH_AUTH_TYPE`

Type of authentication for the LangGraph Cloud API deployment. Valid values: `langsmith`, `noop`.

For deployments to LangGraph Cloud, this environment variable is set automatically. For local development or deployments where authentication is handled externally (e.g. self-hosted), set this environment variable to `noop`.

## `N_JOBS_PER_WORKER`

Number of jobs per worker for the LangGraph Cloud task queue. Defaults to `10`.
