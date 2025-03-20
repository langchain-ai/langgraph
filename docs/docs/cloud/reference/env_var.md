# Environment Variables

The LangGraph Cloud Server supports specific environment variables for configuring a deployment.

## `DD_API_KEY`

Specify `DD_API_KEY` (your [Datadog API Key](https://docs.datadoghq.com/account_management/api-app-keys/)) to automatically enable Datadog tracing for the deployment. Specify other [`DD_*` environment variables](https://ddtrace.readthedocs.io/en/stable/configuration.html) to configure the tracing instrumentation.

If `DD_API_KEY` is specified, the application process is wrapped in the [`ddtrace-run` command](https://ddtrace.readthedocs.io/en/stable/installation_quickstart.html). Other `DD_*` environment variables (e.g. `DD_SITE`, `DD_ENV`, `DD_SERVICE`, `DD_TRACE_ENABLED`) are typically needed to properly configure the tracing instrumentation. See [`DD_*` environment variables](https://ddtrace.readthedocs.io/en/stable/configuration.html) for more details.

## `LANGCHAIN_TRACING_SAMPLING_RATE`

Sampling rate for traces sent to LangSmith. Valid values: Any float between `0` and `1`.

See <a href="https://docs.smith.langchain.com/how_to_guides/tracing/sample_traces" target="_blank">LangSmith documentation</a> for more details.

## `LANGGRAPH_AUTH_TYPE`

Type of authentication for the LangGraph Cloud Server deployment. Valid values: `langsmith`, `noop`.

For deployments to LangGraph Cloud, this environment variable is set automatically. For local development or deployments where authentication is handled externally (e.g. self-hosted), set this environment variable to `noop`.

## `LANGSMITH_RUNS_ENDPOINTS`

For [Bring Your Own Cloud (BYOC)](../../concepts/bring_your_own_cloud.md) deployments with [self-hosted LangSmith](https://docs.smith.langchain.com/self_hosting) only.

Set this environment variable to have a BYOC deployment send traces to a self-hosted LangSmith instance. The value of `LANGSMITH_RUNS_ENDPOINTS` is a JSON string: `{"<SELF_HOSTED_LANGSMITH_HOSTNAME>":"<LANGSMITH_API_KEY>"}`.

`SELF_HOSTED_LANGSMITH_HOSTNAME` is the hostname of the self-hosted LangSmith instance. It must be accessible to the BYOC deployment. `LANGSMITH_API_KEY` is a LangSmith API generated from the self-hosted LangSmith instance.

## `N_JOBS_PER_WORKER`

Number of jobs per worker for the LangGraph Cloud task queue. Defaults to `10`.

## `POSTGRES_URI_CUSTOM`

For [Bring Your Own Cloud (BYOC)](../../concepts/bring_your_own_cloud.md) deployments only.

Specify `POSTGRES_URI_CUSTOM` to use an externally managed Postgres instance. The value of `POSTGRES_URI_CUSTOM` must be a valid [Postgres connection URI](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING-URIS).

Postgres:

- Version 15.8 or higher.
- An initial database must be present and the connection URI must reference the database.

Control Plane Functionality:

- If `POSTGRES_URI_CUSTOM` is specified, the LangGraph Control Plane will not provision a database for the server.
- If `POSTGRES_URI_CUSTOM` is removed, the LangGraph Control Plane will not provision a database for the server and will not delete the externally managed Postgres instance.
- If `POSTGRES_URI_CUSTOM` is removed, deployment of the revision will not succeed. Once `POSTGRES_URI_CUSTOM` is specified, it must always be set for the lifecycle of the deployment.
- If the deployment is deleted, the LangGraph Control Plane will not delete the externally managed Postgres instance.
- The value of `POSTGRES_URI_CUSTOM` can be updated. For example, a password in the URI can be updated.

Database Connectivity:

- The externally managed Postgres instance must be accessible by the LangGraph Server service in the ECS cluster. The BYOC user is responsible for ensuring connectivity.
- For example, if an AWS RDS Postgres instance is provisioned, it can be provisioned in the same VPC (`langgraph-cloud-vpc`) as the ECS cluster with the `langgraph-cloud-service-sg` security group to ensure connectivity.
