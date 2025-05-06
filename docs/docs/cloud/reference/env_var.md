# Environment Variables

The LangGraph Server supports specific environment variables for configuring a deployment.

## `BG_JOB_ISOLATED_LOOPS`

Set `BG_JOB_ISOLATED_LOOPS` to `True` to execute background runs in an isolated event loop separate from the serving API event loop.

This environment variable should be set to `True` if the implementation of a graph/node contains synchronous code. In this situation, the synchronous code will block the serving API event loop, which may cause the API to be unavailable. A symptom of an unavailable API is continuous application restarts due to failing health checks.

Defaults to `False`.

## `BG_JOB_TIMEOUT_SECS`

The timeout of a background run can be increased. However, the infrastructure for a Cloud SaaS deployment enforces a 1 hour timeout limit for API requests. This means the connection between client and server will timeout after 1 hour. This is not configurable.

A background run can execute for longer than 1 hour, but a client must reconnect to the server (e.g. join stream via `POST /threads/{thread_id}/runs/{run_id}/stream`) to retrieve output from the run if the run is taking longer than 1 hour.

Defaults to `3600`.

## `DD_API_KEY`

Specify `DD_API_KEY` (your [Datadog API Key](https://docs.datadoghq.com/account_management/api-app-keys/)) to automatically enable Datadog tracing for the deployment. Specify other [`DD_*` environment variables](https://ddtrace.readthedocs.io/en/stable/configuration.html) to configure the tracing instrumentation.

If `DD_API_KEY` is specified, the application process is wrapped in the [`ddtrace-run` command](https://ddtrace.readthedocs.io/en/stable/installation_quickstart.html). Other `DD_*` environment variables (e.g. `DD_SITE`, `DD_ENV`, `DD_SERVICE`, `DD_TRACE_ENABLED`) are typically needed to properly configure the tracing instrumentation. See [`DD_*` environment variables](https://ddtrace.readthedocs.io/en/stable/configuration.html) for more details.

## `LANGCHAIN_TRACING_SAMPLING_RATE`

Sampling rate for traces sent to LangSmith. Valid values: Any float between `0` and `1`.

See <a href="https://docs.smith.langchain.com/how_to_guides/tracing/sample_traces" target="_blank">LangSmith documentation</a> for more details.

## `LANGGRAPH_AUTH_TYPE`

Type of authentication for the LangGraph Server deployment. Valid values: `langsmith`, `noop`.

For deployments to LangGraph Cloud, this environment variable is set automatically. For local development or deployments where authentication is handled externally (e.g. self-hosted), set this environment variable to `noop`.

## `LANGSMITH_RUNS_ENDPOINTS`

For [Bring Your Own Cloud (BYOC)](../../concepts/bring_your_own_cloud.md) deployments with [self-hosted LangSmith](https://docs.smith.langchain.com/self_hosting) only.

Set this environment variable to have a BYOC deployment send traces to a self-hosted LangSmith instance. The value of `LANGSMITH_RUNS_ENDPOINTS` is a JSON string: `{"<SELF_HOSTED_LANGSMITH_HOSTNAME>":"<LANGSMITH_API_KEY>"}`.

`SELF_HOSTED_LANGSMITH_HOSTNAME` is the hostname of the self-hosted LangSmith instance. It must be accessible to the BYOC deployment. `LANGSMITH_API_KEY` is a LangSmith API generated from the self-hosted LangSmith instance.

## `LANGSMITH_TRACING`

!!! info "Only for Self-Hosted Data Plane, Self-Hosted Control Plane, and Standalone Container"
    Disabling LangSmith tracing is only available for [Self-Hosted Data Plane](../../concepts/langgraph_self_hosted_data_plane.md), [Self-Hosted Control Plane](../../concepts/langgraph_self_hosted_control_plane.md), and [Standalone Container](../../concepts/langgraph_standalone_container.md) deployments.

Set `LANGSMITH_TRACING` to `false` to disable tracing to LangSmith.

## `LOG_LEVEL`

Configure [log level](https://docs.python.org/3/library/logging.html#logging-levels). Defaults to `INFO`.

## `LOG_JSON`

Set `LOG_JSON` to `true` to render all log messages as JSON objects using the configured `JSONRenderer`. This produces structured logs that can be easily parsed or ingested by log management systems. Defaults to `false`.

## `LOG_COLOR`

This is mainly relevant in the context of using the dev server via the `langgraph dev` command. Set `LOG_COLOR` to `true` to enable ANSI-colored console output when using the default console renderer. Disabling color output by setting this variable to `false` produces monochrome logs. Defaults to `true`.

## `N_JOBS_PER_WORKER`

Number of jobs per worker for the LangGraph Server task queue. Defaults to `10`.

## `POSTGRES_URI_CUSTOM`

!!! info "Only for Self-Hosted Data Plane and Self-Hosted Control Plane"
    Custom Postgres instances are only available for [Self-Hosted Data Plane](../../concepts/langgraph_self_hosted_data_plane.md) and [Self-Hosted Control Plane](../../concepts/langgraph_self_hosted_control_plane.md) deployments.

Specify `POSTGRES_URI_CUSTOM` to use a custom Postgres instance. The value of `POSTGRES_URI_CUSTOM` must be a valid [Postgres connection URI](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING-URIS).

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

- The custom Postgres instance must be accessible by the LangGraph Server. The user is responsible for ensuring connectivity.

## `LANGGRAPH_POSTGRES_POOL_MAX_SIZE`

Beginning with langgraph-api version `0.2.12`, the maximum size of the Postgres connection pool can be controlled using the `LANGGRAPH_POSTGRES_POOL_MAX_SIZE` environment variable. By setting this variable, you can determine the upper bound on the number of simultaneous connections the server will establish with the Postgres database. This is particularly useful for deployments where database resources are limited (or more available) or where you need to tune connection behavior for performance or scaling reasons. If not specified, the pool size defaults to 150 connections.

## `REDIS_URI_CUSTOM`

!!! info "Only for Self-Hosted Data Plane and Self-Hosted Control Plane"
    Custom Redis instances are only available for [Self-Hosted Data Plane](../../concepts/langgraph_self_hosted_data_plane.md) and [Self-Hosted Control Plane](../../concepts/langgraph_self_hosted_control_plane.md) deployments.

Specify `REDIS_URI_CUSTOM` to use a custom Redis instance. The value of `REDIS_URI_CUSTOM` must be a valid [Redis connection URI](https://redis-py.readthedocs.io/en/stable/connections.html#redis.Redis.from_url).

## `REDIS_KEY_PREFIX`

!!! info "Available in API Server version 0.1.9+"
    This environment variable is supported in API Server version 0.1.9 and above.

Specify a prefix for Redis keys. This allows multiple LangGraph Server instances to share the same Redis instance by using different key prefixes. 

Defaults to `''`.

## `REDIS_CLUSTER`

!!! info "Only Allowed in Self-Hosted Deployments"
    Redis Cluster mode is only available in Self-Hosted Deployment models, LangGraph Cloud SaaS will provision a redis instance for you by default.

Set `REDIS_CLUSTER` to `True` to enable Redis Cluster mode. When enabled, the system will connect to Redis using cluster mode. This is useful when connecting to a Redis Cluster deployment.

Defaults to `False`.
