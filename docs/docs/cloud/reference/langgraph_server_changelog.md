# LangGraph Server Changelog

[LangGraph Server](../../concepts/langgraph_server.md) is an API platform for creating and managing agent-based applications. It provides built-in persistence, a task queue, and supports deploying, configuring, and running assistants (agentic workflows) at scale. This changelog documents all notable updates, features, and fixes to LangGraph Server releases.

---

## v0.2.83 (2025-07-09)
- Reduced the `default` TTL for `resumable` `streams` to 2 `minutes`.
- Added `functionality` to `submit` `data` to a LangSmith `instance` if the `endpoint` is `configured`, `supporting` `various` `deployment` `modes` `depending` on `license` and API `key` `configuration`.
- Enabled `submission` of `self`-`hosted` `data` to ``langsmith`` `instance` when the `endpoint` is `configured`.

## v0.2.82 (2025-07-03)
- Resolved a `race` `condition` in `Runs.`next`` by `using` a `join` to `lock` `runs`.

## v0.2.81 (2025-07-03)
- Introduced a `placeholder` `/`ok`` `endpoint` to `ensure` `deployment` `success` `even` when ``disable_meta`=True`.
- Reduced `wait` `time` for `starting` `stream` `calls` by `switching` to a `shorter` `initial` `wait` `period`.

## v0.2.80 (2025-07-03)
- Fixed `incorrect` `parameter` `usage` in ``logger`.`ainfo`()` to `prevent` TypeError `exceptions`.

## v0.2.79 (2025-07-02)
- Fixed an `issue` where `JsonDecodeError` `occurred` `during` `checkpointing` with `invalid` JSON, `ensuring` `proper` `loading` by `handling` `trailing` `slashes` `correctly`.
- Added a ``disable_webhooks`` `configuration` `flag` to `allow` `disabling` `webhooks` `across` all `routes`.

## v0.2.78 (2025-07-02)
- Added `retry` `mechanism` to `webhook` `calls` to `handle` `timeouts`.
- Added ``lg_api_http_requests_total`` `counter` and ``lg_api_http_requests_latency_seconds`` `histogram` `metrics` to `enable` `requests` `per` `second` and `latency` `chart` `plotting`.

## v0.2.77 (2025-07-02)
- Added HTTP `metrics` to `improve` `monitoring` and `performance` `analysis`.
- Changed the Redis `cache` `delimiter` to `reduce` `conflicts` with `subgraph` `messages` by `using` a `different` `character`.

## v0.2.76 (2025-07-01)
- Updated ``redis`` `cache` `delimiter` to `prevent` `conflicts` with `subgraph` `messages`.

## v0.2.74 (2025-06-30)
- Scheduled `webhooks` in an `isolated` `loop` to `ensure` `thread` `safety` and `prevent` `errors` `during` `event` `handling`.

## v0.2.73 (2025-06-27)
- Resolved an `infinite` `loop` `issue` in `frame` `processing` and `removed` ``dict_parser`` for `improved` `logging` `compatibility`.
- Throw a `409 Conflict` `error` when a `deadlock` `occurs` `during` `cancellation` `attempts`.

## v0.2.72 (2025-06-27)
- Added `compatibility` with `future` ``langgraph`` `updates`.
- Added a 409 `error` `response` for `deadlock` `situations` `during` `cancellation` `requests`.

## v0.2.71 (2025-06-26)
- Resolved an `issue` with `Log` `type` that was `causing` `incorrect` `logging` `behavior`.

## v0.2.70 (2025-06-26)
- Improved `error` `handling` by `distinguishing` `between` `user` `TimeoutError` and `run` `timeouts`, `ensuring` `accurate` `error` `logging` and `robust` `execution`.

## v0.2.69 (2025-06-26)
- Enhanced the ``crons`` API by `adding` `sorting` and `pagination`, and `updated` `schema` `definitions` for `accuracy`.

## v0.2.66 (2025-06-26)
- Fixed a 404 error when creating multiple runs with the same thread_id using `on_not_exist="create"`.

## v0.2.65 (2025-06-25)
- Ensured that only fields from `assistant_versions` are returned when necessary.
- Ensured consistent data types for in-memory and PostgreSQL users, improving internal authentication handling.

## v0.2.64 (2025-06-24)
- Added descriptions to version entries for better clarity.

## v0.2.62 (2025-06-23)
- Improved user handling for custom authentication in the JS Studio.
- Added Prometheus-format run statistics to the metrics endpoint for better monitoring.
- Added run statistics in Prometheus format to the metrics endpoint.

## v0.2.61 (2025-06-20)
- Set a maximum idle time for Redis connections to prevent unnecessary open connections.

## v0.2.60 (2025-06-20)
- Enhanced error logging to include traceback details for dictionary operations.
- Added a `/metrics` endpoint to expose queue worker metrics for monitoring.

## v0.2.57 (2025-06-18)
- Removed CancelledError from retriable exceptions to allow local interrupts while maintaining retriability for workers.
- Introduced middleware to gracefully shut down the server after completing in-flight requests upon receiving a SIGINT.
- Reduced metadata stored in checkpoint to only include necessary information.
- Improved error handling in join runs to return error details when present.

## v0.2.56 (2025-06-17)
- Improved application stability by adding a handler for SIGTERM signals.

## v0.2.55 (2025-06-17)
- Improved the handling of cancellations in the queue entrypoint.
- Improved cancellation handling in the queue entry point.

## v0.2.54 (2025-06-16)
- Enhanced error message for LuaLock timeout during license validation.
- Fixed the $contains filter in custom auth by requiring an explicit ::text cast and updated tests accordingly.
- Ensured project and tenant IDs are formatted as UUIDs for consistency.

## v0.2.53 (2025-06-13)
- Resolved a timing issue to ensure the queue starts only after the graph is registered.
- Improved performance by setting thread and run status in a single query and enhanced error handling during checkpoint writes.
- Reduced the default background grace period to 3 minutes.

## v0.2.52 (2025-06-12)
- Now logging expected graphs when one is omitted to improve traceability.
- Implemented a time-to-live (TTL) feature for resumable streams.
- Improved query efficiency and consistency by adding a unique index and optimizing row locking.

## v0.2.51 (2025-06-12)
- Handled `CancelledError` by marking tasks as ready to retry, improving error management in worker processes.
- Added LG API version and request ID to metadata and logs for better tracking.
- Added LG API version and request ID to metadata and logs to improve traceability.
- Improved database performance by creating indexes concurrently.
- Ensured postgres write is committed only after the Redis running marker is set to prevent race conditions.
- Enhanced query efficiency and reliability by adding a unique index on thread_id/running, optimizing row locks, and ensuring deterministic run selection.
- Resolved a race condition by ensuring Postgres updates only occur after the Redis running marker is set.

## v0.2.46 (2025-06-07)
- Introduced a new connection for each operation while preserving transaction characteristics in Threads state `update()` and `bulk()` commands.

## v0.2.45 (2025-06-05)
- Enhanced streaming feature by incorporating tracing contexts.
- Removed an unnecessary query from the Crons.search function.
- Resolved connection reuse issue when scheduling next run for multiple cron jobs.
- Removed an unnecessary query in the Crons.search function to improve efficiency.
- Resolved an issue with scheduling the next cron run by improving connection reuse.

## v0.2.44 (2025-06-04)
- Enhanced the worker logic to exit the pipeline before continuing when the Redis message limit is reached.
- Introduced a ceiling for Redis message size with an option to skip messages larger than 128 MB for improved performance.
- Ensured the pipeline always closes properly to prevent resource leaks.

## v0.2.43 (2025-06-04)
- Improved performance by omitting logs in metadata calls and ensuring output schema compliance in value streaming.
- Ensured the connection is properly closed after use.
- Aligned output format to strictly adhere to the specified schema.
- Stopped sending internal logs in metadata requests to improve privacy.

## v0.2.42 (2025-06-04)
- Added timestamps to track the start and end of a request's run.
- Added tracer information to the configuration settings.
- Added support for streaming with tracing contexts.

## v0.2.41 (2025-06-03)
- Added locking mechanism to prevent errors in pipelined executions.
