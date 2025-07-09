# LangGraph Server Changelog

[LangGraph Server](../../concepts/langgraph_server.md) is an API platform for creating and managing agent-based applications. It provides built-in persistence, a task queue, and supports deploying, configuring, and running assistants (agentic workflows) at scale. This changelog documents all notable updates, features, and fixes to LangGraph Server releases.

---

## v0.2.83 (2025-07-09)
- Reduced the default TTL for resumable streams to 2 minutes for improved resource management.
- Enhanced data submission logic to send self-hosted data to the LangSmith instance, with conditional submissions to Beacon based on license type and deployment mode.
- Enabled automatic submission of self-hosted data to a Langsmith instance when an endpoint is configured.

## v0.2.82 (2025-07-03)
- Implemented a join to lock runs and prevent race conditions across CTEs in `Runs.next`, ensuring smoother background operations.

## v0.2.81 (2025-07-03)
- Retained the `/ok` endpoint to ensure successful deployment even when `disable_meta=True`.
- Optimized stream processing by starting with a shorter wait time to improve response times for older or nonexistent runs.

## v0.2.80 (2025-07-03)
- Resolved a TypeError in `logger.ainfo()` by correctly passing the `event` parameter as a named argument.

## v0.2.79 (2025-07-02)
- Resolved a JsonDecodeError in checkpointing with remote graphs by improving how invalid JSON is handled.
- Introduced a configuration flag to disable webhooks across all routes.

## v0.2.78 (2025-07-02)
- Added retries for webhook calls that experience timeouts to improve reliability.
- Added HTTP request counter and latency histogram metrics for enhanced monitoring and analysis.

## v0.2.77 (2025-07-02)
- Added HTTP metrics to enhance monitoring capabilities.
- Updated the Redis cache delimiter to reduce conflicts with subgraph messages.

## v0.2.76 (2025-07-01)
- Updated the Redis cache delimiter to prevent conflicts with subgraph messages.

## v0.2.74 (2025-06-30)
- Scheduled webhook events in an isolated loop for enhanced thread safety and error prevention.

## v0.2.73 (2025-06-27)
- Fixed an infinite frame loop and removed the dict_parser to streamline logging.
- Issued a 409 error when encountering a deadlock during the cancel operation.

## v0.2.72 (2025-06-27)
- Avoid catching cancellation errors in SSE heartbeat to improve process handling.
- Returned a 409 error when encountering a deadlock during cancellation.

## v0.2.71 (2025-06-26)
- Improved logging mechanism to enhance tracking and debugging with detailed type information.

## v0.2.70 (2025-06-26)
- Improved error handling by distinguishing between user and runtime TimeoutErrors for clearer logging and encapsulation.

## v0.2.69 (2025-06-26)
- Added sorting and pagination to the crons API and updated schema definitions for improved accuracy.

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
