# LangGraph Server Changelog

> **Note:** This changelog is no longer actively maintained. For the most up-to-date LangGraph Server changelog, please visit our new documentation site: [LangGraph Server Changelog](https://docs.langchain.com/langgraph-platform/langgraph-server-changelog#langgraph-server-changelog)

[LangGraph Server](../../concepts/langgraph_server.md) is an API platform for creating and managing agent-based applications. It provides built-in persistence, a task queue, and supports deploying, configuring, and running assistants (agentic workflows) at scale. This changelog documents all notable updates, features, and fixes to LangGraph Server releases.

---

## v0.2.111 (2025-07-29)
- Started the heartbeat immediately upon connection to prevent JS graph streaming errors during long startups.

## v0.2.110 (2025-07-29)
- Added interrupts as default values for all operations except streams to maintain consistent behavior.

## v0.2.109 (2025-07-28)
- Fixed an issue where missing config schema occurred when `config_type` was not set.

## v0.2.108 (2025-07-28)
- Added compatibility for langgraph v0.6, including new context API support and a migration to enhance context handling in assistant operations.

## v0.2.107 (2025-07-27)
- Implemented caching for authentication processes to improve performance.
- Merged count and select queries to improve database query efficiency.

## v0.2.106 (2025-07-27)
- Log whether run uses resumable streams.

## v0.2.105 (2025-07-27)
- Added a `/heapdump` endpoint to capture and save JS process heap data.

## v0.2.103 (2025-07-25)
- Corrected the metadata endpoint to ensure accurate data retrieval.

## v0.2.102 (2025-07-24)
- Captured interrupt events in the wait method to preserve legacy behavior and stream updates by default.
- Added support for SDK structlog in the JavaScript environment, enhancing logging capabilities.

## v0.2.101 (2025-07-24)
- Used the correct metadata endpoint for self-hosted environments, resolving an access issue.

## v0.2.99 (2025-07-22)
- Improved license validation by adding an in-memory cache and handling Redis connection errors more effectively.
- Automatically remove agents from memory that are removed from `langgraph.json` to prevent persistence issues.
- Ensured the UI namespace for generated UI is a valid JavaScript property name to prevent errors.
- Raised a 422 error for improved request validation feedback.

## v0.2.98 (2025-07-19)
- Added langgraph node context for improved log filtering and trace visibility.

## v0.2.97 (2025-07-19)
- Fixed scheduling issue with ckpt ingestion worker that occurred on isolated background loops.
- Ensured queue worker starts only after all migrations have completed.
- Added more detailed error messages for thread state issues and improved response handling when state updates fail.
- Exposed interrupt ID while retrieving thread state for enhanced API response details.

## v0.2.96 (2025-07-17)
- Added a fallback mechanism for configurable header patterns to handle exclude/include settings more effectively.

## v0.2.95 (2025-07-17)
- Avoided setting the future if it is already done to prevent redundant operations.
- Resolved compatibility errors in CI by switching from `typing.TypedDict` to `typing_extensions.TypedDict` for Python versions below 3.12.

## v0.2.94 (2025-07-16)
- Improved performance by omitting pending sends for langgraph versions 0.5 and above.
- Improved server startup logs to provide clearer warnings when the DD_API_KEY environment variable is set.

## v0.2.93 (2025-07-16)
- Removed the GIN index for run metadata to improve performance.

## v0.2.92 (2025-07-16)
- Enabled copying functionality for blobs and checkpoints, improving data management flexibility.

## v0.2.91 (2025-07-16)
- Reduced writes to the `checkpoint_blobs` table by inlining small values (null, numeric, str, etc.). This means we don't need to store extra values for channels that haven't been updated.

## v0.2.90 (2025-07-16)
- Improve checkpoint writes via node-local background queueing.


## v0.2.89 (2025-07-15)
- Decoupled checkpoint writing from thread/run state by removing foreign keys and updated logger to prevent timeout-related failures.

## v0.2.88 (2025-07-14)
- Removed the foreign key constraint for `thread` in the `run` table to simplify database schema.

## v0.2.87 (2025-07-14)
- Added more detailed logs for Redis worker signaling to improve debugging.

## v0.2.86 (2025-07-11)
- Honored tool descriptions in the `/mcp` endpoint to align with expected functionality.

## v0.2.85 (2025-07-10)
- Added support for the `on_disconnect` field to `runs/wait` and included disconnect logs for better debugging.

## v0.2.84 (2025-07-09)
- Removed unnecessary status updates to streamline thread handling and updated version to 0.2.84.

## v0.2.83 (2025-07-09)
- Reduced the default time-to-live for resumable streams to 2 minutes.
- Enhanced data submission logic to send data to both Beacon and LangSmith instance based on license configuration.
- Enabled submission of self-hosted data to a Langsmith instance when the endpoint is configured.

## v0.2.82 (2025-07-03)
- Addressed a race condition in background runs by implementing a lock using join, ensuring reliable execution across CTEs.

## v0.2.81 (2025-07-03)
- Optimized run streams by reducing initial wait time to improve responsiveness for older or non-existent runs.

## v0.2.80 (2025-07-03)
- Corrected parameter passing in the `logger.ainfo()` API call to resolve a TypeError.

## v0.2.79 (2025-07-02)
- Fixed a JsonDecodeError in checkpointing with remote graph by correcting JSON serialization to handle trailing slashes properly.
- Introduced a configuration flag to disable webhooks globally across all routes.

## v0.2.78 (2025-07-02)
- Added timeout retries to webhook calls to improve reliability.
- Added HTTP request metrics, including a request count and latency histogram, for enhanced monitoring capabilities.

## v0.2.77 (2025-07-02)
- Added HTTP metrics to improve performance monitoring.
- Changed the Redis cache delimiter to reduce conflicts with subgraph message names and updated caching behavior.

## v0.2.76 (2025-07-01)
- Updated Redis cache delimiter to prevent conflicts with subgraph messages.

## v0.2.74 (2025-06-30)
- Scheduled webhooks in an isolated loop to ensure thread-safe operations and prevent errors with PYTHONASYNCIODEBUG=1.

## v0.2.73 (2025-06-27)
- Fixed an infinite frame loop issue and removed the dict_parser due to structlog's unexpected behavior.
- Throw a 409 error on deadlock occurrence during run cancellations to handle lock conflicts gracefully.

## v0.2.72 (2025-06-27)
- Ensured compatibility with future langgraph versions.
- Implemented a 409 response status to handle deadlock issues during cancellation.

## v0.2.71 (2025-06-26)
- Improved logging for better clarity and detail regarding log types.

## v0.2.70 (2025-06-26)
- Improved error handling to better distinguish and log TimeoutErrors caused by users from internal run timeouts.

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
