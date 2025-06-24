# Data Storage and Privacy

This document provides a comprehensive overview of what data is stored, collected, and processed when using LangGraph, particularly with the CLI tools like `langgraph dev`.

## What Data is Stored

### CLI Telemetry (Opt-out)

By default, the LangGraph CLI collects minimal analytics data to help improve the tool:

**Data Collected:**
- CLI command used (e.g., `dev`, `up`, `build`)
- CLI version
- Operating system type and version
- Python version
- Anonymized parameter usage (boolean flags indicating non-default options were used)

**Data NOT Collected:**
- Actual parameter values
- File contents or paths
- Personal information
- Code or graph implementations
- API keys or sensitive data

**How to Opt Out:**
Set the environment variable `LANGGRAPH_CLI_NO_ANALYTICS=1` to disable all CLI analytics collection.

### LangSmith Integration (Opt-in)

When a `LANGSMITH_API_KEY` is provided (not required):
- Metadata on number of runs executed
- Current API version being run
- Trace data (if tracing is enabled)

This data is only sent when explicitly configured with LangSmith credentials.

### Tracing Data (Opt-in)

When tracing is enabled:
- Execution traces are logged to the configured tracing backend
- This requires explicit configuration and is not enabled by default

## What Data is NOT Stored Remotely

- **Checkpoints**: Stored locally in your development environment
- **Memory store data**: Persisted locally, not transmitted
- **Graph state**: Remains in your local environment
- **Application data**: Your actual application logic and data stay local

## Local Data Storage

### Development Mode (`langgraph dev`)

When using `langgraph dev`:
- State is persisted to a local directory
- Checkpoints are stored locally for debugging and development
- No remote storage or transmission of your application data

### Checkpoints and State Persistence

LangGraph automatically persists:
- **Checkpoints**: Snapshots of graph state at each execution step
- **Thread data**: Conversation/execution history organized by thread IDs
- **Graph state**: Node outputs, intermediate results, and execution metadata
- **Memory/Store data**: Information that persists across multiple threads

**Storage Locations:**
- **Local development**: Local directory (configurable)
- **Docker deployment**: Local Docker volumes
- **LangGraph Platform**: Managed database infrastructure

## Security and Encryption

### Data Encryption

- Checkpointers can optionally encrypt all persisted state
- Encryption uses AES encryption via `EncryptedSerializer`
- When `LANGGRAPH_AES_KEY` environment variable is present, encryption is automatically enabled on LangGraph Platform

### Data Retention

- **TTL (Time-to-Live)**: Configurable automatic cleanup of old data
- **Default TTL**: Can be set in minutes for automatic expiration
- **Automatic sweeping**: Expired data is automatically removed at configurable intervals

## Privacy Controls

### Environment Variables

Key environment variables for controlling data collection and storage:

- `LANGGRAPH_CLI_NO_ANALYTICS=1`: Disable CLI analytics collection
- `LANGGRAPH_AES_KEY`: Enable automatic encryption of stored data
- `LANGSMITH_TRACING=false`: Disable tracing to LangSmith (self-hosted deployments)
- `LANGSMITH_API_KEY`: Enable LangSmith integration (opt-in)

### Logging Controls

- `LOG_LEVEL`: Control verbosity of logs
- `LOG_JSON`: Format logs as JSON
- Various other logging configuration options

## Security Policy

For security vulnerabilities:
- Report through the huntr.com bounty program
- LangGraph is in-scope for security bounties
- Security contact: `security@langchain.dev`

## Best Practices for Privacy

1. **Review Analytics**: Set `LANGGRAPH_CLI_NO_ANALYTICS=1` if you prefer not to share usage analytics
2. **Enable Encryption**: Use `LANGGRAPH_AES_KEY` for sensitive data
3. **Configure TTL**: Set appropriate data retention policies
4. **Monitor Tracing**: Only enable tracing when needed and review what data is being sent
5. **Environment Variables**: Audit your environment variables to ensure proper privacy controls

## Summary

LangGraph is designed with privacy in mind:
- Minimal data collection (analytics can be disabled)
- Local storage by default for development
- Optional encryption for sensitive data
- Clear opt-in requirements for external services
- Comprehensive privacy controls through environment variables

Your application data, checkpoints, and state remain under your control and are not transmitted unless you explicitly configure external services like LangSmith.