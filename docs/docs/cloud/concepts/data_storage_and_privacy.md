# CLI Data Storage and Privacy

This document describes how [LangGraph CLI](../../concepts/langgraph_cli.md) handles data storage, collection, and privacy. It covers what data is stored locally versus remotely, security measures, and privacy controls.

## Overview

LangGraph CLI is designed with privacy-first principles:

- **Local-first storage**: Your data stays on your machine by default
- **Minimal telemetry**: Optional analytics that can be disabled
- **Encryption support**: Optional encryption for sensitive data
- **Clear opt-in controls**: External services require explicit configuration

## Data Storage Locations

### Local Storage (Default)

All application data is stored locally unless explicitly configured otherwise:

- **Checkpoints**: Graph state snapshots for debugging and development
- **Thread data**: Conversation and execution history
- **Graph state**: Node outputs and intermediate results
- **Memory/Store data**: Information that persists across threads

**Storage locations:**

- **Development mode**: Local directory (configurable)
- **Docker deployment**: Local Docker volumes
- **LangGraph Platform**: Managed database infrastructure

### Remote Storage (Optional)

Remote storage only occurs when explicitly configured:

#### CLI Telemetry (Opt-out)

Minimal analytics data collected by default to improve the tool:

**Collected data:**

- CLI commands used (e.g., `dev`, `up`, `build`)
- CLI version
- Operating system and Python version
- Anonymized parameter usage (boolean flags for non-default options)

**Not collected:**

- Parameter values
- File contents or paths
- Personal information
- Code or graph implementations
- API keys or sensitive data

**To disable:** Set `LANGGRAPH_CLI_NO_ANALYTICS=1`

#### Frontend Telemetry (Opt-out)

LangGraph Studio collects usage analytics to improve the user experience:

**Collected data:**

- Page visits and navigation patterns
- User actions (button clicks)
- Feature usage statistics
- Browser type and version
- Screen resolution and viewport size

**Not collected:**

- Application data or code
- Sensitive configuration details

**To disable:** Use browser privacy settings or ad blockers to prevent analytics collection

#### LangSmith Tracing (Opt-in)

Tracing data is only sent when explicitly configured:

**Requirements:**

- `LANGSMITH_API_KEY` must be set
- `LANGSMITH_TRACING` must not be set to `false`

**Data sent when enabled:**

- Number of runs executed
- API version information
- Execution traces and metadata

**To disable:** Set `LANGSMITH_TRACING=false` or remove `LANGSMITH_API_KEY`

## Security Features

### Encryption

- **Optional encryption**: Available for all persisted state
- **AES encryption**: Uses `EncryptedSerializer`
- **Automatic enablement**: When `LANGGRAPH_AES_KEY` environment variable is set

For more information on encryption, [see here](../../concepts/persistence.md#encryption).

### Data Retention

- **Configurable TTL**: Automatic cleanup of old data
- **Default expiration**: Set in minutes
- **Automatic sweeping**: Removes expired data at configurable intervals

For more information on TTLs, [see here](../../how-tos/ttl/configure_ttl.md).

## Privacy Controls

### Environment Variables

Control data collection and storage behavior:

| Variable                       | Purpose                   | Default                          |
| ------------------------------ | ------------------------- | -------------------------------- |
| `LANGGRAPH_CLI_NO_ANALYTICS=1` | Disable CLI analytics     | Analytics enabled                |
| `LANGGRAPH_AES_KEY`            | Enable data encryption    | Encryption disabled              |
| `LANGSMITH_API_KEY`            | Enable LangSmith tracing  | Tracing disabled                 |
| `LANGSMITH_TRACING=false`      | Disable LangSmith tracing | Tracing enabled (if API key set) |
