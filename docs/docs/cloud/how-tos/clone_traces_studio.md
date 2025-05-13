# Debug LangSmith traces

This guide explains how to open LangSmith traces in LangGraph Studio for interactive investigation and debugging.

## Open deployed threads

1. Open the LangSmith trace, selecting the root run.
2. Click "Run in Studio".

This will open LangGraph Studio connected to the associated LangGraph Platform deployment with the trace's parent thread selected.

## Testing local agents with remote traces

This section explains how to test a local agent against remote traces from LangSmith. This enables you to use production traces as input for local testing, allowing you to debug and verify agent modifications in your development environment.

### Requirements

- A LangSmith traced thread
- A locally running agent. See [here](../how-tos/studio/quick_start.md#local-development-server) for setup
  instructions.

!!! info "Local agent requirements"

    - langgraph>=0.3.18
    - langgraph-api>=0.0.32
    - Contains the same set of nodes present in the remote trace

### Cloning Thread

1. Open the LangSmith trace, selecting the root run.
2. Click the dropdown next to "Run in Studio".
3. Enter your local agent's URL.
4. Select "Clone thread locally".
5. If multiple graphs exist, select the target graph.

A new thread will be created in your local agent with the thread history inferred and copied from the remote thread, and you will be navigated to LangGraph Studio for your locally running application.
