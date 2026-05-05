# CLI Graph Examples

This directory contains LangGraph CLI example graphs declared in
`langgraph.json`.

## Optional release-readiness scan

The included `shipgate.yaml` and `tool-inventory.json` let you run an advisory
static scan of the example tool surface before adapting these graphs for
production-like use. The inventory documents the search tool used by the dynamic
LangChain `tools` binding in `agent.py`.

```bash
pipx install agents-shipgate
agents-shipgate scan -c libs/cli/examples/graphs/shipgate.yaml --ci-mode advisory
```

Agents Shipgate reads local source and manifest files only. It does not execute
the graph, call an LLM, call tools, connect to MCP servers, make scanner network
calls, or collect telemetry.

The scan is advisory for these examples. It is intended to show which
release-review metadata would need attention before wiring similar graphs to
real external tools or production credentials.
