# Files

- [Command and Send](command-and-send.md) - Primitives for dynamic routing, error recovery, and human-in-the-loop workflows—Send for fan-out task dispatch, Command for multi-directional control including resume and graph navigation.
- [Functional API](functional-api.md) - The @entrypoint and @task decorators for function-based graph composition, supporting retry, cache, and timeout policies with streamlined task invocation.
- [Managed Values](managed-values.md) - Special state fields auto-populated by the LangGraph runtime before node execution. Managed values provide deterministic metadata (like execution position) without version tracking.
- [Subgraphs and Nesting](subgraphs-and-nesting.md) - Composing LangGraph applications by nesting compiled graphs as nodes, with schema adaptation, isolated state, and independent checkpointing enabling modular workflows and multi-agent coordination.
