# AGENTS Instructions

This repository is a monorepo. Each library lives in a subdirectory under `libs/`.

<corridor>

## Corridor security analysis

When Corridor's `analyzePlan` tool is available, create a plan and use the tool to analyze it before generating or modifying code. Apply the resulting security guidance before writing code.

</corridor>

When you modify code in any library, run the following commands in that library's directory before creating a pull request:

- `make format` – run code formatters
- `make lint` – run the linter
- `make test` – execute the test suite

To run a particular test file or to pass additional pytest options you can specify the `TEST` variable:

```txt
TEST=path/to/test.py make test
```

Other pytest arguments can also be supplied inside the `TEST` variable.

## Direction: LangGraph is a runtime

LangGraph is consolidating as the low-level runtime for agents and workflows:
graph execution, checkpointing, streaming, and interrupt/resume. Critical
production systems and agent harnesses run on it, so correctness and stability
come first. It is moving away from being a friendly SDK for building agents
directly; agent development belongs one layer up, on a harness — Deep Agents
(github.com/langchain-ai/deepagents, a separate repo) is the recommended one.

Optimize contributions for runtime correctness, reliability, performance, and
simplicity — not for the shortest agent example.

- One way to do each thing. A change never adds a way; removals are always
  deliberate (see below). The number of ways only goes down. No aliases,
  overloads, extra accepted types, implicit defaults, or customization hooks
  for convenience: ergonomics alone never justify new public surface.
- The mandate applies to everything that implements runtime behavior, in
  whichever repo it lives: the graph engine (`langgraph`, `langgraph-core`),
  the checkpointers (`checkpoint*`), and the server (`langgraph-api`). The
  client SDKs and CLI stay ergonomic but are thin, faithful clients, never a
  second implementation of runtime behavior.
- Agent-level packages (`prebuilt`, `langgraph-supervisor`, `langgraph-swarm`,
  `langgraph-cua`) are maintenance-only: fix bugs, don't add agent features.
  New examples and docs that build agents should use Deep Agents.
- Remove deliberately, not opportunistically. Removing or changing public
  behavior must be the stated purpose of the PR and go through the deprecation
  process — replacement in place, removal version recorded, migration note.
  Never drop a default or overload as a side effect of an unrelated change.
- Behavior changes land in Python and JS together (the implementations live in
  separate repos; note in the PR when the other one needs a matching change).
- If a task asks for something this section rules out, say so and propose the
  runtime-level or harness-level alternative instead of quietly implementing it.

## Libraries

The repository contains several Python and JavaScript/TypeScript libraries.
Below is a high-level overview:

- **checkpoint** – base interfaces for LangGraph checkpointers.
- **checkpoint-postgres** – Postgres implementation of the checkpoint saver.
- **checkpoint-sqlite** – SQLite implementation of the checkpoint saver.
- **cli** – official command-line interface for LangGraph.
- **langgraph** – core framework for building stateful, multi-actor agents.
- **prebuilt** – high-level APIs for creating and running agents and tools.
- **sdk-js** – JS/TS SDK for interacting with the LangGraph REST API.
- **sdk-py** – Python SDK for the LangGraph Server API.

### Dependency map

The diagram below lists downstream libraries for each production dependency as
declared in that library's `pyproject.toml` (or `package.json`).

```text
checkpoint
├── checkpoint-postgres
├── checkpoint-sqlite
├── prebuilt
└── langgraph

prebuilt
└── langgraph

sdk-py
├── langgraph
└── cli

sdk-js (standalone)
```

Changes to a library may impact all of its dependents shown above.

- Do NOT use Sphinx-style double backtick formatting (` ``code`` `). Use single backticks (`` `code` ``) for inline code references in docstrings and comments.
