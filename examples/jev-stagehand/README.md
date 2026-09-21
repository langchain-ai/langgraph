# Jev browser agent with LangGraph and Stagehand

This example rewrites the core idea behind [Jev Ultrafast](https://github.com/browser-use/jev-ultrafast) with:

- the [LangGraph Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api) for the bounded agent loop;
- [`TypeSafeClassifier`](https://docs.langchain.com/oss/python/integrations/providers/typesafe) for Jev's typed, probabilistic decisions;
- [Stagehand](https://docs.stagehand.dev/v4/reference/stagehand) for browser observation and deterministic actions;
- a small LangChain chat model only when an action needs free-form text.

Jev receives one shared state and answers speculative `Choice` questions for the next operation and compatible targets in parallel. The graph executes only the target head selected by the operation. Stagehand snapshot IDs are resolved to selectors by code, so model output never becomes arbitrary JavaScript or a free-form selector.

## Run it

```bash
uv sync --all-groups
export TYPESAFE_API_KEY=...
export OPENAI_API_KEY=...
uv run python agent.py \
  'https://www.google.com/travel/flights?hl=en' \
  'Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy. Stop when matching flight options are visible.'
```

Use `--headed` to watch the run and `--max-steps` to lower the action budget. `TEXT_MODEL` defaults to `gpt-5.4-mini` and is called only for text-entry actions.

Stagehand launches a fresh temporary Chrome profile by default. Keep it isolated: page content is sent to the configured model providers, and browser agents can make mistakes. The example stops instead of executing an action that Jev classifies as potentially sending, publishing, purchasing, deleting, or otherwise causing an irreversible side effect. Do not use it with credentials or sensitive pages without adding application-specific controls.

## How it maps to LangGraph

- `jev_browser_agent` is the `@entrypoint` and owns the action budget, browser lifecycle, and stop conditions.
- `decide` is a `@task` that calls the LangChain TypeSafe integration.
- `write_field_value` is a `@task` that uses a structured-output chat model because Jev makes decisions but does not generate strings.
- Stagehand's `page.snapshot()` supplies the accessibility tree and the snapshot-ID-to-XPath map; `page.locator()` performs the selected action.

This is an educational example, not a production browser security boundary. Production deployments should additionally restrict navigation domains, require human confirmation for consequential actions, redact traces, and independently verify completion.
