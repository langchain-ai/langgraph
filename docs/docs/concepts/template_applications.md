---
search:
  boost: 2
---

# Template Applications

Templates are open source reference applications designed to help you get started quickly when building with LangGraph. They provide working examples of common agentic workflows that can be customized to your needs.

You can create an application from a template using the LangGraph CLI.

:::python
!!! info "Requirements"

    - Python >= 3.11
    - [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/): Requires langchain-cli[inmem] >= 0.1.58

## Install the LangGraph CLI

```bash
pip install "langgraph-cli[inmem]" --upgrade
```

Or via [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (recommended):

```bash
uvx --from "langgraph-cli[inmem]" langgraph dev --help
```

:::

:::js

```bash
npx @langchain/langgraph-cli --help
```

:::

## Available Templates

:::python
| Template | Description | Link |
| -------- | ----------- | ------ |
| **New LangGraph Project** | A simple, minimal chatbot with memory. | [Repo](https://github.com/langchain-ai/new-langgraph-project) |
| **ReAct Agent** | A simple agent that can be flexibly extended to many tools. | [Repo](https://github.com/langchain-ai/react-agent) |
| **Memory Agent** | A ReAct-style agent with an additional tool to store memories for use across threads. | [Repo](https://github.com/langchain-ai/memory-agent) |
| **Retrieval Agent** | An agent that includes a retrieval-based question-answering system. | [Repo](https://github.com/langchain-ai/retrieval-agent-template) |
| **Data-Enrichment Agent** | An agent that performs web searches and organizes its findings into a structured format. | [Repo](https://github.com/langchain-ai/data-enrichment) |

:::

:::js
| Template | Description | Link |
| -------- | ----------- | ------ |
| **New LangGraph Project** | A simple, minimal chatbot with memory. | [Repo](https://github.com/langchain-ai/new-langgraphjs-project) |
| **ReAct Agent** | A simple agent that can be flexibly extended to many tools. | [Repo](https://github.com/langchain-ai/react-agent-js) |
| **Memory Agent** | A ReAct-style agent with an additional tool to store memories for use across threads. | [Repo](https://github.com/langchain-ai/memory-agent-js) |
| **Retrieval Agent** | An agent that includes a retrieval-based question-answering system. | [Repo](https://github.com/langchain-ai/retrieval-agent-template-js) |
| **Data-Enrichment Agent** | An agent that performs web searches and organizes its findings into a structured format. | [Repo](https://github.com/langchain-ai/data-enrichment-js) |
:::

## 🌱 Create a LangGraph App

To create a new app from a template, use the `langgraph new` command.

:::python

```bash
langgraph new
```

Or via [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (recommended):

```bash
uvx --from "langgraph-cli[inmem]" langgraph new
```

:::

:::js

```bash
npm create langgraph
```

:::

## Next Steps

Review the `README.md` file in the root of your new LangGraph app for more information about the template and how to customize it.

After configuring the app properly and adding your API keys, you can start the app using the LangGraph CLI:

:::python

```bash
langgraph dev
```

Or via [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (recommended):

```bash
uvx --from "langgraph-cli[inmem]" --with-editable . langgraph dev
```

!!! info "Missing Local Package?"

    If you are not using `uv` and run into a "`ModuleNotFoundError`" or "`ImportError`", even after installing the local package (`pip install -e .`), it is likely the case that you need to install the CLI into your local virtual environment to make the CLI "aware" of the local package. You can do this by running `python -m pip install "langgraph-cli[inmem]"` and re-activating your virtual environment before running `langgraph dev`.

:::

:::js

```bash
npx @langchain/langgraph-cli dev
```

:::

See the following guides for more information on how to deploy your app:

- **[Launch Local LangGraph Server](../tutorials/langgraph-platform/local-server.md)**: This quick start guide shows how to start a LangGraph Server locally for the **ReAct Agent** template. The steps are similar for other templates.
- **[Deploy to LangGraph Platform](../cloud/quick_start.md)**: Deploy your LangGraph app using LangGraph Platform.
