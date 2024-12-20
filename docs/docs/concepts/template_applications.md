# Template Applications

Templates are open source reference applications designed to help you get started quickly when building with LangGraph. They provide working examples of common agentic workflows that can be customized to your needs.

You can create an application from a template using the LangGraph CLI.

!!! info "Requirements"

    - Python >= 3.11
    - [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/): Requires langchain-cli[inmem] >= 0.1.58

## Install the LangGraph CLI

```bash
pip install "langgraph-cli[inmem]" --upgrade
```

## Available Templates

| Template                  | Description                                                                              | Python                                                           | JS/TS                                                               |
|---------------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------|---------------------------------------------------------------------|
| **New LangGraph Project** | A simple, minimal chatbot with memory.                                                   | [Repo](https://github.com/langchain-ai/new-langgraph-project)    | [Repo](https://github.com/langchain-ai/new-langgraphjs-project)     |
| **ReAct Agent**           | A simple agent that can be flexibly extended to many tools.                              | [Repo](https://github.com/langchain-ai/react-agent)              | [Repo](https://github.com/langchain-ai/react-agent-js)              |
| **Memory Agent**          | A ReAct-style agent with an additional tool to store memories for use across threads.    | [Repo](https://github.com/langchain-ai/memory-agent)             | [Repo](https://github.com/langchain-ai/memory-agent-js)             |
| **Retrieval Agent**       | An agent that includes a retrieval-based question-answering system.                      | [Repo](https://github.com/langchain-ai/retrieval-agent-template) | [Repo](https://github.com/langchain-ai/retrieval-agent-template-js) |
| **Data-Enrichment Agent** | An agent that performs web searches and organizes its findings into a structured format. | [Repo](https://github.com/langchain-ai/data-enrichment)          | [Repo](https://github.com/langchain-ai/data-enrichment-js)          |


## 🌱 Create a LangGraph App

To create a new app from a template, use the `langgraph new` command.

```bash
langgraph new
```

## Next Steps

Review the `README.md` file in the root of your new LangGraph app for more information about the template and how to customize it.

After configuring the app properly and adding your API keys, you can start the app using the LangGraph CLI:

```bash
langgraph dev 
```

See the following guides for more information on how to deploy your app:

- **[Launch Local LangGraph Server](../tutorials/langgraph-platform/local-server.md)**: This quick start guide shows how to start a LangGraph Server locally for the **ReAct Agent** template. The steps are similar for other templates.
- **[Deploy to LangGraph Cloud](../cloud/quick_start.md)**: Deploy your LangGraph app using LangGraph Cloud.
 
### LangGraph Framework

- **[LangGraph Concepts](../concepts/index.md)**: Learn the foundational concepts of LangGraph.
- **[LangGraph How-to Guides](../how-tos/index.md)**: Guides for common tasks with LangGraph.

### 📚 Learn More about LangGraph Platform

Expand your knowledge with these resources:

- **[LangGraph Platform Concepts](../concepts/index.md#langgraph-platform)**: Understand the foundational concepts of the LangGraph Platform.
- **[LangGraph Platform How-to Guides](../how-tos/index.md#langgraph-platform)**: Discover step-by-step guides to build and deploy applications.
