# Template Applications

Templates are open source reference applications designed to help you get started quickly when building with LangGraph. They provide working examples of common agentic workflows that can be customized to your needs.

You can create an application from a template using the LangGraph CLI.


## Available templates

| Template                  | Description                                                                              | Python                                                           | JS/TS                                                               |
|---------------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------|---------------------------------------------------------------------|
| **New LangGraph Project** | A simple, minimal chatbot with memory.                                                   | [Repo](https://github.com/langchain-ai/new-langgraph-project)    | [Repo](https://github.com/langchain-ai/new-langgraphjs-project)     |
| **ReAct Agent**           | A simple agent that can be flexibly extended to many tools.                              | [Repo](https://github.com/langchain-ai/react-agent)              | [Repo](https://github.com/langchain-ai/react-agent-js)              |
| **Memory Agent**          | A ReAct-style agent with an additional tool to store memories for use across threads.    | [Repo](https://github.com/langchain-ai/memory-agent)             | [Repo](https://github.com/langchain-ai/memory-agent-js)             |
| **Retrieval Agent**       | An agent that includes a retrieval-based question-answering system.                      | [Repo](https://github.com/langchain-ai/retrieval-agent-template) | [Repo](https://github.com/langchain-ai/retrieval-agent-template-js) |
| **Data-Enrichment Agent** | An agent that performs web searches and organizes its findings into a structured format. | [Repo](https://github.com/langchain-ai/data-enrichment)          | [Repo](https://github.com/langchain-ai/data-enrichment-js)          |



## 🌱 Create a LangGraph App

To create a new app from a template, use the `langgraph new` command. This command will create a new directory with the specified template.

```shell

This is a quick start guide to help you get a LangGraph app up and running locally.

!!! info "Requirements"

    - [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/): Requires langchain-cli[inmem] >= 0.1.58

## Install the LangGraph CLI

```bash
pip install "langgraph-cli[inmem]==0.1.58" python-dot-env
```



