<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset="https://langchain-ai.github.io/langgraph/static/wordmark_dark.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://langchain-ai.github.io/langgraph/static/wordmark_light.svg">
  <img alt="LangGraph Logo" src="https://langchain-ai.github.io/langgraph/static/wordmark_dark.svg" width="80%">
</picture>

<div>
<br>
</div>

[![Version](https://img.shields.io/pypi/v/langgraph.svg)](https://pypi.org/project/langgraph/)
[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://langchain-ai.github.io/langgraph/)

> [!NOTE]
> Looking for the JS version? See the [JS repo](https://github.com/langchain-ai/langgraphjs) and the [JS docs](https://langchain-ai.github.io/langgraphjs/).

LangGraph is a low-level orchestration framework for building controllable agents. While [LangChain](https://python.langchain.com/docs/introduction/) provides integrations and composable components to streamline LLM application development, the LangGraph library enables agent orchestration — offering customizable architectures, long-term memory, and human-in-the-loop to reliably handle complex tasks.

## Get started

First, install LangGraph:

```
pip install -U langgraph
```

There are several main ways to get started with LangGraph:

- [Use LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/): Customize your architectures, use long-term memory, and implement human-in-the-loop to reliably handle complex tasks.
- [Use prebuilt components](https://langchain-ai.github.io/langgraph/agents/overview/): Construct agentic systems quickly and reliably—without the need to implement orchestration, memory, or human feedback handling from scratch.
- [Use LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/): Test, debug, and deploy production applications.

## What LangGraph provides

LangGraph provides low-level supporting infrastructure that sits underneath *any* workflow or agent. It does not abstract prompts or architecture, and provides three central benefits:

### Persistence

LangGraph has a [persistence layer](https://langchain-ai.github.io/langgraph/concepts/persistence/), which offers a number of benefits:

- [Memory](https://langchain-ai.github.io/langgraph/concepts/memory/): LangGraph persists arbitrary aspects of your application's state, supporting memory of conversations and other updates within and across user interactions;
- [Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/): Because state is checkpointed, execution can be interrupted and resumed, allowing for decisions, validation, and corrections via human input.

### Streaming

LangGraph provides support for [streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/) workflow / agent state to the user (or developer) over the course of execution. LangGraph supports streaming of both events ([such as feedback from a tool call](https://langchain-ai.github.io/langgraph/how-tos/streaming.ipynb#updates)) and [tokens from LLM calls](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens.ipynb) embedded in an application.

### Debugging and deployment

LangGraph provides an easy onramp for testing, debugging, and deploying applications via [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/). This includes [Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), an IDE that enables visualization, interaction, and debugging of workflows or agents. This also includes numerous [options](https://langchain-ai.github.io/langgraph/tutorials/deployment/) for deployment. 

## LangGraph’s ecosystem

While LangGraph can be used standalone, it also integrates seamlessly with any LangChain product, giving developers a full suite of tools for building agents. To improve your LLM application development, pair LangGraph with:

- [LangSmith](http://www.langchain.com/langsmith) — Helpful for agent evals and observability. Debug poor-performing LLM app runs, evaluate agent trajectories, gain visibility in production, and improve performance over time.
- [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#langgraph-platform) — Deploy and scale agents effortlessly with a purpose-built deployment platform for long running, stateful workflows. Discover, reuse, configure, and share agents across teams — and iterate quickly with visual prototyping in [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/).

## Additional resources

- [Guides](https://langchain-ai.github.io/langgraph/how-tos/): Quick, actionable code snippets for topics such as streaming, adding memory & persistence, and design patterns (e.g. branching, subgraphs, etc.).
- [Reference](https://langchain-ai.github.io/langgraph/reference/graphs/): Detailed reference on core classes, methods, how to use the graph and checkpointing APIs, and higher-level prebuilt components.
- [Examples](https://langchain-ai.github.io/langgraph/tutorials/): Guided examples on getting started with LangGraph.
- [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph): Learn the basics of LangGraph in our free, structured course.
- [Templates](https://langchain-ai.github.io/langgraph/concepts/template_applications/): Pre-built reference apps for common agentic workflows (e.g. ReAct agent, memory, retrieval etc.) that can be cloned and adapted.
- [Case studies](https://www.langchain.com/built-with-langgraph): Hear how industry leaders use LangGraph to ship powerful, production-ready AI applications.

## Acknowledgements

LangGraph is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/). LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.