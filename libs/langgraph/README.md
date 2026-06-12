# 🦜🕸️ LangGraph

[![PyPI - Version](https://img.shields.io/pypi/v/langgraph?label=%20)](https://pypi.org/project/langgraph/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langgraph)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langgraph)](https://pypistats.org/packages/langgraph)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

Looking for the JS/TS version? Check out [LangGraph.js](https://github.com/langchain-ai/langgraphjs).

To help you ship LangGraph apps to production faster, check out [LangSmith](https://www.langchain.com/langsmith).
[LangSmith](https://www.langchain.com/langsmith) is a unified developer platform for building, testing, and monitoring LLM applications.

## Quick Install

```bash
uv add langgraph
```

## 🤔 What is this?

LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents. LangGraph provides the infrastructure for durable execution, streaming, human-in-the-loop, persistence, memory, and more.

We recommend you use LangGraph when you have advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency. Use [LangChain](https://docs.langchain.com/oss/python/langchain/overview) when you want to quickly build agents and applications powered by LLMs using pre-built agent architectures and model integrations.

LangChain [agents](https://docs.langchain.com/oss/python/langchain/agents) are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. (You do not need to know LangGraph for basic LangChain agent usage.)

Trusted by companies shaping the future of agents – including Klarna, Replit, Elastic, and more – LangGraph is used to ship AI applications at scale.

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/langgraph/). For conceptual guides, tutorials, and examples on using LangGraph, see the [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/overview). Get started with the [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart).

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## Acknowledgements

LangGraph is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/). LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.
