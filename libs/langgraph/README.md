<div align="center">
  <a href="https://www.langchain.com/langgraph">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="../../.github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="../../.github/images/logo-light.svg">
      <img alt="LangGraph Logo" src=".github/images/logo-dark.svg" width="50%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>Low-level orchestration framework for building stateful agents.</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/pypi/l/langgraph" alt="PyPI - License"></a>
  <a href="https://pypistats.org/packages/langgraph" target="_blank"><img src="https://img.shields.io/pepy/dt/langgraph" alt="PyPI - Downloads"></a>
  <a href="https://pypi.org/project/langgraph/" target="_blank"><img src="https://img.shields.io/pypi/v/langgraph.svg?label=%20" alt="Version"></a>
  <a href="https://github.com/langchain-ai/langgraph/issues" target="_blank"><img src="https://img.shields.io/github/issues-raw/langchain-ai/langgraph" alt="Open Issues"></a>
  <a href="https://docs.langchain.com/oss/python/langgraph/overview" target="_blank"><img src="https://img.shields.io/badge/docs-latest-blue" alt="Docs"></a>
  <a href="https://x.com/langchain" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain" alt="Twitter / X"></a>
</div>

<br>

Trusted by companies shaping the future of agents – including Klarna, Replit, Elastic, and more – LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents.

## Get started

Install LangGraph:

```
pip install -U langgraph
```

Create a simple workflow:

```python
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    text: str


def node_a(state: State) -> dict:
    return {"text": state["text"] + "a"}


def node_b(state: State) -> dict:
    return {"text": state["text"] + "b"}


graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")

print(graph.compile().invoke({"text": ""}))
# {'text': 'ab'}
```

Get started with the [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart).

To quickly build agents with LangChain's `create_agent` (built on LangGraph), see the [LangChain Agents documentation](https://docs.langchain.com/oss/python/langchain/agents).

## Core benefits

LangGraph provides low-level supporting infrastructure for *any* long-running, stateful workflow or agent. LangGraph does not abstract prompts or architecture, and provides the following central benefits:

- [Durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution): Build agents that persist through failures and can run for extended periods, automatically resuming from exactly where they left off.
- [Human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/interrupts): Seamlessly incorporate human oversight by inspecting and modifying agent state at any point during execution.
- [Comprehensive memory](https://docs.langchain.com/oss/python/langgraph/memory): Create truly stateful agents with both short-term working memory for ongoing reasoning and long-term persistent memory across sessions.
- [Debugging with LangSmith](http://www.langchain.com/langsmith): Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.
- [Production-ready deployment](https://docs.langchain.com/langsmith/app-development): Deploy sophisticated agent systems confidently with scalable infrastructure designed to handle the unique challenges of stateful, long-running workflows.

## LangGraph’s ecosystem

While LangGraph can be used standalone, it also integrates seamlessly with any LangChain product, giving developers a full suite of tools for building agents. To improve your LLM application development, pair LangGraph with:

- [LangSmith](http://www.langchain.com/langsmith) — Helpful for agent evals and observability. Debug poor-performing LLM app runs, evaluate agent trajectories, gain visibility in production, and improve performance over time.
- [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments) — Deploy and scale agents effortlessly with a purpose-built deployment platform for long running, stateful workflows. Discover, reuse, configure, and share agents across teams — and iterate quickly with visual prototyping in [LangGraph Studio](https://docs.langchain.com/oss/python/langgraph/studio).
- [LangChain](https://docs.langchain.com/oss/python/langchain/overview) – Provides integrations and composable components to streamline LLM application development.

> [!NOTE]
> Looking for the JS version of LangGraph? See the [JS repo](https://github.com/langchain-ai/langgraphjs) and the [JS docs](https://docs.langchain.com/oss/javascript/langgraph/overview).

## Additional resources

- [Guides](https://docs.langchain.com/oss/python/langgraph/guides): Quick, actionable code snippets for topics such as streaming, adding memory & persistence, and design patterns (e.g. branching, subgraphs, etc.).
- [Reference](https://reference.langchain.com/python/langgraph/): Detailed reference on core classes, methods, how to use the graph and checkpointing APIs, and higher-level prebuilt components.
- [Examples](https://docs.langchain.com/oss/python/langgraph/agentic-rag): Guided examples on getting started with LangGraph.
- [LangChain Forum](https://forum.langchain.com/): Connect with the community and share all of your technical questions, ideas, and feedback.
- [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph): Learn the basics of LangGraph in our free, structured course.
- [Case studies](https://www.langchain.com/built-with-langgraph): Hear how industry leaders use LangGraph to ship AI applications at scale.

## Acknowledgements

LangGraph is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/). LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.
