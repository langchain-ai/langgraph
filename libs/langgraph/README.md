<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset="docs/docs/static/wordmark_dark.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/docs/static/wordmark_light.svg">
  <img alt="LangGraph Logo" src="docs/docs/static/wordmark_dark.svg" width="80%">
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

LangGraph — used by Replit, Uber, LinkedIn, GitLab and more — is a low-level orchestration framework for building controllable agents. While langchain provides integrations and composable components to streamline LLM application development, the LangGraph library enables agent orchestration — offering customizable architectures, long-term memory, and human-in-the-loop to reliably handle complex tasks.

```bash
pip install -U langgraph
```

To learn more about how to use LangGraph, check out [the docs](https://langchain-ai.github.io/langgraph/). We show a simple example below of how to create a ReAct agent.

```python
# This code depends on pip install langchain[anthropic]
from langgraph.prebuilt import create_react_agent

def search(query: str):
    """Call to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

agent = create_react_agent("anthropic:claude-3-7-sonnet-latest", tools=[search])
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

## Why use LangGraph?

LangGraph is built for developers who want to build powerful, adaptable AI agents. Developers choose LangGraph for:

- **Reliability and controllability.** Steer agent actions with moderation checks and human-in-the-loop approvals. LangGraph persists context for long-running workflows, keeping your agents on course.
- **Low-level and extensible.** Build custom agents with fully descriptive, low-level primitives – free from rigid abstractions that limit customization. Design scalable multi-agent systems, with each agent serving a specific role tailored to your use case.
- **First-class streaming support.** With token-by-token streaming and streaming of intermediate steps, LangGraph gives users clear visibility into agent reasoning and actions as they unfold in real time.

LangGraph is trusted in production and powering agents for companies like:

- [Klarna](https://blog.langchain.dev/customers-klarna/): Customer support bot for 85 million active users
- [Elastic](https://www.elastic.co/blog/elastic-security-generative-ai-features): Security AI assistant for threat detection
- [Uber](https://dpe.org/sessions/ty-smith-adam-huda/this-year-in-ubers-ai-driven-developer-productivity-revolution/): Automated unit test generation
- [Replit](https://www.langchain.com/breakoutagents/replit): Code generation
- And many more ([see list here](https://www.langchain.com/built-with-langgraph))

## LangGraph’s ecosystem

While LangGraph can be used standalone, it also integrates seamlessly with any LangChain product, giving developers a full suite of tools for building agents. To improve your LLM application development, pair LangGraph with:

- [LangSmith](http://www.langchain.com/langsmith) — Helpful for agent evals and observability. Debug poor-performing LLM app runs, evaluate agent trajectories, gain visibility in production, and improve performance over time.
- [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#langgraph-platform) — Deploy and scale agents effortlessly with a purpose-built deployment platform for long running, stateful workflows. Discover, reuse, configure, and share agents across teams — and iterate quickly with visual prototyping in [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/).

## Pairing with LangGraph Platform

While LangGraph is our open-source agent orchestration framework, enterprises that need scalable agent deployment can benefit from [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/).

LangGraph Platform can help engineering teams:

- **Accelerate agent development**: Quickly create agent UXs with configurable templates and [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) for visualizing and debugging agent interactions.
- **Deploy seamlessly**: We handle the complexity of deploying your agent. LangGraph Platform includes robust APIs for memory, threads, and cron jobs plus auto-scaling task queues & servers.
- **Centralize agent management & reusability**: Discover, reuse, and manage agents across the organization. Business users can also modify agents without coding.

## Additional resources

- [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph): Learn the basics of LangGraph in our free, structured course.
- [Tutorials](https://langchain-ai.github.io/langgraph/tutorials/): Simple walkthroughs with guided examples on getting started with LangGraph.
- [Templates](https://langchain-ai.github.io/langgraph/concepts/template_applications/): Pre-built reference apps for common agentic workflows (e.g. ReAct agent, memory, retrieval etc.) that can be cloned and adapted.
- [How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/): Quick, actionable code snippets for topics such as streaming, adding memory & persistence, and design patterns (e.g. branching, subgraphs, etc.).
- [API Reference](https://langchain-ai.github.io/langgraph/reference/graphs/): Detailed reference on core classes, methods, how to use the graph and checkpointing APIs, and higher-level prebuilt components.
- [Built with LangGraph](https://www.langchain.com/built-with-langgraph): Hear how industry leaders use LangGraph to ship powerful, production-ready AI applications.

## Acknowledgements

LangGraph is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/). LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.