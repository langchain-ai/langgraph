<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
  <img alt="LangGraph Logo" src=".github/images/logo-dark.svg" width="50%">
</picture>

<div>
<br>
</div>

[English](README.md)

[![Version](https://img.shields.io/pypi/v/langgraph.svg)](https://pypi.org/project/langgraph/)
[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://docs.langchain.com/oss/python/langgraph/overview)

受到 Klarna、Replit、Elastic 等引领 AI 未来的公司信赖，LangGraph 是一个用于构建、管理和部署长期运行的有状态 Agent 的底层编排框架。

## 快速开始

安装 LangGraph：

```
pip install -U langgraph
```

创建一个简单的工作流：

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

通过 [LangGraph 快速入门](https://docs.langchain.com/oss/python/langgraph/quickstart) 开始上手。

如需使用 LangChain 的 `create_agent`（基于 LangGraph 构建）快速搭建 Agent，请参阅 [LangChain Agents 文档](https://docs.langchain.com/oss/python/langchain/agents)。

> [!TIP]
> 如需开发、调试和部署 AI Agent 及 LLM 应用，请参阅 [LangSmith](https://docs.langchain.com/langsmith/home)。

## 核心优势

LangGraph 为*任何*长期运行的有状态工作流或 Agent 提供底层基础设施支持。LangGraph 不对提示词或架构进行抽象封装，并提供以下核心优势：

- [持久化执行](https://docs.langchain.com/oss/python/langgraph/durable-execution)：构建能够在故障中持续运行的 Agent，支持长时间运行，并能从中断处自动恢复。
- [人在回路](https://docs.langchain.com/oss/python/langgraph/interrupts)：在执行过程中的任意节点检查和修改 Agent 状态，无缝集成人工监督。
- [全面的记忆管理](https://docs.langchain.com/oss/python/langgraph/memory)：构建真正有状态的 Agent，同时支持用于当前推理的短期工作记忆和跨会话的长期持久记忆。
- [使用 LangSmith 调试](http://www.langchain.com/langsmith)：通过可视化工具深入了解复杂 Agent 行为，追踪执行路径、捕获状态转换并提供详细的运行时指标。
- [生产就绪的部署](https://docs.langchain.com/langsmith/app-development)：借助专为有状态长期运行工作流设计的可扩展基础设施，自信地部署复杂的 Agent 系统。

## LangGraph 生态系统

LangGraph 既可独立使用，也能与任何 LangChain 产品无缝集成，为开发者提供完整的 Agent 构建工具链。为提升 LLM 应用开发效率，可将 LangGraph 与以下产品配合使用：

- [LangSmith](http://www.langchain.com/langsmith) — 适用于 Agent 评估与可观测性。调试表现不佳的 LLM 应用运行、评估 Agent 轨迹、获得生产环境的可见性，并持续改进性能。
- [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments) — 使用专为长期运行有状态工作流打造的部署平台，轻松部署和扩展 Agent。跨团队发现、复用、配置和共享 Agent，并通过 [LangGraph Studio](https://docs.langchain.com/oss/python/langgraph/studio) 中的可视化原型工具快速迭代。
- [LangChain](https://docs.langchain.com/oss/python/langchain/overview) — 提供集成能力和可组合组件，简化 LLM 应用开发流程。

> [!NOTE]
> 寻找 LangGraph 的 JS 版本？请查看 [JS 仓库](https://github.com/langchain-ai/langgraphjs) 和 [JS 文档](https://docs.langchain.com/oss/javascript/langgraph/overview)。

## 更多资源

- [指南](https://docs.langchain.com/oss/python/langgraph/overview)：针对流式处理、添加记忆与持久化、设计模式（如分支、子图等）的快速实用代码片段。
- [参考文档](https://reference.langchain.com/python/langgraph/)：核心类、方法的详细参考，以及图和检查点 API 的使用说明，以及更高层次的预置组件。
- [示例](https://docs.langchain.com/oss/python/langgraph/agentic-rag)：LangGraph 入门的引导式示例。
- [LangChain 论坛](https://forum.langchain.com/)：加入社区，分享技术问题、想法和反馈。
- [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph)：通过我们的免费结构化课程学习 LangGraph 基础知识。
- [案例研究](https://www.langchain.com/built-with-langgraph)：了解行业领导者如何使用 LangGraph 在大规模场景下交付 AI 应用。

## 致谢

LangGraph 的设计灵感来源于 [Pregel](https://research.google/pubs/pub37252/) 和 [Apache Beam](https://beam.apache.org/)。其公共接口的设计参考了 [NetworkX](https://networkx.org/documentation/latest/)。LangGraph 由 LangChain 的创建者 LangChain Inc 开发，但也可以在不使用 LangChain 的情况下独立使用。
