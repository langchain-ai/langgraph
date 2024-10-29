---
hide:
  - navigation
title: Concepts
description: Conceptual Guide for LangGraph
---

# Conceptual Guide

This guide provides explanations of the key concepts behind the LangGraph framework and AI applications more broadly.

We recommend that you go through at least the [Quick Start](../tutorials/introduction.ipynb) before diving into the conceptual guide. This will provide practical context that will make it easier to understand the concepts discussed here.

The conceptual guide does not cover step-by-step instructions or specific implementation examples — those are found in the [Tutorials](../tutorials/index.md) and [How-to guides](../how-tos/index.md). 
For detailed reference material, please see the [API reference](../reference/index.md).

## LangGraph Concepts

**High Level**

- [Why LangGraph?](high_level.md): A high-level overview of LangGraph and its goals.

**Concepts**

- [LangGraph Glossary](low_level.md): LangGraph workflows are designed as graphs, with nodes representing different components and edges representing the flow of information between them. This guide provides an overview of the key concepts associated with LangGraph graph primitives.
- [Common Agentic Patterns](agentic_concepts.md): An agent are LLMs that can pick its own control flow to solve more complex problems! Agents are a key building block in many LLM applications. This guide explains the different types of agent architectures and how they can be used to control the flow of an application.
- [Multi-Agent Systems](multi_agent.md): Complex LLM applications can often be broken down into multiple agents, each responsible for a different part of the application. This guide explains common patterns for building multi-agent systems.
- [Human-in-the-Loop](human_in_the_loop.md): Explains different ways of integrating human feedback into a LangGraph application.
- [Persistence](persistence.md): LangGraph has a built-in persistence layer, implemented through checkpointers. This persistence layer helps to support powerful capabilities like human-in-the-loop, memory, time travel, and fault-tolerance.
- [Memory](memory.md): Memory in AI applications refers to the ability to process, store, and effectively recall information from past interactions. With memory, your agents can learn from feedback and adapt to users' preferences.  
- [Streaming](streaming.md): Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs. 
- [FAQ](faq.md): Frequently asked questions about LangGraph.

## LangGraph Deploy 

LangGraph Deploy is an opinionated deployment framework for agent-based applications, providing an API server tailored for interacting with assistants, with essential features such as streaming, human-in-the-loop interactions, persistence, cron jobs, web hooks, and more. It includes built-in queuing and persistence layers, and an integrated IDE, LangGraph Studio, to streamline development.

The following section describes concepts that appear in the LangGraph Deploy:

**High Level**

- [Why LangGraph Deploy?](./langgraph_platform.md): An overview of what features LangGraph Deploy provides and why you might want to use it to deploy your LangGraph applications.
- [Deployment](./deployment.md): Deployment is the process of making your LangGraph application available to users. This guide goes over general considerations when deploying LangGraph applications.
- [LangGraph Studio](./langgraph_studio.md): LangGraph Studio is a specialized agent IDE that enables visualization, interaction, and debugging of complex agentic applications locally.

**Concepts**

- [Assistants](./assistants.md): Assistants are a way to save and manage different configurations of your LangGraph applications.
- [Double Texting](./double_texting.md): Double texting is a common issue in LLM applications where users may send multiple messages before the graph has finished running. This guide explains how to handle double texting with LangGraph Deploy.