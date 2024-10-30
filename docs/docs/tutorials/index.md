---
hide:
  - navigation
title: Tutorials
---

# Tutorials

Welcome to the LangGraph Tutorials! These notebooks introduce LangGraph through building various language agents and applications.

## Quick Start

Learn the basics of LangGraph through a comprehensive quick start in which you will build an agent from scratch.

- [Quick Start](introduction.ipynb): In this tutorial, you will build a support chatbot using LangGraph.
- [LangGraph Platform Quick Start](../cloud/quick_start.md): In this tutorial, you will learn how to deploy a tool-calling chatbot agent to LangGraph Platform.

## Use cases

Learn from example implementations of graphs designed for specific scenarios and that implement common design patterns.

### Chatbots

- [Customer Support](customer-support/customer-support.ipynb): Build a customer support chatbot to manage flights, hotel reservations, car rentals, and other tasks
- [Prompt Generation from User Requirements](chatbots/information-gather-prompting.ipynb): Build an information gathering chatbot
- [Code Assistant](code_assistant/langgraph_code_assistant.ipynb): Build a code analysis and generation assistant


### RAG

- [Agentic RAG](rag/langgraph_agentic_rag.ipynb): Use an agent to figure out how to retrieve the most relevant information before using the retrieved information to answer the user's question.
- [Adaptive RAG](rag/langgraph_adaptive_rag.ipynb): Adaptive RAG is a strategy for RAG that unites (1) query analysis with (2) active / self-corrective RAG. Implementation of: https://arxiv.org/abs/2403.14403
    - For a version that uses a local LLM: [Adaptive RAG using local LLMs](rag/langgraph_adaptive_rag_local.ipynb)
- [Corrective RAG](rag/langgraph_crag.ipynb): Uses an LLM to grade the quality of the retrieved information from the given source, and if the quality is low, it will try to retrieve the information from another source. Implementation of: https://arxiv.org/pdf/2401.15884.pdf 
    - For a version that uses a local LLM: [Corrective RAG using local LLMs](rag/langgraph_crag_local.ipynb)
- [Self-RAG](rag/langgraph_self_rag.ipynb): Self-RAG is a strategy for RAG that incorporates self-reflection / self-grading on retrieved documents and generations. Implementation of https://arxiv.org/abs/2310.11511.
    - For a version that uses a local LLM: [Self-RAG using local LLMs](rag/langgraph_self_rag_local.ipynb) 
- [SQL Agent](sql-agent.ipynb): Build a SQL agent that can answer questions about a SQL database.


### Agent Architectures

#### Multi-Agent Systems

- [Network](multi_agent/multi-agent-collaboration.ipynb): Enable two or more agents to collaborate on a task
- [Supervisor](multi_agent/agent_supervisor.ipynb): Use an LLM to orchestrate and delegate to individual agents
- [Hierarchical Teams](multi_agent/hierarchical_agent_teams.ipynb): Orchestrate nested teams of agents to solve problems
 
#### Planning Agents

- [Plan-and-Execute](plan-and-execute/plan-and-execute.ipynb): Implement a basic planning and execution agent
- [Reasoning without Observation](rewoo/rewoo.ipynb): Reduce re-planning by saving observations as variables
- [LLMCompiler](llm-compiler/LLMCompiler.ipynb): Stream and eagerly execute a DAG of tasks from a planner

#### Reflection & Critique 

- [Basic Reflection](reflection/reflection.ipynb): Prompt the agent to reflect on and revise its outputs
- [Reflexion](reflexion/reflexion.ipynb): Critique missing and superfluous details to guide next steps
- [Language Agent Tree Search](lats/lats.ipynb): Use reflection and rewards to drive a tree search over agents
- [Self-Discover Agent](self-discover/self-discover.ipynb): Analyze an agent that learns about its own capabilities

### Evaluation

- [Agent-based](chatbot-simulation-evaluation/agent-simulation-evaluation.ipynb): Evaluate chatbots via simulated user interactions
- [In LangSmith](chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation.ipynb): Evaluate chatbots in LangSmith over a dialog dataset

### Experimental

- [Web Research (STORM)](storm/storm.ipynb): Generate Wikipedia-like articles via research and multi-perspective QA
- [TNT-LLM](tnt-llm/tnt-llm.ipynb): Build rich, interpretable taxonomies of user intentand using the classification system developed by Microsoft for their Bing Copilot application.
- [Web Navigation](web-navigation/web_voyager.ipynb): Build an agent that can navigate and interact with websites
- [Competitive Programming](usaco/usaco.ipynb): Build an agent with few-shot "episodic memory" and human-in-the-loop collaboration to solve problems from the USA Computing Olympiad; adapted from the ["Can Language Models Solve Olympiad Programming?"](https://arxiv.org/abs/2404.10952v1) paper by Shi, Tang, Narasimhan, and Yao.
- [Complex data extraction](extraction/retries.ipynb): Build an agent that can use function calling to do complex extraction tasks
