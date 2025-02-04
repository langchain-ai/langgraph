---
title: Tutorials
---

# Tutorials

New to LangGraph or LLM app development? Read this material to get up and running building your first applications.

## Get Started 🚀 {#quick-start}

- [LangGraph Quickstart](introduction.md): Build a chatbot that can use tools and keep track of conversation history. Add human-in-the-loop capabilities and explore how time-travel works.
- [Common Workflows](workflows/index.md): Overview of the most common workflows using LLMs implemented with LangGraph.
- [LangGraph Server Quickstart](langgraph-platform/local-server.md): Launch a LangGraph server locally and interact with it using REST API and LangGraph Studio Web UI.
- [LangGraph Template Quickstart](../concepts/template_applications.md): Start building with LangGraph Platform using a template application.
- [Deploy with LangGraph Cloud Quickstart](../cloud/quick_start.md): Deploy a LangGraph app using LangGraph Cloud.

## Use cases 🛠️ {#use-cases}


Explore practical implementations tailored for specific scenarios:

### Chatbots

- [Customer Support](customer-support/customer-support.md): Build a multi-functional support bot for flights, hotels, and car rentals.
- [Prompt Generation from User Requirements](chatbots/information-gather-prompting.md): Build an information gathering chatbot.
- [Code Assistant](code_assistant/langgraph_code_assistant.md): Build a code analysis and generation assistant.

### RAG

- [Agentic RAG](rag/langgraph_agentic_rag.md): Use an agent to figure out how to retrieve the most relevant information before using the retrieved information to answer the user's question.
- [Adaptive RAG](rag/langgraph_adaptive_rag.md): Adaptive RAG is a strategy for RAG that unites (1) query analysis with (2) active / self-corrective RAG. Implementation of: https://arxiv.org/abs/2403.14403
    - For a version that uses a local LLM: [Adaptive RAG using local LLMs](rag/langgraph_adaptive_rag_local.md)
- [Corrective RAG](rag/langgraph_crag.md): Uses an LLM to grade the quality of the retrieved information from the given source, and if the quality is low, it will try to retrieve the information from another source. Implementation of: https://arxiv.org/pdf/2401.15884.pdf 
    - For a version that uses a local LLM: [Corrective RAG using local LLMs](rag/langgraph_crag_local.md)
- [Self-RAG](rag/langgraph_self_rag.md): Self-RAG is a strategy for RAG that incorporates self-reflection / self-grading on retrieved documents and generations. Implementation of https://arxiv.org/abs/2310.11511.
    - For a version that uses a local LLM: [Self-RAG using local LLMs](rag/langgraph_self_rag_local.md) 
- [SQL Agent](sql-agent.md): Build a SQL agent that can answer questions about a SQL database.


### Agent Architectures

#### Multi-Agent Systems

- [Network](multi_agent/multi-agent-collaboration.md): Enable two or more agents to collaborate on a task
- [Supervisor](multi_agent/agent_supervisor.md): Use an LLM to orchestrate and delegate to individual agents
- [Hierarchical Teams](multi_agent/hierarchical_agent_teams.md): Orchestrate nested teams of agents to solve problems
 
#### Planning Agents

- [Plan-and-Execute](plan-and-execute/plan-and-execute.md): Implement a basic planning and execution agent
- [Reasoning without Observation](rewoo/rewoo.md): Reduce re-planning by saving observations as variables
- [LLMCompiler](llm-compiler/LLMCompiler.md): Stream and eagerly execute a DAG of tasks from a planner

#### Reflection & Critique 

- [Basic Reflection](reflection/reflection.md): Prompt the agent to reflect on and revise its outputs
- [Reflexion](reflexion/reflexion.md): Critique missing and superfluous details to guide next steps
- [Tree of Thoughts](tot/tot.md): Search over candidate solutions to a problem using a scored tree
- [Language Agent Tree Search](lats/lats.md): Use reflection and rewards to drive a monte-carlo tree search over agents
- [Self-Discover Agent](self-discover/self-discover.md): Analyze an agent that learns about its own capabilities

### Evaluation

- [Agent-based](chatbot-simulation-evaluation/agent-simulation-evaluation.md): Evaluate chatbots via simulated user interactions
- [In LangSmith](chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation.md): Evaluate chatbots in LangSmith over a dialog dataset

### Experimental

- [Web Research (STORM)](storm/storm.md): Generate Wikipedia-like articles via research and multi-perspective QA
- [TNT-LLM](tnt-llm/tnt-llm.md): Build rich, interpretable taxonomies of user intentand using the classification system developed by Microsoft for their Bing Copilot application.
- [Web Navigation](web-navigation/web_voyager.md): Build an agent that can navigate and interact with websites
- [Competitive Programming](usaco/usaco.md): Build an agent with few-shot "episodic memory" and human-in-the-loop collaboration to solve problems from the USA Computing Olympiad; adapted from the ["Can Language Models Solve Olympiad Programming?"](https://arxiv.org/abs/2404.10952v1) paper by Shi, Tang, Narasimhan, and Yao.
- [Complex data extraction](extraction/retries.md): Build an agent that can use function calling to do complex extraction tasks

## LangGraph Platform 🧱 {#platform}

### Authentication & Access Control

Add custom authentication and authorization to an existing LangGraph Platform deployment in the following three-part guide:

1. [Setting Up Custom Authentication](auth/getting_started.md): Implement OAuth2 authentication to authorize users on your deployment
2. [Resource Authorization](auth/resource_auth.md): Let users have private conversations
3. [Connecting an Authentication Provider](auth/add_auth_server.md): Add real user accounts and validate using OAuth2