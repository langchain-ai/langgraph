---
title: Tutorials
search:
  boost: 0.5
---

# Tutorials

New to LangGraph or LLM app development? Read this material to get up and running building your first applications.

## Get Started 🚀 {#quick-start}

- [LangGraph Quickstart](introduction.ipynb): Build a chatbot that can use tools and keep track of conversation history. Add human-in-the-loop capabilities and explore how time-travel works.
- [Common Workflows](workflows/index.md): Overview of the most common workflows using LLMs implemented with LangGraph.
- [LangGraph Server Quickstart](langgraph-platform/local-server.md): Launch a LangGraph server locally and interact with it using REST API and LangGraph Studio Web UI.
- [LangGraph Template Quickstart](../concepts/template_applications.md): Start building with LangGraph Platform using a template application.
- [Deploy with LangGraph Cloud Quickstart](../cloud/quick_start.md): Deploy a LangGraph app using LangGraph Cloud.

## Use cases 🛠️ {#use-cases}


Explore practical implementations tailored for specific scenarios:

### Chatbots

- [Customer Support](customer-support/customer-support.ipynb): Build a multi-functional support bot for flights, hotels, and car rentals.
- [Prompt Generation from User Requirements](chatbots/information-gather-prompting.ipynb): Build an information gathering chatbot.
- [Code Assistant](code_assistant/langgraph_code_assistant.ipynb): Build a code analysis and generation assistant.

### RAG

- [Agentic RAG](rag/langgraph_agentic_rag.ipynb): Use an agent to figure out how to retrieve the most relevant information before using the retrieved information to answer the user's question.
- [SQL Agent](sql-agent.ipynb): Build a SQL agent that can answer questions about a SQL database.


### Agent Architectures

#### Multi-Agent Systems

- [Network](multi_agent/multi-agent-collaboration.ipynb): Enable two or more agents to collaborate on a task
- [Supervisor](multi_agent/agent_supervisor.ipynb): Use an LLM to orchestrate and delegate to individual agents
 
#### Planning Agents

- [Plan-and-Execute](plan-and-execute/plan-and-execute.ipynb): Implement a basic planning and execution agent
- [Reasoning without Observation](rewoo/rewoo.ipynb): Reduce re-planning by saving observations as variables
- [LLMCompiler](llm-compiler/LLMCompiler.ipynb): Stream and eagerly execute a DAG of tasks from a planner

#### Reflection & Critique 

- [Basic Reflection](reflection/reflection.ipynb): Prompt the agent to reflect on and revise its outputs
- [Reflexion](reflexion/reflexion.ipynb): Critique missing and superfluous details to guide next steps
- [Tree of Thoughts](tot/tot.ipynb): Search over candidate solutions to a problem using a scored tree
- [Language Agent Tree Search](lats/lats.ipynb): Use reflection and rewards to drive a monte-carlo tree search over agents
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

## LangGraph Platform 🧱 {#platform}

### Authentication & Access Control

Add custom authentication and authorization to an existing LangGraph Platform deployment in the following three-part guide:

1. [Setting Up Custom Authentication](auth/getting_started.md): Implement OAuth2 authentication to authorize users on your deployment
2. [Resource Authorization](auth/resource_auth.md): Let users have private conversations
3. [Connecting an Authentication Provider](auth/add_auth_server.md): Add real user accounts and validate using OAuth2