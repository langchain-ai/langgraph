# Tutorials

Welcome to the LangGraph Tutorials! These notebooks introduce LangGraph through building various language agents and applications.

## Introduction to LangGraph

Learn the basics of LangGraph through the onboarding tutorials.

- [Introduction to LangGraph](introduction.ipynb)


## AgentExecutor

Learn to build a simple agent in LangGraph.

- [Base](agent_executor/base.ipynb): Learn to build a LangGraph agent "from scratch"
- [High-Level](agent_executor/high-level.ipynb): Learn to use the `create_agent_executor`

## Chat Agent Executor

Learn to build a simple chat agent executor, which is a basic graph with an agentic loop that also supports dialog with a user.

- [Base](chat_agent_executor_with_function_calling/base.ipynb): Build a chat agent executor with function calling
- [High-Level](chat_agent_executor_with_function_calling/high-level.ipynb): Using the high-level chat agent executor API
- [High-Level Tools](chat_agent_executor_with_function_calling/high-level-tools.ipynb): Integrating tools into the high-level chat agent executor

## Use cases

Learn from example implementations of graphs designed for specific scenarios and that implement common design patterns.

#### Chatbots

- [Customer Support](customer-support/customer-support.ipynb): Build a customer support chatbot to manage flights, hotel reservations, car rentals, and other tasks
- [Info Gathering](chatbots/information-gather-prompting.ipynb): Build an information gathering chatbot  
- [Code Assistant](code_assistant/langgraph_code_assistant.ipynb): Building a code analysis and generation assistant
- [Web Navigation](web-navigation/web_voyager.ipynb): Building an agent that can navigate and interact with websites


#### Multi-Agent Systems

- [Collaboration](multi_agent/multi-agent-collaboration.ipynb): Enabling two agents to collaborate on a task  
- [Supervision](multi_agent/agent_supervisor.ipynb): Using an LLM to orchestrate and delegate to individual agents
- [Hierarchical Teams](multi_agent/hierarchical_agent_teams.ipynb): Orchestrating nested teams of agents to solve problems

#### RAG

- [Adaptive RAG](rag/langgraph_adaptive_rag.ipynb)
    - [Adaptive RAG using Cohere](rag/langgraph_adaptive_rag_cohere.ipynb) 
    - [Adaptive RAG using local models](rag/langgraph_adaptive_rag_local.ipynb)
- [Agentic RAG.ipynb](rag/langgraph_agentic_rag.ipynb)
- [Corrective RAG](rag/langgraph_crag.ipynb)
    - [Corrective RAG with local models](rag/langgraph_crag_local.ipynb)
- [Self-RAG](rag/langgraph_self_rag.ipynb)
    - [Self-RAG with local models](rag/langgraph_self_rag_local.ipynb)

- [Web Research (STORM)](storm/storm.ipynb): Generating Wikipedia-like articles via research and multi-perspective QA


#### Planning Agents

- [Plan-and-Execute](plan-and-execute/plan-and-execute.ipynb): Implementing a basic planning and execution agent  
- [Reasoning without Observation](rewoo/rewoo.ipynb): Reducing re-planning by saving observations as variables 
- [LLMCompiler](llm-compiler/LLMCompiler.ipynb): Streaming and eagerly executing a DAG of tasks from a planner

#### Reflection & Critique 

- [Basic Reflection](reflection/reflection.ipynb): Prompting the agent to reflect on and revise its outputs
- [Reflexion](reflexion/reflexion.ipynb): Critiquing missing and superfluous details to guide next steps
- [Language Agent Tree Search](lats/lats.ipynb): Using reflection and rewards to drive a tree search over agents
- [Self-Discovering Agent](self-discover/self-discover.ipynb): Analyzing an agent that learns about its own capabilities

#### Competitve Programming

- [Can Language Models Solve Olympiad Programming?](usaco/usaco.ipynb): Build an agent with few-shot "episodic memory" and human-in-the-loop collaboration to solve problems from the USA Computing Olympiad; adapted from the [paper of the same name](https://arxiv.org/abs/2404.10952v1) by Shi, Tang, Narasimhan, and Yao.

#### Evaluation

- [Agent-based](chatbot-simulation-evaluation/agent-simulation-evaluation.ipynb): Evaluating chatbots via simulated user interactions
- [Dataset-based](chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation.ipynb): Evaluating chatbots in LangSmith over a dialog dataset