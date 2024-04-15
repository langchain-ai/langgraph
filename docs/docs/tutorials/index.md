# Tutorials

Welcome to the LangGraph Tutorials! These notebooks provide end-to-end walkthroughs for building various types of language agents and applications using LangGraph.

#### Basics

- [Base](chat_agent_executor_with_function_calling/base.ipynb): Implementing a chat agent executor with function calling
- [High-Level](chat_agent_executor_with_function_calling/high-level.ipynb): Using the high-level chat agent executor API
- [High-Level Tools](chat_agent_executor_with_function_calling/high-level-tools.ipynb): Integrating tools into the high-level chat agent executor

## AgentExecutor

- [Base](agent_executor/base.ipynb): Build a LangGraph agent "from scratch"
- [High-Level](agent_executor/high-level.ipynb): Using the high-level `create_agent_executor` function

#### Graph Structure

- [Subgraphs](subgraph.ipynb): How to modularize your graphs with subgraphs
- [Branching](branching.ipynb): How to create branching logic in your graphs


## Use cases

#### Chatbots

- [Customer Support](chatbots/customer-support.ipynb): Building a customer support chatbot
- [Info Gathering](chatbots/information-gather-prompting.ipynb): Building an information gathering chatbot  
- [Code Assistant](code_assistant/langgraph_code_assistant.ipynb): Building a code analysis and generation assistant
- [Web Navigation](web-navigation/web_voyager.ipynb): Building an agent that can navigate and interact with websites


#### Multi-Agent Systems

- [Collaboration](multi_agent/multi-agent-collaboration.ipynb): Enabling two agents to collaborate on a task  
- [Supervision](multi_agent/agent_supervisor.ipynb): Using an LLM to orchestrate and delegate to individual agents
- [Hierarchical Teams](multi_agent/hierarchical_agent_teams.ipynb): Orchestrating nested teams of agents to solve problems

#### RAG

**Retrieval-Augmented Generation**

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


### Evaluation & Analysis

**Chatbot Evaluation via Simulation**  

- [Agent-based](chatbot-simulation-evaluation/agent-simulation-evaluation.ipynb): Evaluating chatbots via simulated user interactions
- [Dataset-based](chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation.ipynb): Evaluating chatbots over a dialog dataset