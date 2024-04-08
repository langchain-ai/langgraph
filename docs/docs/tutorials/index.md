# Tutorials

Welcome to the LangGraph Tutorials! These notebooks provide end-to-end walkthroughs for building various types of language agents and applications using LangGraph.

## Agent Executors

- **Chat Agent (Function Calling)**
    - [Base](chat_agent_executor_with_function_calling/base.ipynb): Implementing a chat agent executor with function calling
    - [High-Level](chat_agent_executor_with_function_calling/high-level.ipynb): Using the high-level chat agent executor API
    - [High-Level Tools](chat_agent_executor_with_function_calling/high-level-tools.ipynb): Integrating tools into the high-level chat agent executor
    - **Modifications**
        - [Human-in-the-Loop](chat_agent_executor_with_function_calling/human-in-the-loop.ipynb)
        - [Force Tool First](chat_agent_executor_with_function_calling/force-calling-a-tool-first.ipynb)
        - [Respond in Format](chat_agent_executor_with_function_calling/respond-in-format.ipynb)
        - [Dynamic Direct Return](chat_agent_executor_with_function_calling/dynamically-returning-directly.ipynb)
        - [Manage Agent Steps](chat_agent_executor_with_function_calling/managing-agent-steps.ipynb)

- **LangChain Agent**
    - [Base](agent_executor/base.ipynb): Implementing an agent executor with Langchain agents  
    - [High-Level](agent_executor/high-level.ipynb): Using the high-level Langchain agent executor API
    - **Modifications** 
        - [Human-in-the-Loop](agent_executor/human-in-the-loop.ipynb)
        - [Force Tool First](agent_executor/force-calling-a-tool-first.ipynb)
        - [Manage Agent Steps](agent_executor/managing-agent-steps.ipynb)

## Planning Agents

- [Plan-and-Execute](plan-and-execute/plan-and-execute.ipynb): Implementing a basic planning and execution agent  
- [Reasoning without Observation](rewoo/rewoo.ipynb): Reducing re-planning by saving observations as variables 
- [LLMCompiler](llm-compiler/LLMCompiler.ipynb): Streaming and eagerly executing a DAG of tasks from a planner

## Reflection & Critique 

- [Basic Reflection](reflection/reflection.ipynb): Prompting the agent to reflect on and revise its outputs
- [Reflexion](reflexion/reflexion.ipynb): Critiquing missing and superfluous details to guide next steps
- [Language Agent Tree Search](lats/lats.ipynb): Using reflection and rewards to drive a tree search over agents

## Multi-Agent Systems

- [Collaboration](multi_agent/multi-agent-collaboration.ipynb): Enabling two agents to collaborate on a task  
- [Supervision](multi_agent/agent_supervisor.ipynb): Using an LLM to orchestrate and delegate to individual agents
- [Hierarchical Teams](multi_agent/hierarchical_agent_teams.ipynb): Orchestrating nested teams of agents to solve problems

## Research & QA

- **Retrieval-Augmented Generation**
    - [langgraph_adaptive_rag.ipynb](rag/langgraph_adaptive_rag.ipynb)
    - [langgraph_adaptive_rag_cohere.ipynb](rag/langgraph_adaptive_rag_cohere.ipynb) 
    - [langgraph_adaptive_rag_local.ipynb](rag/langgraph_adaptive_rag_local.ipynb)
    - [langgraph_agentic_rag.ipynb](rag/langgraph_agentic_rag.ipynb)
    - [langgraph_crag.ipynb](rag/langgraph_crag.ipynb)
    - [langgraph_crag_local.ipynb](rag/langgraph_crag_local.ipynb)
    - [langgraph_self_rag.ipynb](rag/langgraph_self_rag.ipynb)
    - [langgraph_self_rag_local.ipynb](rag/langgraph_self_rag_local.ipynb)
- [Web Research (STORM)](storm/storm.ipynb): Generating Wikipedia-like articles via research and multi-perspective QA

## Applications

- **Chatbots**
    - [Customer Support](chatbots/customer-support.ipynb): Building a customer support chatbot
    - [Info Gathering](chatbots/information-gather-prompting.ipynb): Building an information gathering chatbot  
- [Code Assistant](code_assistant/langgraph_code_assistant.ipynb): Building a code analysis and generation assistant
- [Web Navigation](web-navigation/web_voyager.ipynb): Building an agent that can navigate and interact with websites

## Evaluation & Analysis

- **Chatbot Evaluation via Simulation**  
    - [Agent-based](chatbot-simulation-evaluation/agent-simulation-evaluation.ipynb): Evaluating chatbots via simulated user interactions
    - [Dataset-based](chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation.ipynb): Evaluating chatbots over a dialog dataset