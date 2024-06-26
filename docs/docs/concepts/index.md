# Conceptual Guides

In this guide we will explore the concepts behind build agentic and multi-agent systems with LangGraph. We assume you have already learned the basic covered in the [introduction tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction) and want to deepen your understanding of LangGraph's underlying design and inner workings.

There are three main parts to this concept guide. First, we'll discuss at a very high level what it means to be agentic. Next, we'll look at lower-level concepts in LangGraph that are core for understanding how to build your own agentic systems. Finally, we'll discuss common agentic patterns and how you can achieve those with LangGraph. These will be mostly conceptual guides - for more technical, hands-on guides see our [how-to guides](https://langchain-ai.github.io/langgraph/how-tos/)


LangGraph for Agentic Applications

- [What does it mean to be agentic?](high_level#what-does-it-mean-to-be-agentic)
- [Why LangGraph](high_level#why-langgraph)
- [Deployment](high_level#deployment)

Low Level Concepts

- [Graphs](low_level#graphs)
  - [StateGraph](low_level#stategraph)
  - [MessageGraph](low_level#messagegraph)
  - [Compiling Your Graph](low_level#compiling-your-graph)
- [State](low_level#state)
  - [Schema](low_level#schema)
  - [Reducers](low_level#reducers)
  - [MessageState](low_level#messagestate)
- [Nodes](low_level#nodes)
  - [`START` node](low_level#start-node)
  - [`END` node](low_level#end-node)
- [Edges](low_level#edges)
  - [Normal Edges](low_level#normal-edges)
  - [Conditional Edges](low_level#conditional-edges)
  - [Entry Point](low_level#entry-point)
  - [Conditional Entry Point](low_level#conditional-entry-point)
- [Send](low_level#send)
- [Checkpointer](low_level#checkpointer)
- [Threads](low_level#threads)
- [Checkpointer states](low_level#checkpointer-state)
  - [Get state](low_level#get-state)
  - [Get state history](low_level#get-state-history)
  - [Update state](low_level#update-state)
- [Configuration](low_level#configuration)
- [Visualization](low_level#visualization)
- [Streaming](low_level#streaming)

Common Agentic Patterns

- [Structured output](agentic_concepts#structured-output)
- [Tool calling](agentic_concepts#tool-calling)
- [Memory](agentic_concepts#memory)
- [Human in the loop](agentic_concepts#human-in-the-loop)
  - [Approval](agentic_concepts#approval)
  - [Wait for input](agentic_concepts#wait-for-input)
  - [Edit agent actions](agentic_concepts#edit-agent-actions)
  - [Time travel](agentic_concepts#time-travel)
- [Map-Reduce](agentic_concepts#map-reduce)
- [Multi-agent](agentic_concepts#multi-agent)
- [Planning](agentic_concepts#planning)
- [Reflection](agentic_concepts#reflection)
- [Off-the-shelf ReAct Agent](agentic_concepts#react-agent)
