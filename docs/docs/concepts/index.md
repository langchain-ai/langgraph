# Conceptual Guides

In this guide we will explore the concepts behind build agentic and multi-agent systems with LangGraph. We assume you have already learned the basic covered in the [introduction tutorial](/tutorials/introduction) and want to deepen your understanding of LangGraph's underlying design and inner workings.

There are three main parts to this concept guide. First, we'll discuss at a very high level what it means to be agentic. Next, we'll look at lower-level concepts in LangGraph that are core for understanding how to build your own agentic systems. Finally, we'll discuss common agentic patterns and how you can achieve those with LangGraph. These will be mostly conceptual guides - for more technical, hands-on guides see our [how-to guides](/how-tos/)


LangGraph for Agentic Applications

- [What does it mean to be agentic?](high_level.md#what-does-it-mean-to-be-agentic)
- [Why LangGraph](high_level.md#why-langgraph)
- [Deployment](high_level.md#deployment)

Low Level Concepts

- [Graphs](low_level.md#graphs)
    - [StateGraph](low_level.md#stategraph)
    - [MessageGraph](low_level.md#messagegraph)
    - [Compiling Your Graph](low_level.md#compiling-your-graph)
- [State](low_level.md#state)
    - [Schema](low_level.md#schema)
    - [Reducers](low_level.md#reducers)
    - [MessageState](low_level.md#messagestate)
- [Nodes](low_level.md#nodes)
    - [`START` node](low_level.md#start-node)
    - [`END` node](low_level.md#end-node)
- [Edges](low_level.md#edges)
    - [Normal Edges](low_level.md#normal-edges)
    - [Conditional Edges](low_level.md#conditional-edges)
    - [Entry Point](low_level.md#entry-point)
    - [Conditional Entry Point](low_level.md#conditional-entry-point)
- [Send](low_level.md#send)
- [Checkpointer](low_level.md#checkpointer)
- [Threads](low_level.md#threads)
- [Checkpointer states](low_level.md#checkpointer-state)
    - [Get state](low_level.md#get-state)
    - [Get state history](low_level.md#get-state-history)
    - [Update state](low_level.md#update-state)
- [Configuration](low_level.md#configuration)
- [Visualization](low_level.md#visualization)
- [Streaming](low_level.md#streaming)

Common Agentic Patterns

- [Structured output](agentic_concepts.md#structured-output)
- [Tool calling](agentic_concepts.md#tool-calling)
- [Memory](agentic_concepts.md#memory)
- [Human in the loop](agentic_concepts.md#human-in-the-loop)
    - [Approval](agentic_concepts.md#approval)
    - [Wait for input](agentic_concepts.md#wait-for-input)
    - [Edit agent actions](agentic_concepts.md#edit-agent-actions)
    - [Time travel](agentic_concepts.md#time-travel)
- [Map-Reduce](agentic_concepts.md#map-reduce)
- [Multi-agent](agentic_concepts.md#multi-agent)
- [Planning](agentic_concepts.md#planning)
- [Reflection](agentic_concepts.md#reflection)
- [Off-the-shelf ReAct Agent](agentic_concepts.md#react-agent)
