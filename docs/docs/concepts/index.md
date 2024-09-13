# Conceptual Guides

In these guides we will explore the concepts behind build agentic and multi-agent systems with LangGraph. We assume you have already learned the basic covered in the [introduction tutorial](../tutorials/introduction.ipynb) and want to deepen your understanding of LangGraph's underlying design and inner workings.

These conceptual guides are arranged in the following way:

* first, we'll discuss at a very high level the challenges that arise when building agents and [how LangGraph addresses them](high_level.md)
* next, we'll look at a list of [core LangGraph concepts](low_level.md) that are key for understanding how to build your own agentic systems
* then, we'll discuss [common agentic patterns](agentic_concepts.md) and how you can achieve those with LangGraph
* finally, we'll take an in-depth look at several concepts - [human-in-the-loop](human_in_the_loop.md), [multi-agent systems](multi_agent.md), [persistence](persistence.md) and [streaming](streaming.md)

For more technical, hands-on guides see our [how-to guides](../how-tos/index.md) and [tutorials](../tutorials/index.md).

## [Why LangGraph?](high_level.md)

- [Core Principles](high_level.md#core-principles)
- [Debugging](high_level.md#debugging)
- [Deployment](high_level.md#deployment)

## [LangGraph Glossary](low_level.md)

- [Graphs](low_level.md#graphs)
    - [StateGraph](low_level.md#stategraph)
    - [MessageGraph](low_level.md#messagegraph)
    - [Compiling Your Graph](low_level.md#compiling-your-graph)
- [State](low_level.md#state)
    - [Schema](low_level.md#schema)
    - [Reducers](low_level.md#reducers)
    - [MessageState](low_level.md#working-with-messages-in-graph-state)
- [Nodes](low_level.md#nodes)
    - [`START` node](low_level.md#start-node)
    - [`END` node](low_level.md#end-node)
- [Edges](low_level.md#edges)
    - [Normal Edges](low_level.md#normal-edges)
    - [Conditional Edges](low_level.md#conditional-edges)
    - [Entry Point](low_level.md#entry-point)
    - [Conditional Entry Point](low_level.md#conditional-entry-point)
- [Send](low_level.md#send)
- [Persistence](low_level.md#persistence)
- [Graph Migrations](low_level.md#graph-migrations)
- [Configuration](low_level.md#configuration)
- [Breakpoints](low_level.md#breakpoints)
- [Visualization](low_level.md#visualization)
- [Streaming](low_level.md#streaming)

## [Common Agentic Patterns](agentic_concepts.md)

- [Structured output](agentic_concepts.md#structured-output)
- [Tool calling](agentic_concepts.md#tool-calling)
- [Memory](agentic_concepts.md#memory)
- [Human in the loop](agentic_concepts.md#human-in-the-loop)
- [Map-Reduce](agentic_concepts.md#map-reduce)
- [Multi-agent](agentic_concepts.md#multi-agent)
- [Planning](agentic_concepts.md#planning)
- [Reflection](agentic_concepts.md#reflection)
- [Off-the-shelf ReAct Agent](agentic_concepts.md#react-agent)

## [Human-in-the-Loop](human_in_the_loop.md)

 - [Approval](human_in_the_loop.md#approval)
 - [Wait for input](human_in_the_loop.md#wait-for-input)
 - [Edit agent actions](human_in_the_loop.md#edit-agent-actions)
 - [Time travel](human_in_the_loop.md#time-travel)
 - [Review tool calls](human_in_the_loop.md#review-tool-calls)

## [Multi-Agent Systems](multi_agent.md)

## [Persistence](persistence.md)

## [Streaming](streaming.md)

 - [Streaming graph outputs](streaming.md#streaming-graph-outputs-stream-and-astream)
 - [Streaming LLM tokens and events](streaming.md#streaming-llm-tokens-and-events-astream_events)