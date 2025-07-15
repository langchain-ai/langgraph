# Guides

The pages in this section provide a conceptual overview and how-tos for the following topics:

## LangGraph APIs

- [Graph API](../concepts/low_level.md): Use the Graph API to define workflows using a graph paradigm.
- [Functional API](../concepts/functional_api.md): Use Functional API to build workflows using a functional paradigm without thinking about the graph structure.
- [Runtime](../concepts/pregel.md): Pregel implements LangGraph's runtime, managing the execution of LangGraph applications.

## Core capabilities

These capabilities are available in both LangGraph OSS and the LangGraph Platform.

- [Streaming](../concepts/streaming.md): Stream outputs from a LangGraph graph.
- [Persistence](../concepts/persistence.md): Persist the state of a LangGraph graph.
- [Durable execution](../concepts/durable_execution.md): Save progress at key points in the graph execution.
- [Memory](../concepts/memory.md): Remember information about previous interactions.
- [Context](../agents/context.md): Pass outside data to a LangGraph graph to provide context for the graph execution.
- [Models](../agents/models.md): Integrate various LLMs into your LangGraph application.
- [Tools](../concepts/tools.md): Interface directly with external systems.
- [Human-in-the-loop](../concepts/human_in_the_loop.md): Pause a graph and wait for human input at any point in a workflow.
- [Time travel](../concepts/time-travel.md): Travel back in time to a specific point in the execution of a LangGraph graph.
- [Subgraphs](../concepts/subgraphs.md): Build modular graphs.
- [Multi-agent](../concepts/multi_agent.md): Break down a complex workflow into multiple agents.
- [MCP](../concepts/mcp.md): Use MCP servers in a LangGraph graph.
- [Evaluation](../agents/evals.md): Use LangSmith to evaluate your graph's performance.

## Platform-only capabilities

These capabilities are only available in [LangGraph Platform](../concepts/langgraph_platform.md).

- [Authentication and access control](../concepts/auth.md): Authenticate and authorize users to access a LangGraph graph.
- [Assistants](../concepts/assistants.md): Build assistants that can be used to interact with a LangGraph graph.
- [Double-texting](../concepts/double_texting.md): Handle double-texting (consecutive messages before a first response is returned) in a LangGraph graph.
- [Webhooks](../cloud/concepts/webhooks.md): Send webhooks to a LangGraph graph.
- [Cron jobs](../cloud/concepts/cron_jobs.md): Schedule jobs to run at a specific time.
- [Server customization](../how-tos/http/custom_lifespan.md): Customize the server that runs a LangGraph graph.
- [Data management](../cloud/concepts/data_storage_and_privacy.md): Manage data in a LangGraph graph.
- [Deployment](../concepts/deployment_options.md): Deploy a LangGraph graph to a server.
