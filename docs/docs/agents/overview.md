---
title: Overview
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# Agent development with LangGraph

**LangGraph** provides both low-level primitives and high-level prebuilt components for building agent-based applications. This section focuses on the **prebuilt**, **reusable** components designed to help you construct agentic systems quickly and reliably—without the need to implement orchestration, memory, or human feedback handling from scratch.

## What is an agent?

An *agent* consists of three components: a **large language model (LLM)**, a set of **tools** it can use, and a **prompt** that provides instructions.

The LLM operates in a loop. In each iteration, it selects a tool to invoke, provides input, receives the result (an observation), and uses that observation to inform the next action. The loop continues until a stopping condition is met — typically when the agent has gathered enough information to respond to the user.

<figure markdown="1">
![image](./assets/agent.png){: style="max-height:400px"}
<figcaption>Agent loop: the LLM selects tools and uses their outputs to fulfill a user request.</figcaption>
</figure>

## Key features

LangGraph includes several capabilities essential for building robust, production-ready agentic systems:

- [**Memory integration**](./memory.md): Native support for *short-term* (session-based) and *long-term* (persistent across sessions) memory, enabling stateful behaviors in chatbots and assistants.
- [**Human-in-the-loop control**](./human-in-the-loop.md): Execution can pause *indefinitely* to await human feedback—unlike websocket-based solutions limited to real-time interaction. This enables asynchronous approval, correction, or intervention at any point in the workflow.
- [**Streaming support**](./streaming.md): Real-time streaming of agent state, model tokens, tool outputs, or combined streams.
- [**Deployment tooling**](./deployment.md): Includes infrastructure-free deployment tools. [**LangGraph Platform**](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/) supports testing, debugging, and deployment.
    - **[Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)**: A visual IDE for inspecting and debugging workflows.
    - Supports multiple [**deployment options**](https://langchain-ai.github.io/langgraph/tutorials/deployment/) for production.

## High-level building blocks

LangGraph comes with a set of prebuilt components that implement common agent behaviors and workflows. These abstractions are built on top of the LangGraph framework, offering a faster path to production while remaining flexible for advanced customization.

Using LangGraph for agent development allows you to focus on your application's logic and behavior, instead of building and maintaining the supporting infrastructure for state, memory, and human feedback.

## Package ecosystem

The high-level components are organized into several packages, each with a specific focus.

| Package                                    | Description                                                                 | Installation                            |
|--------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------|
| `langgraph-prebuilt` (part of `langgraph`) | Prebuilt components to [**create agents**](./agents.md)                     | `pip install -U langgraph langchain`    |
| `langgraph-supervisor`                     | Tools for building [**supervisor**](./multi-agent.md#supervisor) agents     | `pip install -U langgraph-supervisor`   |
| `langgraph-swarm`                          | Tools for building a [**swarm**](./multi-agent.md#swarm) multi-agent system | `pip install -U langgraph-swarm`        |
| `langchain-mcp-adapters`                   | Interfaces to [**MCP servers**](./mcp.md) for tool and resource integration | `pip install -U langchain-mcp-adapters` |
| `langmem`                                  | Agent memory management: [**short-term and long-term**](./memory.md)        | `pip install -U langmem`                |
| `agentevals`                               | Utilities to [**evaluate agent performance**](./evals.md)                   | `pip install -U agentevals`             |

## Agent Graph Explorer

The **Agent Graph Explorer** is a tool for visualizing the graph generated under the hood of `create_react_agent`.
It allows you to explore the infrastructure of the agent as defined by the presence:

* `tools`: A list of tools (functions, APIs, or other callable objects) that the agent can use to perform tasks.
* `pre_model_hook`: A function that is called before the model is invoked. It can be used to condense messages or perform other preprocessing tasks.
* `post_model_hook`: A function that is called after the model is invoked. It can be used to implement guardrails, human-in-the-loop flows, or other postprocessing tasks.
* `response_format`: A data structure used to constrain the type of the final output, ex a `pydantic` `BaseModel`.

<div class="agent-layout">
  <div class="agent-graph-features-container">
    <div class="agent-graph-features">
      <h3 class="agent-section-title">Agent features</h3>
      <label><input type="checkbox" id="tools" checked> <code>tools</code></label>
      <label><input type="checkbox" id="pre_model_hook"> <code>pre_model_hook</code></label>
      <label><input type="checkbox" id="post_model_hook"> <code>post_model_hook</code></label>
      <label><input type="checkbox" id="response_format"> <code>response_format</code></label>
    </div>
  </div>

  <div class="agent-graph-container">
    <h3 class="agent-section-title">Agent graph</h3>
    <div class="mermaid" id="agent-graph"></div>
  </div>
</div>


### Code snippet

<div class="language-python">
  <pre><code id="agent-code" class="language-python"></code></pre>
</div>

<script>
mermaid.initialize({ startOnLoad: false });

var graphData = graphData || null;

function getKey() {
    return [
        document.getElementById("response_format").checked ? "1" : "0",
        document.getElementById("post_model_hook").checked ? "1" : "0",
        document.getElementById("pre_model_hook").checked ? "1" : "0",
        document.getElementById("tools").checked ? "1" : "0"
    ].join("");
}

function generateCodeSnippet({ tools, pre, post, response }) {
    const lines = [];
    lines.push("from langgraph.prebuilt import create_react_agent");
    lines.push("from langchain_openai import ChatOpenAI");
    if (response) lines.push("from pydantic import BaseModel");
    lines.push("");
    lines.push('model = ChatOpenAI("o4-mini")\n');
    if (tools) {
        lines.push("def tool() -> None:");
        lines.push('    """Testing tool."""');
        lines.push("    ...\n");
    }
    if (pre) {
        lines.push("def pre_model_hook() -> None:");
        lines.push('    """Pre-model hook."""');
        lines.push("    ...\n");
    }
    if (post) {
        lines.push("def post_model_hook() -> None:");
        lines.push('    """Post-model hook."""');
        lines.push("    ...\n");
    }
    if (response) {
        lines.push("class ResponseFormat(BaseModel):");
        lines.push('    """Response format for the agent."""');
        lines.push("    result: str\n");
    }
    lines.push("agent = create_react_agent(");
    lines.push("    model,");
    if (tools) lines.push("    tools=[tool],");
    if (pre) lines.push("    pre_model_hook=pre_model_hook,");
    if (post) lines.push("    post_model_hook=post_model_hook,");
    if (response) lines.push("    response_format=ResponseFormat,");
    lines.push(")");
    lines.push("");
    lines.push("agent.get_graph().draw_mermaid_png()");
    return lines.join("\n");
}

async function render() {
    const key = getKey();
    const default_graph = "graph TD;\n  A --> B;";
    const graph = graphData === null ? default_graph : (graphData[key] || default_graph);
    const codeContainer = document.getElementById("agent-code");
    const graphContainer = document.getElementById("agent-graph");

    const flags = {
        tools: document.getElementById("tools").checked,
        pre: document.getElementById("pre_model_hook").checked,
        post: document.getElementById("post_model_hook").checked,
        response: document.getElementById("response_format").checked
    };

    codeContainer.textContent = generateCodeSnippet(flags);
    graphContainer.innerHTML = graph;
    graphContainer.removeAttribute("data-processed");
    await mermaid.run({ nodes: [graphContainer] });
}

async function loadGraphData() {
  if (graphData !== null) {
      return;
  }
  // Load the graph data from the JSON file
  try {
      const response = await fetch("../assets/react-agent-graphs.json");
      graphData = await response.json();
  } catch (err) {
      console.error("Failed to load graphData.json:", err);
  }

}

async function initializeWidget() {
    await loadGraphData();
    await render();
    document.querySelectorAll(".agent-graph-features input").forEach((input) => {
        input.addEventListener("change", async () => await render());
        console.log("added event listener");
    });
}

// Handle initial load and subsequent navigation
// See admonition for more details: https://squidfunk.github.io/mkdocs-material/customization/#additional-javascript
window.addEventListener("DOMContentLoaded", initializeWidget);
document$.subscribe(initializeWidget);
</script>
