# 🧠 Agent Graph Infrastructure Explorer

<div class="agent-toggle">
  <label><input type="checkbox" id="tools" checked> tools</label>
  <label><input type="checkbox" id="pre_model_hook"> pre_model_hook</label>
  <label><input type="checkbox" id="post_model_hook"> post_model_hook</label>
  <label><input type="checkbox" id="response_format"> response_format</label>
</div>

### 🧾 Generated Code

<div class="language-python highlight">
  <pre><code id="agent-code"></code></pre>
</div>

### 📊 Agent Graph Data

<div class="mermaid" id="agent-graph"></div>

<script src="https://unpkg.com/mermaid@11.6.0/dist/mermaid.min.js"></script>

<script>
mermaid.initialize({ startOnLoad: false });

let graphData = {};

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

  // Imports
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

  return lines.join("\n");
}

async function render() {
  const key = getKey();
  const graph = graphData[key] || "graph TD;\n  A --> B;";
  const codeContainer = document.getElementById("agent-code");
  const graphContainer = document.getElementById("agent-graph");

  const flags = {
    tools: document.getElementById("tools").checked,
    pre: document.getElementById("pre_model_hook").checked,
    post: document.getElementById("post_model_hook").checked,
    response: document.getElementById("response_format").checked
  };

  codeContainer.textContent = generateCodeSnippet(flags);

  // const { svg } = await mermaid.render('agent-graph', graph);
  // console.log(svg);
  // console.log(graph);
  // console.log(graphContainer)
  graphContainer.innerHTML = graph;
  graphContainer.removeAttribute("data-processed");
  await mermaid.run({
    nodes: [graphContainer],
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  try {
    const response = await fetch("./assets/react-agent-graphs.json");
    graphData = await response.json();
    render().catch(console.error);

    document.querySelectorAll(".agent-toggle input").forEach((input) => {
      input.addEventListener("change", () => render().catch(console.error));
    });
  } catch (err) {
    console.error("Failed to load graphData.json:", err);
  }
});
</script>