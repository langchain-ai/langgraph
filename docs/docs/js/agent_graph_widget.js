if (window.location.pathname.includes("langgraph/agents/overview/")) {
    mermaid.initialize({ startOnLoad: false });

    console.log("got here");

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
        console.log("RENDERING");
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
        graphContainer.innerHTML = graph;
        graphContainer.removeAttribute("data-processed");
        await mermaid.run({ nodes: [graphContainer] });
    }

    async function loadGraphData() {
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
}
