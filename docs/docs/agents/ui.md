# UI

You can create a UI for interacting with your agent using [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui).

## Run agent in UI

First, set up LangGraph API server [locally](./deployment.md#launch-langgraph-server-locally) or deploy your agent on [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/quick_start/).

Then, navigate to [Agent Chat UI](https://agentchat.vercel.app):

<video controls src="../assets/base-chat-ui.mp4" type="video/mp4"></video>

## Add human-in-the-loop

Agent Chat UI has full support for [human-in-the-loop](../concepts/human_in_the_loop.md) workflows. To try it out, replace the agent code in `src/agent/graph.py` with this [agent implementation](./human-in-the-loop.md#using-with-agent-inbox):

<video controls src="../assets/interrupt-chat-ui.mp4" type="video/mp4"></video>