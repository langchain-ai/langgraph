# UI

You can use a pre-built chat UI for interacting with any LangGraph agent through the [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) ([deployed version](https://agentchat.vercel.app)) as long as it has a `messages` field in the graph's state.

This repository intends to act as a generic chat interface for any LangGraph agent. Using the [deployed version](https://agentchat.vercel.app) is the quickest way to get started, and allows you to interact with both your local, and deployed graphs.

If you want to run the Agent Chat UI locally, you can either clone the [repository](https://github.com/langchain-ai/agent-chat-ui) or use the [`npx create-agent-chat-app`](https://www.npmjs.com/package/create-agent-chat-app) command, which will also give you a way to pull in up to four prebuilt LangGraph agents (in TypeScript) alongside it.

## Run agent in UI

First, set up LangGraph API server [locally](./deployment.md#launch-langgraph-server-locally) or deploy your agent on [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/quick_start/).

Then, navigate to [Agent Chat UI](https://agentchat.vercel.app), or clone the repository and [run the dev server locally](https://github.com/langchain-ai/agent-chat-ui?tab=readme-ov-file#setup):

<video controls src="../assets/base-chat-ui.mp4" type="video/mp4"></video>

You'll notice that the UI has out of the box support for rendering tool calls, and tool result messages. To customize what messages are shown, see the [Hiding Messages in the Chat](https://github.com/langchain-ai/agent-chat-ui?tab=readme-ov-file#hiding-messages-in-the-chat) section in the Agent Chat UI documentation.

## Add human-in-the-loop

Agent Chat UI has full support for [human-in-the-loop](../concepts/human_in_the_loop.md) workflows. To try it out, replace the agent code in `src/agent/graph.py` (from the deployment quickstart guide) with this [agent implementation](./human-in-the-loop.md#using-with-agent-inbox):

<video controls src="../assets/interrupt-chat-ui.mp4" type="video/mp4"></video>

It's important to note that the Agent Chat UI works best if your LangGraph agent interrupts using the [`HumanInterrupt` schema](TODO: ADD API REF LINK). If you do not use that schema, the Agent Chat UI will be able to render the input passed to the `interrupt` function, but it will not have full support for resuming your graph.

## Generative UI

You can also use generative UI in the Agent Chat UI.

Generative UI allows you to define React components, and push them to the UI from the LangGraph server. For more documentation on building generative UI LangGraph agents, read [these docs](https://langchain-ai.github.io/langgraph/cloud/how-tos/generative_ui_react/).
