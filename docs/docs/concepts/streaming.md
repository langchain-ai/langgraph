---
search:
boost: 2
---

# Streaming

Building a responsive app for end users? Real-time updates are key to keeping users engaged as the application progresses.

LangGraph’s streaming system lets you surface live feedback from graph runs to your app.  
There are three main categories of data you can stream:

1. **Workflow progress** — get state updates after each graph node is executed.
2. **LLM tokens** — stream language model tokens as they’re generated.
3. **Custom updates** — emit user-defined signals (e.g., “Fetched 10/100 records”).

<figure markdown="1">
![image](../../agents/assets/fast_parrot.png){: style="max-height:300px"}
<figcaption>
Waiting is for pigeons.
</figcaption>
</figure>


## What’s possible with LangGraph streaming

- **Stream LLM tokens** — capture token streams from anywhere: inside nodes, subgraphs, or tools.
- **Emit progress notifications from tools** — send custom updates or progress signals directly from tool functions.
- **Stream from subgraphs** — include outputs from both the parent graph and any nested subgraphs.
- **Use any LLM** — stream tokens from any LLM, even if it's not a LangChain model using the `custom` streaming mode.
- **Use multiple streaming modes** — choose from `values` (full state), `updates` (state deltas), `messages` (LLM tokens + metadata), `custom` (arbitrary user data), or `debug` (detailed traces).