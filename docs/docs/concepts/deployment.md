# Deployment

Once you build your LangGraph agent, you then have to think about deploying it.
How you deploy your applications depends a bit on what it is and how you intend to use it.
For simple applications, it can be pretty easy to deploy LangGraph.
As your applications get more complicated (more nodes, longer running) or rely more on persistence (conversation memory, long-term memory, human-in-the-loop) then deployment can get trickier.
In this guide, we go over a few of the considerations you will want to take into account when deploying more complicated LangGraph applications.

> 📘 LangGraph Platform
>
> This guide covers concepts related to deployment in general. [LangGraph Platform](./langgraph_platform.md) is an opinionated way to deploy LangGraph agents built by the LangChain team, and handles all the issues raised here.

## Simple deployment

If your application is simple, then it is quite easy to deploy LangGraph agents - you can deploy them as you would any other Python application. For example, you could [FastAPI](https://fastapi.tiangolo.com/) (or an equivalent framework) to deploy your LangGraph application.

```python
# main.py

from fastapi import FastAPI
from your_agent_package import graph

app = FastAPI()

@app.get("/foo")
async def root():
    return await graph.ainvoke({...})
```

For simple applications, this is sufficient and is not that different from deploying Python applications in general, and so we will not go into this in too much detail here.

For more complicated agent deployments (whether using LangGraph or not), you will run into some issues. The rest of this guide will discuss those issues.

> 📘 LangGraph Platform
>
> This guide covers concepts related to deployment in general. [LangGraph Platform](./langgraph_platform.md) is an opinionated way to deploy LangGraph agents built by the LangChain team, and handles all the issues raised here.

## Streaming support

[Streaming](streaming.md) is a key part of LLM applications. It is usually necessary to stream back both tokens and intermediate steps of longer running agents. This is because LLM applications can take a while to run end-to-end, and so streaming helps show the user that something is happening.

LangGraph supports [multiple different streaming modes](streaming.md). When deploying, you will want to make sure that you expose the relevant endpoints.

## Background runs

As your LLM application gets more complicated and starts taking longer to run, you may no longer want to stream back results. This is because these runs may take a while to finish and it may not make sense to hold open a connection for all that time.

As a result, you may also want to run your LangGraph in the background. When doing this, you will need to expose an endpoint to start the background run and another one that you can poll to check the status of the run (e.g. if the run is finished).

## Long runs

In addition to adding endpoints for [background runs](#background-runs), you also need to make sure your infrastructure can support long-lasting runs. Some web infrastructure is optimized for synchronous, shorter interactions, and may break if asked to execute long running jobs.

## Burstiness

Some LLM applications can often be rather bursty. For example, if you put an agent into a spreadsheet-UX, the user could be kicking of thousands of agents at the same time. You need to make sure your infrastructure can handle this burstiness.

This is typically handled by implementing some sort of task queue.

## Double texting

Many times users might interact with your graph in unintended ways. For instance, a user may send one message and before the graph has finished running send a second message.

We call this ["double texting"](double_texting.md) and you will need to think about this when you deploy your agent.

## Checkpointers

Many applications require some level of persistence. An example of this is persistent conversation history, where you can have a conversation one day, and then come back later and resume that conversation.

LangGraph has built in support for this with [checkpointers](persistence.md#checkpoints). When deploying your LangGraph application, you therefore need to also deploy a production-grade checkpointer along side it.


## Human-in-the-loop

Agents can make mistakes. A common UX paradigm to overcome this is to add more [human-in-the-loop](human_in_the_loop.md) features to your application.

These will all be powered by [checkpointers](persistence.md#checkpoints). Assuming you have already deployed one (see [above](#checkpointers)) you additionally need to expose the correct endpoints to allow your application to access the current state and the state history, and to update the state.


## Memory

In addition to thread-level persistence (covered above by [checkpointers](#checkpointers)) you may need to deploy some storage to support cross-thread memory.

LangGraph has built in support for this with [stores](persistence.md#memory-store). When deploying your LangGraph application, you therefore need to also deploy a production-grade store alongside it.