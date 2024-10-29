# LangGraph Platform

## Overview

LangGraph Platform is an opinionated framework for building agent-based applications an API server tailored for interacting with assistants, with features such as streaming, [human-in-the-loop](./human_in_the_loop.md) interactions, [persistence](./persistence), cron jobs, web hooks, and more. It includes built-in queuing and persistence layers, an integrated IDE (LangGraph Studio, to streamline development.

Deploying applications with LangGraph shortens the time-to-market for developers. 

The LangGraph Cloud API exposes functionality of your LangGraph application through Assistants. An assistant abstracts the cognitive architecture of your graph. Invoke an assistant by calling the pre-built API endpoints.

LangGraph Cloud is seamlessly integrated with LangSmith and is accessible from within the LangSmith UI.

LangGraph Cloud applications can be tested and debugged using the LangGraph Studio Desktop.


## What is it?

LangGraph Platform is an opinionated way to deploy and manage LangGraph applications.

See [this section](#overview) for an overview of how LangGraph Platform actually deploys your agent.

LangGraph Platform was built to tackle [the issues](deployment.md) we saw users running into when trying to deploy LangGraph applications themselves. See [this section](#features) for an overview of the features present in LangGraph Platform and what issues they solve.

LangGraph Platform has its own concepts, features, and API related to [deployment](deployment.md) of agents. See [this section](#data-models) for an overview of the underlying data models for LangGraph Platform.

There are multiple ways to deploy LangGraph Platform, including a free option. See [this section](#deployment-options) for an overview and comparison of the different deployment options. 

## Overview

LangGraph Platform provides an opinionated way to deploy your agent.

It provides additional [features](#features) and [concepts](#data-models) compared to the open source.
It is designed to work seamlessly with your agent regardless of how it is defined, what tools it uses, or any dependencies.

With LangGraph Platform, you define your environment with a simple configuration file and it generates a Docker image with our API server built in.

## Features

Deploying agents in production can be [challenging](deployment.md).
We built LangGraph Platform to incorporate best practices for agent deployment so you can focus on building your agent logic.

With LangGraph Platform you get:

*   **Streaming endpoints**: Endpoints that expose [multiple different streaming modes](streaming.md). We've made these work even for long running agents that may go minutes between steam event.
*  **Background runs**:  We've exposed endpoints for running your graph in the background.
This includes endpoints for polling for the status of a run.
* **Support for long runs**: In addition to adding endpoints for background runs, we've also designed our endpoints and infrastructure to support long running agents.
* **Task queue**: We've added a task queue to make sure we don't drop any requests if they arrive in a bursty nature.
* **Horizontally scalable infrastructure**: LangGraph Platform is designed to be horizontally scalable, allowing you to scale up and down your usage as needed.
*  **Double texting support**: Many times users might interact with your graph in unintended ways. For instance, a user may send one message and before the graph has finished running send a second message. We call this ["double texting"](double_texting.md) and have added four different ways to handle this.
* **Optimized checkpointer**: LangGraph Platform comes with a built-in [checkpointer](persistence.md#checkpoints) optimized for LangGraph applications.
* **Human-in-the-loop endpoints**: We've exposed all endpoints needed to support [human-in-the-loop](human_in_the_loop.md) features.
* **Memory**: In addition to thread-level persistence (covered above by checkpointers), LangGraph Platform also comes with a built-in [memory store](persistence.md#memory-store).


### Graphs

In order to use LangGraph Platform, you need to specify the graph(s) you want to deploy.
You can specify multiple graphs to deploy at the same time.
You do not need to specify [checkpointers](persistence.md#checkpoints) or [memory stores](persistence.md#memory-store) when compiling your graphs - LangGraph platform will add those in automatically.

### Assistants

An [Assistant](assistants.md) refers to a [graph](#graphs) plus specific [configuration](low_level.md#configuration) settings for that graph.

When building agents, it is fairly common to make rapid changes that *do not* alter the graph logic. For example, simply changing prompts or the LLM selection can have significant impacts on the behavior of the agents. Assistants offer an easy way to make and save these types of changes to agent configuration. 


# Why use LangGraph Platform? 

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

As a result, you may also want to consider exposing an endpoint to run your LangGraph as a background run. When doing this, you will need to expose an endpoint that you can poll to check if your run is finished, or what its status is.

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

LangGraph has built in support for this with [checkpointers](persistence.md#checkpoints). When deploying your LangGraph application, you therefor need to also deploy a production-grade checkpointer along side it.


## Human-in-the-loop

Long running agents can mess up. A common UX paradigm to overcome this is to add more [human-in-the-loop](human_in_the_loop.md) features to your application.

These will all be powered by [checkpointers](persistence.md#checkpoints). Assuming you have already deployed one (see [above](#checkpointers)) you additionally need to expose the correct endpoints to allow your application to access the current state and the state history, and to update the state.


## Memory

In addition to thread-level persistence (covered above by [checkpointers](#checkpointers)) you may need to deploy some storage to support cross-thread memory.

LangGraph has built in support for this with [stores](persistence.md#memory-store). When deploying your LangGraph application, you therefor need to also deploy a production-grade store alongside it.
