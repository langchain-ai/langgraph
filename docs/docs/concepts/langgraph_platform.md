# LangGraph Platform

LangGraph Platform is an opinionated way to deploy and manage your LangGraph application.

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

**Streaming endpoints**

Endpoints that expose [multiple different streaming modes](streaming.md). We've made these work even for long running agents that may go minutes between steam event.

**Background runs**

We've exposed endpoints for running your graph in the background.
This includes endpoints for polling for the status of a run.

**Support for long runs**

In addition to adding endpoints for background runs, we've also designed our endpoints and infrastructure to support long running agents.

**Task queue**

We've added a task queue to make sure we don't drop any requests if they arrive in a bursty nature.

**Horizontally scalable infrastructure**

LangGraph Platform is designed to be horizontally scalable, allowing you to scale up and down your usage as needed.

**Double texting support**

Many times users might interact with your graph in unintended ways. For instance, a user may send one message and before the graph has finished running send a second message.

We call this ["double texting"](double_texting.md) and have added four different ways to handle this.

**Optimized checkpointer**

LangGraph Platform comes with a built-in [checkpointer](persistence.md#checkpoints) optimized for LangGraph applications.


**Human-in-the-loop endpoints**

We've exposed all endpoints needed to support [human-in-the-loop](human_in_the_loop.md) features.


**Memory**

In addition to thread-level persistence (covered above by checkpointers), LangGraph Platform also comes with a built-in [memory store](persistence.md#memory-store).


## Deployment Options

There are several deployment options for LangGraph Platform.

|                                | Free | Self Hosted | Bring Your Own Cloud | Cloud |
|--------------------------------|------|-------------|----------------------|-------|
| Who manages the infrastructure | You  | You         | Us                   | Us    |
| With whom does the data reside | You  | You         | You                  | Us    |
| LangGraph Studio Included?     | ❌    | ✅           | ✅                    | ✅     |
| Assistants Included?           | ❌    | ✅           | ✅                    | ✅     |


### Free

All you need in order to use the free version of LangGraph Platform is a [LangSmith](https://smith.langchain.com/) API key.

You need to add this as an environment variable when running LangGraph Platform. It should be provided as `LANGSMITH_API_KEY=...`.

LangGraph Platform will provide a one-time check when starting up to ensure that it is a valid LangSmith key.

The free version of LangGraph Platform does not have access to some features that they other versions have, namely LangGraph Studio and Assistants.


### Cloud

The Cloud version of LangGraph Platform is hosted as part of [LangSmith](https://smith.langchain.com/).
This deployment option provides a seamless integration with GitHub to easily deploy your code from there.
It also integrates seamlessly with LangSmith for observability and testing.

While in beta, the Cloud version of LangGraph Platform is available to all users of LangSmith on the [Plus or Enterprise plans](https://docs.smith.langchain.com/administration/pricing).

### Self Hosted

The Self Hosted version of LangGraph Platform can be set up in the same way as the free version.
The only difference is that rather than specifying a LangSmith API key, you pass in a license key.

This license key gives you access to all LangGraph Platform features, like LangGraph Studio and Assistants.

This is a paid offering. Please contact sales@langchain.dev for pricing.

### Bring your own cloud

This combines the best of both worlds for Cloud and Self Hosted.
We manage the infrastructure, so you don't have to, but the infrastructure all runs within your cloud.

This is currently only available on AWS.

This is a paid offering. Please contact sales@langchain.dev for pricing.

## Cloud Deployment

This section describes the high-level concepts of the cloud deployment of LangGraph Platform.

### Deployment

A deployment is an instance of a LangGraph API. A single deployment can have many [revisions](#revision). When a deployment is created, all of the necessary infrastructure (e.g. database, containers, secrets store) are automatically provisioned. See the [architecture diagram](#architecture) below for more details.

See the [how-to guide](../cloud/deployment/cloud.md#create-new-deployment) for creating a new deployment.

### Revision

A revision is an iteration of a [deployment](#deployment). When a new deployment is created, an initial revision is automatically created. To deploy new code changes or update environment variable configurations for a deployment, a new revision must be created. When a revision is created, a new container image is built automatically.

See the [how-to guide](../cloud/deployment/cloud.md#create-new-revision) for creating a new revision.

### Asynchronous Deployment

Infrastructure for [deployments](#deployment) and [revisions](#revision) are provisioned and deployed asynchronously. They are not deployed immediately after submission. Currently, deployment can take up to several minutes.

### Architecture

!!! warning "Subject to Change"
    The LangGraph Cloud deployment architecture may change in the future.

A high-level diagram of a LangGraph Cloud deployment.

![diagram](img/langgraph_cloud_architecture.png)

## Data Models

The LangGraph Platform API consists of a few core data models: [Graphs](#graphs), [Assistants](#assistants), [Threads](#threads), [Runs](#runs), and [Cron Jobs](#cron-jobs).


### Graphs

In order to use LangGraph Platform, you need to specify the graph(s) you want to deploy.
You can specify multiple graphs to deploy at the same time.
You do not need to specify [checkpointers](persistence.md#checkpoints) or [memory stores](persistence.md#memory-store) when compiling your graphs - LangGraph platform will add those in automatically.

### Assistants

An [Assistant](assistants.md) refers to a [graph](#graphs) plus specific [configuration](low_level.md#configuration) settings for that graph.


When building agents, it is fairly common to make rapid changes that *do not* alter the graph logic. For example, simply changing prompts or the LLM selection can have significant impacts on the behavior of the agents. Assistants offer an easy way to make and save these types of changes to agent configuration. 

### Threads

A thread contains the accumulated state of a group of runs. If a run is executed on a thread, then the [state](low_level.md#state) of the underlying graph of the assistant will be persisted to the thread. A thread's current and historical state can be retrieved. To persist state, a thread must be created prior to executing a run.

The state of a thread at a particular point in time is called a checkpoint.

For more on threads and checkpoints, see this section of the [LangGraph conceptual guide](low_level.md#persistence).

The LangGraph Cloud API provides several endpoints for creating and managing threads and thread state. See the [API reference](../reference/api/api_ref.html#tag/threadscreate) for more details.

### Runs

A run is an invocation of an assistant. Each run may have its own input, configuration, and metadata, which may affect execution and output of the underlying graph. A run can optionally be executed on a thread.

The LangGraph Cloud API provides several endpoints for creating and managing runs. See the [API reference](../reference/api/api_ref.html#tag/runscreate) for more details.


### Cron Jobs

It's often useful to run assistants on some schedule. LangGraph Cloud supports cron jobs, which run on a user defined schedule. The user specifies a schedule, an assistant, and some input. After than, on the specified schedule LangGraph cloud will:

- Create a new thread with the specified assistant
- Send the specified input to that thread

Note that this sends the same input to the thread every time. See the [how-to guide](../cloud/how-tos/cron_jobs.md) for creating cron jobs.

The LangGraph Cloud API provides several endpoints for creating and managing cron jobs. See the [API reference](../reference/api/api_ref.html#tag/runscreate/POST/threads/{thread_id}/runs/crons) for more details.
