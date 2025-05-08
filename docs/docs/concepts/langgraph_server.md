---
search:
  boost: 2
---

# LangGraph Server

**LangGraph Server** offers an API for creating and managing agent-based applications. It is built on the concept of [assistants](assistants.md), which are agents configured for specific tasks, and includes built-in [persistence](persistence.md#memory-store) and a **task queue**. This versatile API supports a wide range of agentic application use cases, from background processing to real-time interactions.

Use LangGraph Serverto create and manage [assistants](assistants.md), [threads](../cloud/concepts/threads.md), [runs](../cloud/concepts/runs.md), [cron jobs](../cloud/concepts/cron_jobs.md), [webhooks](../cloud/concepts/webhooks.md), and more.

!!! tip "API reference"
  
    For detailed information on the API endpoints and data models, see [LangGraph Platform API reference docs](../cloud/reference/api/api_ref.html).

## Server versions

There are two versions of LangGraph Server:

- `Lite` is a limited version of the LangGraph Server that you can run locally or in a self-hosted manner (up to 1 million nodes executed per year).
- `Enterprise` is the full version of the LangGraph Server. To use the `Enterprise` version, you must acquire a license key that you will need to specify when running the Docker image. To acquire a license key, please email sales@langchain.dev.

Feature Differences:

|       | Lite       | Enterprise |
|-------|------------|------------|
| [Cron Jobs](../clouds/concepts/cron-jobs.md) |❌|✅|
| [Custom Authentication](../concepts/auth.md) |❌|✅|
| [Deployment options](../concepts/deployment_options.md) | Standalone container | Cloud Saas, Self-Hosted Data Plane, Self-Hosted Control Plane, Standalone container

## Application structure

To deploy a LangGraph Server application, you need to specify the graph(s) you want to deploy, as well as any relevant configuration settings, such as dependencies and environment variables.

Read the [application structure](./application_structure.md) guide to learn how to structure your LangGraph application for deployment.

## Parts of a deployment

When you deploy LangGraph Server, you are deploying one or more [graphs](#graphs), a database for [persistence](persistence.md), and a task queue.

### Graphs

When you deploy a graph with LangGraph Server, you are deploying a "blueprint" for an [Assistant](assistants.md). 

An [Assistant](assistants.md) is a graph paired with specific configuration settings. You can create multiple assistants per graph, each with unique settings to accommodate different use cases
that can be served by the same graph.

Upon deployment, LangGraph Server will automatically create a default assistant for each graph using the graph's default configuration settings.

!!! note

    We often think of a graph as implementing an [agent](agentic_concepts.md), but a graph does not necessarily need to implement an agent. For example, a graph could implement a simple
    chatbot that only supports back-and-forth conversation, without the ability to influence any application control flow. In reality, as applications get more complex, a graph will often implement a more complex flow that may use [multiple agents](./multi_agent.md) working in tandem.

### Persistence and task queue

LangGraph Server leverages a database for [persistence](persistence.md) and a task queue.

Currently, only [Postgres](https://www.postgresql.org/) is supported as a database for LangGraph Server and [Redis](https://redis.io/) as the task queue.

If you're deploying using [LangGraph Cloud](./langgraph_cloud.md), these components are managed for you. If you're deploying LangGraph Server on your own infrastructure, you'll need to set up and manage these components yourself.

Please review the [deployment options](./deployment_options.md) guide for more information on how these components are set up and managed.

## Learn more

* LangGraph [Application Structure](./application_structure.md) guide explains how to structure your LangGraph application for deployment.
* The [LangGraph Cloud API Reference](../cloud/reference/api/api_ref.html) provides detailed information on the API endpoints and data models.
