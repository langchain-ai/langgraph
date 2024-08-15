# LangGraph Cloud (beta)

!!! tip
    - LangGraph is an MIT-licensed open-source library, which we are committed to maintaining and growing for the community.
    - LangGraph Cloud is an optional managed hosting service for LangGraph, which provides additional features geared towards production deployments.
    - We are actively contributing improvements back to LangGraph informed by our work on LangGraph Cloud.
    - You can always deploy LangGraph applications on your own infrastructure using the open-source LangGraph project.

!!! warning "Under Construction"
    LangGraph Cloud documentation is under construction. Contents may change until general availability.


<video controls preload="auto" allowfullscreen="true" poster="how-tos/img/studio_forks_poster.png">
    <source src="how-tos/img/studio_forks.mp4" type="video/mp4">
</video>


## Overview

LangGraph Cloud is a managed service for deploying and hosting LangGraph applications. Deploying applications with LangGraph Cloud shortens the time-to-market for developers. With one click, deploy a production-ready API with built-in persistence for your LangGraph application. LangGraph Cloud APIs are horizontally scalable and deployed with durable storage.

The LangGraph Cloud API exposes functionality of your LangGraph application through [Assistants](./concepts/api.md#assistants). An assistant abstracts the cognitive architecture of your graph. Invoke an assistant by calling the pre-built [API endpoints](./reference/api/api_ref.md).

LangGraph Cloud is seamlessly integrated with [LangSmith](https://www.langchain.com/langsmith) and is accessible from within the LangSmith UI.

LangGraph Cloud projects can be tested and debugged using the [LangGraph Studio Desktop](https://github.com/langchain-ai/langgraph-studio). 

## Key Features

The LangGraph Cloud API supports key LangGraph features in addition to new functionality for enabling complex, agentic workflows.

- **Assistants and Threads**: Assistants abstract the cognitive architecture of graphs and threads track the state/history of graphs.
- **Streaming**: API support for [LangGraph streaming modes](../concepts/low_level.md#streaming) including setting multiple streaming modes at the same time.
- **Human-in-the-Loop**: API support for [LangGraph human-in-the-loop features](../concepts/agentic_concepts.md#human-in-the-loop).
- **Double Texting**: Configure how assistants respond when new input is received while processing a previous input. Interrupt, rollback, reject, or enqueue.
- **Background Runs/Cron Jobs**: A built-in task queue enables background runs and scheduled cron jobs.
- **Stateless Runs**: For simpler use cases, invoke an assistant without needing to create a thread.

## Documentation

- [Tutorials](./quick_start.md): Learn to build and deploy applications for LangGraph Cloud.
- [How-to Guides](./how-tos/index.md): Learn how to set up a LangGraph application for deployment and implement features of the LangGraph Cloud API such as streaming tokens, configuring double texting, and creating cron jobs. Go here if you want to copy and run a specific code snippet.
- [Conceptual Guides](./concepts/api.md): In-depth explanations of the core data models (e.g. assistants), key features of the LangGraph Cloud API (e.g. double texting), and the architecture of a LangGraph Cloud deployment.
- [Reference](./reference/api/api_ref.md): References for the LangGraph Cloud API, the corresponding Python and JS/TS SDKs, the LangGraph CLI, and deployment environment variables.
