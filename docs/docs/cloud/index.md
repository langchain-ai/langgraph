# LangGraph Cloud (alpha)

!!! danger "Important"
    LangGraph Cloud is a closed source, paid product in an invite-only stage. We are currently focused on providing high bandwidth support to make our select early customers successful. If you are interested in applying for access, please fill out [this form](https://airtable.com/app5PiMJxXukqPLq3/pagveJsW7XOjDspqw/form).

!!! warning "Under Construction"
    LangGraph Cloud documentation is under construction. Contents may change until general availability.

## Overview

LangGraph Cloud is a managed service for deploying and hosting LangGraph applications. Deploying applications with LangGraph Cloud shortens the time-to-market for developers. With one click, deploy a production-ready API with built-in persistence for your LangGraph application. LangGraph Cloud APIs are horizatonally scalable and deployed with durable storage.

The LangGraph Cloud API exposes functionality of your LangGraph application through [Assistants](./concepts/index.md#assistants). An assistant abstracts the cognitive architecture of your graph. Invoke an assistant by calling the pre-built [API endpoints](./reference/api/api_ref.md).

LangGraph Cloud is seamlessly integrated with [LangSmith](https://www.langchain.com/langsmith) and is accessible from within the LangSmith UI.

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
- [How-to Guides](./how-tos/cloud_examples/stream_values/): Implement specific features of the LangGraph Cloud API such as streaming tokens, configuring double texting, and creating cron jobs. Go here if you want to copy and run a specific code snippet.
- [Conceptual Guides](./concepts/): In-depth explanations of the core data models (e.g. assistants) and key features (e.g. double texting) of the LangGraph Cloud API.
- [Reference](./reference/api/api_ref.md): References for the LangGraph Cloud API, the corresponding Python and JS/TS SDKs, and the LangGraph CLI.
