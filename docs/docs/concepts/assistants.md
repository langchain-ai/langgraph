---
search:
  boost: 2
---

# Assistants

!!! info "Prerequisites"

    - [LangGraph Server](./langgraph_server.md)
    - [Configuration](./low_level.md#configuration)

When building agents, it is common to make rapid changes that _do not_ alter the graph logic. For example, simply changing prompts or the LLM selection can have significant impacts on the behavior of the agent but does not require updating your graph's architecture. Assistants offer a straightforward way to manage these configurations separately from your graph's core logic.

For example, imagine you have a general writing agent. You have created a general graph architecture that works well for writing, however there are different types of writing styles (e.g. blogs vs tweets). In order to tune the performance for each use case, you need to make changes to the models and prompts used. To accomplish this, you could create two assistants - one for blog writing and one for tweeting. These assistants would share the same graph structure but their configurations would be different.

![assistant versions](img/assistants.png)

## Configuring Assistants

Assistants build on the LangGraph open source concept of [configuration](low_level.md#configuration).
While configuration is available in the open source LangGraph library, assistants are only present in [LangGraph Platform](langgraph_platform.md).
This is due to the fact that Assistants are tightly coupled to your deployed graph. Upon deployment, LangGraph Server will automatically create a default assistant for each graph using the graph's default configuration settings.

In practice, an assistant is just an _instance_ of a graph with a specific configuration. Therefore, multiple assistants can reference the same graph but can contain different configurations (e.g. prompts, models, tools). The LangGraph Server API provides several endpoints for creating and managing assistants. See the [API reference](../cloud/reference/api/api_ref.html) and [this how-to](../cloud/how-tos/configuration_cloud.md) for more details on how to create assistants.

## Versioning Assistants

Assistants also support versioning, to make it easier to track changes over time.
Once you've created an assistant, subsequent edits to that assistant will create new versions. See [this how-to](../cloud/how-tos/assistant_versioning.md) for more details on how to manage assistant versions.
