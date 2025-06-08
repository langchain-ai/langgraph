# Assistants

!!! info "Prerequisites"

    - [LangGraph Server](./langgraph_server.md)
    - [Configuration](./low_level.md#configuration)

When building agents, it is common to make rapid changes that _do not_ alter the graph logic. For example, simply changing prompts or the LLM selection can have significant impacts on the behavior of the agent but does not require updating your graph's architecture. Assistants offer a straightforward way to manage these configurations separately from your graph's core logic.

Imagine a general-purpose writing agent built on a common graph architecture. While the structure remains the same, different writing styles—such as blog posts and tweets—require tailored configurations to optimize performance. To support these variations, you can create multiple assistants (e.g., one for blogs and another for tweets) that share the underlying graph but differ in model selection and system prompt.

![assistant versions](img/assistants.png)

## Configuring assistants

Assistants build on the LangGraph open source concept of [configuration](low_level.md#configuration).
While configuration is available in the open source LangGraph library, assistants are only present in [LangGraph Platform](langgraph_platform.md).
This is due to the fact that assistants are tightly coupled to your deployed graph. Upon deployment, LangGraph Server will automatically create a default assistant for each graph using the graph's default configuration settings.

In practice, an assistant is just an _instance_ of a graph with a specific configuration. Therefore, multiple assistants can reference the same graph but can contain different configurations (e.g. prompts, models, tools). The LangGraph Server API provides several endpoints for creating and managing assistants. See the [API reference](../cloud/reference/api/api_ref.html) and [this how-to](../cloud/how-tos/configuration_cloud.md) for more details on how to create assistants.

## Versioning assistants

Assistants support versioning to track changes over time.
Once you've created an assistant, subsequent edits to that assistant will create new versions. See [this how-to](../cloud/how-tos/configuration_cloud.md#create-a-new-version-for-your-assistant) for more details on how to manage assistant versions.

## Learn more

* The LangGraph Cloud API provides several endpoints for creating and managing assistants and their versions. See the [API reference](../cloud/reference/api/api_ref.html#tag/assistants) for more details.