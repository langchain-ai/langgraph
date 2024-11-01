# Assistants

!!! info "Prerequisites"

    - [LangGraph Server](./langgraph_server.md)

When building agents, it is fairly common to make rapid changes that *do not* alter the graph logic. For example, simply changing prompts or the LLM selection can have significant impacts on the behavior of the agents. Assistants offer an easy way to make and save these types of changes to agent configuration. This can have at least two use-cases:

* Assistants give developers a quick and easy way to modify and version agents for experimentation.
* Assistants can be modified via LangGraph Studio, offering a no-code way to configure agents  (e.g., for business users). 

Assistants build off the concept of ["configuration"](low_level.md#configuration). 
While ["configuration"](low_level.md#configuration) is available in the open source LangGraph library as well,  assistants are only present in [LangGraph Platform](langgraph_platform.md).
This is because Assistants are tightly coupled to your deployed graph, and so we can only make them available when we are also deploying the graphs.

## Configuring Assistants

In practice, an assistant is just an *instance* of a graph with a specific configuration. Because of this, multiple assistants can reference the same graph but can contain different configurations, such as prompts, models, and other graph configuration options. The LangGraph Cloud API provides several endpoints for creating and managing assistants. See the [API reference](../cloud/reference/api/api_ref.html) and [this how-to](../cloud/how-tos/configuration_cloud.md) for more details on how to create assistants.

## Versioning Assistants

Once you've created an assistant, you can save and version it to track changes to the configuration over time. You can think about this at three levels:

1) The graph lays out the general agent application logic 
2) The agent configuration options represent parameters that can be changed 
3) Assistant versions save and track specific settings of the agent configuration options 

For example, let's imagine you have a general writing agent. You have created a general graph architecture that works well for writing. However, there are different types of writing, e.g. blogs vs tweets. In order to get the best performance on each use case, you need to make some minor changes to the models and prompts used. In this setup, you could create an assistant for each use case - one for blog writing and one for tweeting. These would share the same graph structure, but they may use different models and different prompts. Read [this how-to](../cloud/how-tos/assistant_versioning.md) to learn how you can use assistant versioning through both the [Studio](../concepts/langgraph_studio.md) and the SDK.

![assistant versions](img/assistants.png)


## Resources

For more information on assistants, see the following resources:

- [Assistants how-to guides](../how-tos/index.md#assistants)