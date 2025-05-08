# How to version Assistants

!!! info "Prerequisites"

    - [Assistants Overview](../../concepts/assistants.md)
    - [How to create an Assistant](./configuration_cloud.md)

In this guide we will show you how to create, manage and use multiple versions of an assistant. If you have not already, please first see [this](./configuration_cloud.md) guide on creating an assistant. For this example, assume you have a graph with the following configuration schema:

=== "Python"

    ```python
    class Config(BaseModel):
        model_name: Literal["anthropic", "openai"] = "anthropic"
        system_prompt: str

    agent = StateGraph(State, config_schema=Config)
    ```

=== "Javascript"

    ```js
    const ConfigAnnotation = Annotation.Root({
        modelName: Annotation<z.enum(["openai", "anthropic"])>({
            default: () => "anthropic",
        }),
        systemPrompt: Annotation<String>
    });

    // the rest of your code

    const agent = new StateGraph(StateAnnotation, ConfigAnnotation);
    ```

And that you have the following assistant already created:

    {
        "assistant_id": "62e209ca-9154-432a-b9e9-2d75c7a9219b",
        "graph_id": "agent",
        "name": "Open AI Assistant"
        "config": {
            "configurable": {
                "model_name": "openai",
                "system_prompt": "You are a helpful assistant."
            }
        },
        "metadata": {}
        "created_at": "2024-08-31T03:09:10.230718+00:00",
        "updated_at": "2024-08-31T03:09:10.230718+00:00",
    }

## Create a new version for your assistant

### LangGraph SDK

To edit the assistant, use the `update` method. This will create a new version of the assistant with the provided edits. See the [Python](../reference/sdk/python_sdk_ref.md#langgraph_sdk.client.AssistantsClient.update) and [JS](../reference/sdk/js_ts_sdk_ref.md#update) SDK reference docs for more information.

!!! note "Note"
    You must pass in the ENTIRE config (and metadata if you are using it). The update endpoint creates new versions completely from scratch and does not rely on previously versions.

For example, to update your assistant's system prompt:
=== "Python"

    ```python
    openai_assistant_v2 = await client.assistants.update(
        openai_assistant["assistant_id"],
        config={
            "configurable": {
                "model_name": "openai",
                "system_prompt": "You are an unhelpful assistant!",
            }
        },
    )
    ```

=== "Javascript"

    ```js
    const openaiAssistantV2 = await client.assistants.update(
        openai_assistant["assistant_id"],
        {
            config: {
                configurable: {
                    model_name: 'openai',
                    system_prompt: 'You are an unhelpful assistant!',
                },
        },
    });
    ```

=== "CURL"

    ```bash
    curl --request PATCH \
    --url <DEPOLYMENT_URL>/assistants/<ASSISTANT_ID> \
    --header 'Content-Type: application/json' \
    --data '{
    "config": {"model_name": "openai", "system_prompt": "You are an unhelpful assistant!"}
    }'
    ```

This will create a new version of the assistant with the updated parameters and set this as the active version of your assistant. If you now run your graph and pass in this assistant id, it will use this latest version.

### LangGraph Platform UI

You can also edit assistants from the LangGraph Platform UI.

Inside your deployment, select the "Assistants" tab. This will load a table of all of the assistants in your deployment, across all graphs.

To edit an existing assistant, select the "Edit" button for the specified assistant. This will open a form where you can edit the assistant's name, description, and configuration.

Additionally, if using LangGraph Studio, you can edit the assistants and create new versions via the "Manage Assistants" button.

## Use a previous assistant version

### LangGraph SDK

You can also change the active version of your assistant. To do so, use the `setLatest` method.

In the example above, to rollback to the first version of the assistant:

=== "Python"

    ```python
    await client.assistants.set_latest(openai_assistant['assistant_id'], 1)
    ```

=== "Javascript"

    ```js
    await client.assistants.setLatest(openaiAssistant['assistant_id'], 1);
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/assistants/<ASSISTANT_ID>/latest \
    --header 'Content-Type: application/json' \
    --data '{
    "version": 1
    }'
    ```

If you now run your graph and pass in this assistant id, it will use the first version of the assistant.

### LangGraph Platform UI

If using LangGraph Studio, to set the active version of your asssistant, click the "Manage Assistants" button and locate the assistant you would like to use. Select the assistant and the version, and then click the "Active" toggle. This will update the assistant to make the selected version active.

!!! warning "Deleting Assistants"
    Deleting as assistant will delete ALL of it's versions. There is currently no way to delete a single version, but by pointing your assistant to the correct version you can skip any versions that you don't wish to use.
