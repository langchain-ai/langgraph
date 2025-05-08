# How to create Assistants

!!! info "Prerequisites"

    - [Assistants Overview](../../concepts/assistants.md)
    - [Configuration](../../concepts/low_level.md#configuration)

In this guide we will show how to create and configure an assistant.

First, as a brief refresher on the concept of configurations, consider the following simple `call_model` node and configuration schema. Observe that this node tries to read and use the `model_name` as defined by the `config` object's `configurable`.

=== "Python"

    ```python

    class ConfigSchema(TypedDict):
        model_name: str

    builder = StateGraph(AgentState, config_schema=ConfigSchema)

    def call_model(state, config):
        messages = state["messages"]
        model_name = config.get('configurable', {}).get("model_name", "anthropic")
        model = _get_model(model_name)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    ```

=== "Javascript"

    ```js
    import { Annotation } from "@langchain/langgraph";

    const ConfigSchema = Annotation.Root({
        model_name: Annotation<string>,
        system_prompt:
    });

    const builder = new StateGraph(AgentState, ConfigSchema)

    function callModel(state: State, config: RunnableConfig) {
      const messages = state.messages;
      const modelName = config.configurable?.model_name ?? "anthropic";
      const model = _getModel(modelName);
      const response = model.invoke(messages);
      // We return a list, because this will get added to the existing list
      return { messages: [response] };
    }
    ```

For more information on configurations, [see here](../../concepts/low_level.md#configuration).

## Creating an Assistant

### LangGraph SDK

To create an assistant, use the [LangGraph SDK](../../concepts/sdk.md) `create` method. See the [Python](../reference/sdk/python_sdk_ref.md#langgraph_sdk.client.AssistantsClient.create) and [JS](../reference/sdk/js_ts_sdk_ref.md#create) SDK reference docs for more information.

This example uses the same configuration schema as above, and creates an assistant with `model_name` set to `openai`.

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    openai_assistant = await client.assistants.create(
        # "agent" is the name of a graph we deployed
        "agent", config={"configurable": {"model_name": "openai"}}, name="Open AI Assistant"
    )

    print(openai_assistant)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    let openAIAssistant = await client.assistants.create({
        graphId: 'agent',
        name: "Open AI Assistant",
        config: { "configurable": { "model_name": "openai" } },
    });

    console.log(openAIAssistant);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants \
        --header 'Content-Type: application/json' \
        --data '{"graph_id":"agent", "config":{"configurable":{"model_name":"openai"}}, "name": "Open AI Assistant"}'
    ```

Output:

    {
        "assistant_id": "62e209ca-9154-432a-b9e9-2d75c7a9219b",
        "graph_id": "agent",
        "name": "Open AI Assistant"
        "config": {
            "configurable": {
                "model_name": "openai"
            }
        },
        "metadata": {}
        "created_at": "2024-08-31T03:09:10.230718+00:00",
        "updated_at": "2024-08-31T03:09:10.230718+00:00",
    }

### LangGraph Platform UI

You can also create assistants from the LangGraph Platform UI.

Inside your deployment, select the "Assistants" tab. This will load a table of all of the assistants in your deployment, across all graphs.

To create a new assistant, select the "+ New assistant" button. This will open a form where you can specify the graph this assistant is for, as well as provide a name, description, and the desired configuration for the assistant based on the configuration schema for that graph.

To confirm, click "Create assistant". This will take you to [LangGraph Studio](../../../concepts/langgraph_studio) where you can test the assistant. If you go back to the "Assistants" tab in the deployment, you will see the newly created assistant in the table.

## Using an Assistant

### LangGraph SDK

We have now created an assistant called "Open AI Assistant" that has `model_name` defined as `openai`. We can now use this assistant with this configuration:

=== "Python"

    ```python
    thread = await client.threads.create()
    input = {"messages": [{"role": "user", "content": "who made you?"}]}
    async for event in client.runs.stream(
        thread["thread_id"],
        # this is where we specify the assistant id to use
        openai_assistant["assistant_id"],
        input=input,
        stream_mode="updates",
    ):
        print(f"Receiving event of type: {event.event}")
        print(event.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    const thread = await client.threads.create();
    let input = { "messages": [{ "role": "user", "content": "who made you?" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      // this is where we specify the assistant id to use
      openAIAssistant["assistant_id"],
      {
        input,
        streamMode: "updates"
      }
    );

    for await (const event of streamResponse) {
      console.log(`Receiving event of type: ${event.event}`);
      console.log(event.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    thread_id=$(curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}' | jq -r '.thread_id') && \
    curl --request POST \
        --url "<DEPLOYMENT_URL>/threads/${thread_id}/runs/stream" \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <OPENAI_ASSISTANT_ID>,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "who made you?"
                    }
                ]
            },
            "stream_mode": [
                "updates"
            ]
        }' | \
        sed 's/\r$//' | \
        awk '
        /^event:/ {
            if (data_content != "") {
                print data_content "\n"
            }
            sub(/^event: /, "Receiving event of type: ", $0)
            printf "%s...\n", $0
            data_content = ""
        }
        /^data:/ {
            sub(/^data: /, "", $0)
            data_content = $0
        }
        END {
            if (data_content != "") {
                print data_content "\n\n"
            }
        }
    '
    ```

Output:

    Receiving event of type: metadata
    {'run_id': '1ef6746e-5893-67b1-978a-0f1cd4060e16'}



    Receiving event of type: updates
    {'agent': {'messages': [{'content': 'I was created by OpenAI, a research organization focused on developing and advancing artificial intelligence technology.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-e1a6b25c-8416-41f2-9981-f9cfe043f414', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}

### LangGraph Platform UI

Inside your deployment, select the "Assistants" tab. For the assistant you would like to use, click the "Studio" button. This will open LangGraph Studio with the selected assistant. When you submit an input (either in Graph or Chat mode), the selected assistant and its configuration will be used.
