# How to create agents with configuration

One of the benefits of LangGraph API is that it lets you create agents with different configurations.
This is useful when you want to:

- Define a cognitive architecture once as a LangGraph
- Let that LangGraph be configurable across some attributes (for example, system message or LLM to use)
- Let users create agents with arbitrary configurations, save them, and then use them in the future

In this guide we will show how to do that for the default agent we have built in.

If you look at the agent we defined, you can see that inside the `call_model` node we have created the model based on some configuration. That node looks like:

=== "Python"

    ```python
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
    function callModel(state: State, config: RunnableConfig) {
      const messages = state.messages;
      const modelName = config.configurable?.model_name ?? "anthropic";
      const model = _getModel(modelName);
      const response = model.invoke(messages);
      // We return a list, because this will get added to the existing list
      return { messages: [response] };
    }
    ```

We are looking inside the config for a `model_name` parameter (which defaults to `anthropic` if none is found). That means that by default we are using Anthropic as our model provider. In this example we will see an example of how to create an example agent that is configured to use OpenAI.

First let's set up our client and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Select an assistant that is not configured
    assistants = await client.assistants.search()
    assistant = [a for a in assistants if not a["config"]][0]
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Select an assistant that is not configured
    const assistants = await client.assistants.search();
    const assistant = assistants.find(a => !a.config);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{
            "limit": 10,
            "offset": 0
        }' | jq -c 'map(select(.config == null or .config == {})) | .[0]'
    ```

We can now call `.get_schemas` to get schemas associated with this graph:

=== "Python"

    ```python
    schemas = await client.assistants.get_schemas(
        assistant_id=assistant["assistant_id"]
    )
    # There are multiple types of schemas
    # We can get the `config_schema` to look at the the configurable parameters
    print(schemas["config_schema"])
    ```

=== "Javascript"

    ```js
    const schemas = await client.assistants.getSchemas(
      assistant["assistant_id"]
    );
    // There are multiple types of schemas
    // We can get the `config_schema` to look at the the configurable parameters
    console.log(schemas.config_schema);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/assistants/<ASSISTANT_ID>/schemas | jq -r '.config_schema'
    ```

Output:

    {
        'model_name': 
            {
                'title': 'Model Name',
                'enum': ['anthropic', 'openai'],
                'type': 'string'
            }
    }

Now we can initialize an assistant with config:

=== "Python"

    ```python
    openai_assistant = await client.assistants.create(
        # "agent" is the name of a graph we deployed
        "agent", config={"configurable": {"model_name": "openai"}}
    )

    print(openai_assistant)
    ```

=== "Javascript"

    ```js
    let openAIAssistant = await client.assistants.create(
      // "agent" is the name of a graph we deployed
      "agent", { "configurable": { "model_name": "openai" } }
    );

    console.log(openAIAssistant);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants \
        --header 'Content-Type: application/json' \
        --data '{"graph_id":"agent","config":{"configurable":{"model_name":"open_ai"}}}'
    ```

Output:

    {
        "assistant_id": "62e209ca-9154-432a-b9e9-2d75c7a9219b",
        "graph_id": "agent",
        "created_at": "2024-08-31T03:09:10.230718+00:00",
        "updated_at": "2024-08-31T03:09:10.230718+00:00",
        "config": {
            "configurable": {
                "model_name": "open_ai"
            }
        },
        "metadata": {}
    }

We can verify the config is indeed taking effect:

=== "Python"

    ```python
    thread = await client.threads.create()
    input = {"messages": [{"role": "user", "content": "who made you?"}]}
    async for event in client.runs.stream(
        thread["thread_id"],
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
                        "role": "human",
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



