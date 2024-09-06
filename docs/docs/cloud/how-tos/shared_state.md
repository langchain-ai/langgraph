# How to share state between threads

By default, state in a graph is scoped to a specific thread. LangGraph also allows you to specify a "scope" for a given key/value pair that exists between threads. This can be useful for storing information that is shared between threads. For instance, you may want to store information about a user's preferences expressed in one thread, and then use that information in another thread.

In this notebook we will go through an example of how to use a graph that has been deployed with shared state.

## Setup

First, make sure that you have a deployed graph that has a shared state key. Your state definition should look something like this (support for shared state channels in JS is coming soon!):

```python
class AgentState(TypedDict):
    # This is scoped to a user_id, so it will be information specific to each user
    info: Annotated[dict, SharedValue.on("user_id")]
    # ... other state keys ...
```
!!! note "Typing shared state keys"
    Shared state channels (keys) MUST be dictionaries (see `info` channel in the AgentState example above)

Now we can setup our client and an initial thread to run the graph on:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent";
    # create thread
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    let thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

## Usage

Now, let's run the graph on the first thread, and provide it some information about the users preferences:

=== "Python"

    ```python
    input = {"messages": [{"role": "human", "content": "i like pepperoni pizza"}]}
    config = {"configurable": {"user_id": "123"}}
    # stream values
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        config=config,
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    // create input
    let input = {
      messages: [
        {
          role: "human",
          content: "i like pepperoni pizza",
        }
      ]
    };
    let config = { configurable: { user_id: "123" } };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        config
      }
    );
    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"i like pepperoni pizza\"}]},
       \"config\":{\"configurable\":{\"user_id\":\"123\"}}
     }" | \
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
             print data_content "\n"
         }
     }
     ' 
    ```

Output:

    Receiving new event of type: metadata...
    {'run_id': '1ef6bdb2-ba0e-6177-84a9-c574772223b3'}



    Receiving new event of type: values...
    {'messages': [{'content': 'i like pepperoni pizza', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'f1244b8b-e54e-4ebe-ada4-63aadf4a7701', 'example': False}]}



    Receiving new event of type: values...
    {'messages': [{'content': 'i like pepperoni pizza', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'f1244b8b-e54e-4ebe-ada4-63aadf4a7701', 'example': False}, {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'function': {'arguments': '{"fact":"Isaac likes pepperoni pizza","topic":"Food"}', 'name': 'Info'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-3.5-turbo-0125'}, 'type': 'ai', 'name': None, 'id': 'run-f086646f-cb38-4419-9a92-fc7cb19340ee', 'example': False, 'tool_calls': [{'name': 'Info', 'args': {'fact': 'Isaac likes pepperoni pizza', 'topic': 'Food'}, 'id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}



    Receiving new event of type: values...
    {'messages': [{'content': 'i like pepperoni pizza', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'f1244b8b-e54e-4ebe-ada4-63aadf4a7701', 'example': False}, {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'function': {'arguments': '{"fact":"Isaac likes pepperoni pizza","topic":"Food"}', 'name': 'Info'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-3.5-turbo-0125'}, 'type': 'ai', 'name': None, 'id': 'run-f086646f-cb38-4419-9a92-fc7cb19340ee', 'example': False, 'tool_calls': [{'name': 'Info', 'args': {'fact': 'Isaac likes pepperoni pizza', 'topic': 'Food'}, 'id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'Saved!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': 'bed77d11-916c-4bad-b8f8-f0c850a8e494', 'tool_call_id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'artifact': None, 'status': 'success'}]}



Let's stay on the same thread and provide some additional information. Note that we are not redefining the config since we want to continue the conversation on the same thread with the same user.

=== "Python"

    ```python
    input = {"messages": [{"role": "human", "content": "i also just moved to SF"}]}
    # stream values
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id, # the graph name
        input=input,
        config=config,
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    input = {
      messages: [
        {
          role: "human",
          content: "i also just moved to SF",
        }
      ]
    };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        config
      }
    );

    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"i also just moved to SF\"}]},
       \"config\":{\"configurable\":{\"user_id\":\"123\"}}
     }" | \
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
             print data_content "\n"
         }
     }
     ' 
    ```

Output:

    Receiving new event of type: metadata...
    {'run_id': '1ef6bdb2-f068-60b6-93a6-b2e2f02f117d'}



    Receiving new event of type: values...
    {'messages': [{'content': 'i like pepperoni pizza', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'f1244b8b-e54e-4ebe-ada4-63aadf4a7701', 'example': False}, {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'function': {'arguments': '{"fact":"Isaac likes pepperoni pizza","topic":"Food"}', 'name': 'Info'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-3.5-turbo-0125'}, 'type': 'ai', 'name': None, 'id': 'run-f086646f-cb38-4419-9a92-fc7cb19340ee', 'example': False, 'tool_calls': [{'name': 'Info', 'args': {'fact': 'Isaac likes pepperoni pizza', 'topic': 'Food'}, 'id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'Saved!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': 'bed77d11-916c-4bad-b8f8-f0c850a8e494', 'tool_call_id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'artifact': None, 'status': 'success'}, {'content': 'i also just moved to SF', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'f1e98940-d9fc-454c-bb65-0036e2c048c6', 'example': False}]}



    Receiving new event of type: values...
    {'messages': [{'content': 'i like pepperoni pizza', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'f1244b8b-e54e-4ebe-ada4-63aadf4a7701', 'example': False}, {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'function': {'arguments': '{"fact":"Isaac likes pepperoni pizza","topic":"Food"}', 'name': 'Info'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-3.5-turbo-0125'}, 'type': 'ai', 'name': None, 'id': 'run-f086646f-cb38-4419-9a92-fc7cb19340ee', 'example': False, 'tool_calls': [{'name': 'Info', 'args': {'fact': 'Isaac likes pepperoni pizza', 'topic': 'Food'}, 'id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'Saved!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': 'bed77d11-916c-4bad-b8f8-f0c850a8e494', 'tool_call_id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'artifact': None, 'status': 'success'}, {'content': 'i also just moved to SF', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'f1e98940-d9fc-454c-bb65-0036e2c048c6', 'example': False}, {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_yPNnY10h9KszyuVf5p6H2c1E', 'function': {'arguments': '{"fact":"Isaac just moved to SF","topic":"Location"}', 'name': 'Info'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-3.5-turbo-0125'}, 'type': 'ai', 'name': None, 'id': 'run-d247f67f-ba1a-4ce7-84c2-0f30180d10c6', 'example': False, 'tool_calls': [{'name': 'Info', 'args': {'fact': 'Isaac just moved to SF', 'topic': 'Location'}, 'id': 'call_yPNnY10h9KszyuVf5p6H2c1E', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}



    Receiving new event of type: values...
    {'messages': [{'content': 'i like pepperoni pizza', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'f1244b8b-e54e-4ebe-ada4-63aadf4a7701', 'example': False}, {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'function': {'arguments': '{"fact":"Isaac likes pepperoni pizza","topic":"Food"}', 'name': 'Info'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-3.5-turbo-0125'}, 'type': 'ai', 'name': None, 'id': 'run-f086646f-cb38-4419-9a92-fc7cb19340ee', 'example': False, 'tool_calls': [{'name': 'Info', 'args': {'fact': 'Isaac likes pepperoni pizza', 'topic': 'Food'}, 'id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'Saved!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': 'bed77d11-916c-4bad-b8f8-f0c850a8e494', 'tool_call_id': 'call_ujnk8CIx0xeguFHHe8P0ecgm', 'artifact': None, 'status': 'success'}, {'content': 'i also just moved to SF', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'f1e98940-d9fc-454c-bb65-0036e2c048c6', 'example': False}, {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_yPNnY10h9KszyuVf5p6H2c1E', 'function': {'arguments': '{"fact":"Isaac just moved to SF","topic":"Location"}', 'name': 'Info'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-3.5-turbo-0125'}, 'type': 'ai', 'name': None, 'id': 'run-d247f67f-ba1a-4ce7-84c2-0f30180d10c6', 'example': False, 'tool_calls': [{'name': 'Info', 'args': {'fact': 'Isaac just moved to SF', 'topic': 'Location'}, 'id': 'call_yPNnY10h9KszyuVf5p6H2c1E', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'Saved!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '7c4b82b0-5ee5-4d97-902b-dbec1499fe39', 'tool_call_id': 'call_yPNnY10h9KszyuVf5p6H2c1E', 'artifact': None, 'status': 'success'}]}







Now, let's run the graph on a completely different thread, and see that it remembered the information we provided it:


=== "Python"

    ```python
    # new thread for new conversation
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "where and what should i eat for dinner? Can you list some restaurants?"}]}
    # stream values
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id, 
        input=input,
        config=config,
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    // new thread for new conversation
    thread = await client.threads.create();

    // create input
    let input = {
      messages: [
        {
          role: "human",
          content: "where and what should i eat for dinner? Can you list some restaurants?",
        }
      ]
    };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        config
      }
    );
    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads \
    --header 'Content-Type: application/json' \
    --data '{}' \
    | jq -r '.thread_id' \
    | xargs -I {} \
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/{}/runs/stream \
        --header 'Content-Type: application/json' \
        --data '{
        "assistant_id": "agent",
        "input": {
            "messages": [{
            "role": "human", 
            "content": "where and what should i eat for dinner? Can you list some restaurants?"
            }]
        },
        "config": {
            "configurable": {
            "user_id": "123"
            }
        }
        }' \
    | sed 's/\r$//' \
    | awk '
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
                print data_content "\n"
            }
        }
    '
    ```

Output:

    Receiving new event of type: metadata...
    {'run_id': '1ef6bde9-d866-623c-8647-a56e33322334'}



    Receiving new event of type: values...
    {'messages': [{'content': 'where and what should i eat for dinner? Can you list some restaurants?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'aaf07830-ddbf-4ec6-b520-2371490abaa8', 'example': False}]}



    Receiving new event of type: values...
    {'messages': [{'content': 'where and what should i eat for dinner? Can you list some restaurants?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'aaf07830-ddbf-4ec6-b520-2371490abaa8', 'example': False}, {'content': "Sure! Since you just moved to SF, I can suggest some popular restaurants in the area. Here are a few options:\n\n1. Tony's Pizza Napoletana - Known for their delicious pizzas, including pepperoni pizza.\n2. The House - Offers Asian fusion cuisine in a cozy setting.\n3. Tadich Grill - A historic seafood restaurant serving classic dishes.\n4. Swan Oyster Depot - A seafood counter known for its fresh seafood selections.\n5. Zuni Cafe - A popular spot for American and Mediterranean-inspired dishes.\n\nDo any of these options sound good to you? Let me know if you need more recommendations or information about any specific cuisine!", 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-3.5-turbo-0125'}, 'type': 'ai', 'name': None, 'id': 'run-dbac2e4c-0e4b-4c4d-b17f-172456222f53', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}



Perfect! The AI recommended restaurants in SF, and included a pizza restaurant at the top of it's list.

Let's now run the graph for another user to verify that the preferences of the first user are self contained:

=== "Python"

    ```python
    # new thread for new conversation
    thread = await client.threads.create()
    # create input
    input = {"messages": [{"role": "human", "content": "where do I live? what do I like to eat?"}]}
    config = {"configurable": {"user_id": "321"}}
    # stream values
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        config=config,
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    // new thread for new conversation
    thread = await client.threads.create();
    // create input
    let input = {
      messages: [
        {
          role: "human",
          content: "where do I live? what do I like to eat?",
        }
      ]
    };
    let config = { configurable: { user_id: "321" } };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        config
      }
    );
    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads \
    --header 'Content-Type: application/json' \
    --data '{}' \
    | jq -r '.thread_id' \
    | xargs -I {} \
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/{}/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"where do I live? what do I like to eat?\"}]},
       \"config\":{\"configurable\":{\"user_id\":\"321\"}}
     }" | \
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
             print data_content "\n"
         }
     }
     ' 
    ```

Output:

    Receiving new event of type: metadata...
    {'run_id': '1ef6bdf3-6aae-63ab-adc4-0a1467251531'}



    Receiving new event of type: values...
    {'messages': [{'content': 'where do I live? what do I like to eat?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '043fd6a6-b59b-411b-9ec3-f6947260e6d3', 'example': False}]}



    Receiving new event of type: values...
    {'messages': [{'content': 'where do I live? what do I like to eat?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '043fd6a6-b59b-411b-9ec3-f6947260e6d3', 'example': False}, {'content': "I don't have that information yet. Can you please provide me with details about where you live and what you like to eat?", 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-3.5-turbo-0125'}, 'type': 'ai', 'name': None, 'id': 'run-4dd3415d-e75b-44d5-9744-14f840e6c696', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}

Perfect! The agent does not have access to the first users preferences as we expect!

