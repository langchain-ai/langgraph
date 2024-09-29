# How to stream events

This guide covers how to stream events from your graph (`stream_mode="events"`). Depending on the use case and user experience of your LangGraph application, your application may process event types differently. Read more about events in this [conceptual guide](https://langchain-ai.github.io/langgraph/concepts/low_level/#astream_events-for-streaming-tokens-of-llm-calls).

## Setup

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

Output:


    {
        'thread_id': '3f4c64e0-f792-4a5e-aa07-a4404e06e0bd',
        'created_at': '2024-06-24T22:16:29.301522+00:00',
        'updated_at': '2024-06-24T22:16:29.301522+00:00',
        'metadata': {},
        'status': 'idle',
        'config': {},
        'values': None
    }

## Stream graph in events mode

Streaming events produces responses containing an `event` key (in addition to other keys such as `data`). See the LangChain [`Runnable.astream_events()` reference](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events) for all event types.


=== "Python"

    ```python
    # create input
    input = {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in SF?",
            }
        ]
    }

    # stream events
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input=input,
        stream_mode="events",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    // create input
    const input = {
      "messages": [
        {
          "role": "user",
          "content": "What's the weather in SF?",
        }
      ]
    }

    // stream events
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        streamMode: "events"
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"What's the weather in sf\"}]},
       \"stream_mode\": [
         \"events\"
       ]
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
    {'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8'}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'role': 'human', 'content': "What's the weather in SF?"}]}}, 'name': 'LangGraph', 'tags': [], 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'agent', 'tags': ['graph:step:6'], 'run_id': '7bb08493-d507-4e28-b9e6-4a5eda9d04f0', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_start', 'data': {'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}]]}}, 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'b', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'e', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'g', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'i', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'n', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_end', 'data': {'output': {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, 'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}]]}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'name': 'should_continue', 'tags': ['seq:step:3'], 'run_id': 'c7fe4d2d-3fb8-4e53-946d-03de13527853', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': 'tool', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'run_id': 'c7fe4d2d-3fb8-4e53-946d-03de13527853', 'name': 'should_continue', 'tags': ['seq:step:3'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '7bb08493-d507-4e28-b9e6-4a5eda9d04f0', 'name': 'agent', 'tags': ['graph:step:6'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}, 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}], 'sleep': None}}, 'run_id': '7bb08493-d507-4e28-b9e6-4a5eda9d04f0', 'name': 'agent', 'tags': ['graph:step:6'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'tool', 'tags': ['graph:step:7'], 'run_id': 'f044fd3d-7271-488f-b8aa-e01572ff9112', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 7, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': 'f044fd3d-7271-488f-b8aa-e01572ff9112', 'name': 'tool', 'tags': ['graph:step:7'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 7, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': None, 'tool_call_id': 'tool_call_id'}]}}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}]}, 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'sleep': None}}, 'run_id': 'f044fd3d-7271-488f-b8aa-e01572ff9112', 'name': 'tool', 'tags': ['graph:step:7'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 7, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'agent', 'tags': ['graph:step:8'], 'run_id': '1f4f95d0-0ce1-4061-85d4-946446bbd3e5', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_start', 'data': {'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}]]}}, 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'run_id': '028a68fb-6435-4b46-a156-c3326f73985c', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'e', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '028a68fb-6435-4b46-a156-c3326f73985c', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'n', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '028a68fb-6435-4b46-a156-c3326f73985c', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'd', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '028a68fb-6435-4b46-a156-c3326f73985c', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_end', 'data': {'output': {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, 'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}]]}}, 'run_id': '028a68fb-6435-4b46-a156-c3326f73985c', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'name': 'should_continue', 'tags': ['seq:step:3'], 'run_id': 'f2b2dfaf-475d-422b-8bf5-02a31bcc7d1a', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': '__end__', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'run_id': 'f2b2dfaf-475d-422b-8bf5-02a31bcc7d1a', 'name': 'should_continue', 'tags': ['seq:step:3'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1f4f95d0-0ce1-4061-85d4-946446bbd3e5', 'name': 'agent', 'tags': ['graph:step:8'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}, 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}], 'sleep': None}}, 'run_id': '1f4f95d0-0ce1-4061-85d4-946446bbd3e5', 'name': 'agent', 'tags': ['graph:step:8'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}, 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'name': 'LangGraph', 'tags': [], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'parent_ids': []}
    
    
    
    Receiving new event of type: end...
    None


## Token-by-Token Streaming

Token-by-token streaming can be implemented with the `events` streaming mode. The `on_chat_model_stream` event type should be processed to stream LLM responses token-by-token.

=== "Python"

    ```python
    llm_response = ""

    # stream token-by-token
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input=input,
        stream_mode="events",
    ):
        if (
            chunk.event == "events" and
            chunk.data["event"] == "on_chat_model_stream" and
            len(chunk.data["data"]["chunk"]["content"]) > 0 and
            'text' in chunk.data["data"]["chunk"]["content"][0]
        ):
            llm_response += chunk.data["data"]["chunk"]["content"][0]['text']
            print(llm_response)
    ```

=== "Javascript"

    ```js
    const llmResponse = "";
    // stream events
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        streamMode: "events"
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.event === "events" && chunk.data.event === "on_chat_model_stream" && chunk.data.chunk.content.length > 0 && 'text' in chunk.data.chunk.content[0]) {
        llmResponse += chunk.data.data.chunk.content[0].text;
        console.log(llmResponse);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"What's the weather in sf\"}]},
       \"stream_mode\": [
         \"events\"
       ]
     }" | sed 's/\r$//' | awk '
    /^event:/ { event = $2 }
    /^data:/ {
        json_data = substr($0, index($0, $2))

        if (event == "events") {
        print json_data
        }
    }' | jq -r '
        select(.event == "on_chat_model_stream") |
        .data.chunk.content[] | .text // empty
    ' | awk '
    BEGIN { llm_response="" }
    $0 != "" && $0 != "null" {
        llm_response = llm_response $0
        print llm_response
    }'
    ```

Output:

    The
    The search
    The search results provide
    The search results provide the current weather conditions
    The search results provide the current weather conditions in San Francisco.
    The search results provide the current weather conditions in San Francisco. According
    The search results provide the current weather conditions in San Francisco. According to the data,
    The search results provide the current weather conditions in San Francisco. According to the data, as
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12,
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024,
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C).
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The win
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is bl
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 k
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph).
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70%
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70% and visibility
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70% and visibility is 6
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70% and visibility is 6 miles
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70% and visibility is 6 miles (10 km
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70% and visibility is 6 miles (10 km).
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70% and visibility is 6 miles (10 km). Overall
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70% and visibility is 6 miles (10 km). Overall, it appears
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70% and visibility is 6 miles (10 km). Overall, it appears to be a nice
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70% and visibility is 6 miles (10 km). Overall, it appears to be a nice sunny day in San
    The search results provide the current weather conditions in San Francisco. According to the data, as of 3:19 PM on August 12, 2024, the weather in San Francisco is sunny with a temperature of 60.8°F (16°C). The wind is blowing from the west-southwest at 13.4 mph (21.6 kph). The humidity is 70% and visibility is 6 miles (10 km). Overall, it appears to be a nice sunny day in San Francisco.


