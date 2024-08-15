# How to configure multiple streaming modes at the same time

This guide covers how to configure multiple streaming modes at the same time.

First let's set up our client and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // create thread
    const thread = await client.threads.create();
    console.log(thread)
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json'
    ```

Output:

    {'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4',
     'created_at': '2024-06-24T21:30:07.980789+00:00',
     'updated_at': '2024-06-24T21:30:07.980789+00:00',
     'metadata': {},
     'status': 'idle',
     'config': {}}

When configuring multiple streaming modes for a run, responses for each respective mode will be produced. In the following example, note that a `list` of modes (`messages`, `events`, `debug`) is passed to the `stream_mode` parameter and the response contains `events`, `debug`, `messages/complete`, `messages/metadata`, and `messages/partial` event types.

=== "Python"

    ```python
    # create input
    input = {
        "messages": [
            {
                "role": "human",
                "content": "What's the weather in SF?",
            }
        ]
    }

    # stream events with multiple streaming modes
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input=input,
        stream_mode=["messages", "events", "debug"],
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
          "role": "human",
          "content": "What's the weather in SF?",
        }
      ]
    }

    // stream events with multiple streaming modes
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      "agent",
      {
        input,
        streamMode: ["messages", "events", "debug"]
      }
    );
    for await (const chunk of streamResponse) {
      console.log(f"Receiving new event of type: {chunk.event}...")
      console.log(chunk.data)
      console.log("\n\n")
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"What's the weather in SF?\"}]},
       \"stream_mode\": [
         \"messages\",
         \"events\",
         \"debug\"
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
    {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'role': 'human', 'content': "What's the weather in SF?"}]}}, 'name': 'LangGraph', 'tags': [], 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'parent_ids': []}
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.116009+00:00', 'step': -1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc7c-6daa-bfff-6b9027c1a50e', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'messages': []}, 'metadata': {'source': 'input', 'step': -1, 'writes': {'messages': [{'role': 'human', 'content': "What's the weather in SF?"}]}}}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.116009+00:00', 'step': -1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc7c-6daa-bfff-6b9027c1a50e', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'messages': []}, 'metadata': {'source': 'input', 'step': -1, 'writes': {'messages': [{'role': 'human', 'content': "What's the weather in SF?"}]}}}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['values', {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: messages/complete...
    [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.117924+00:00', 'step': 0, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc81-68c8-8000-4e18ae7d67a5', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]}, 'metadata': {'source': 'loop', 'step': 0, 'writes': None}}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.117924+00:00', 'step': 0, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc81-68c8-8000-4e18ae7d67a5', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]}, 'metadata': {'source': 'loop', 'step': 0, 'writes': None}}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-06-24T21:34:06.118042+00:00', 'step': 1, 'payload': {'id': '212ed9c2-a454-50c5-a202-12066bbbe7b8', 'name': 'agent', 'input': {'some_bytes': None, 'some_byte_array': None, 'dict_with_bytes': None, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}], 'sleep': None}, 'triggers': ['start:agent']}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task', 'timestamp': '2024-06-24T21:34:06.118042+00:00', 'step': 1, 'payload': {'id': '212ed9c2-a454-50c5-a202-12066bbbe7b8', 'name': 'agent', 'input': {'some_bytes': None, 'some_byte_array': None, 'dict_with_bytes': None, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}], 'sleep': None}, 'triggers': ['start:agent']}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'agent', 'tags': ['graph:step:1'], 'run_id': '72b74d24-5792-48da-a887-102100d6e2c0', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_start', 'data': {'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]]}}, 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'b', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: messages/metadata...
    {'run-2424dd6d-5cf5-4244-8d98-357640ce6e12': {'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}}}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'b', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'e', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'be', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'g', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'beg', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'i', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'begi', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'n', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_end', 'data': {'output': {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, 'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]]}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'name': 'should_continue', 'tags': ['seq:step:3'], 'run_id': '227afb0f-f909-4d54-a042-556ca6d98a69', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': 'tool', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'run_id': '227afb0f-f909-4d54-a042-556ca6d98a69', 'name': 'should_continue', 'tags': ['seq:step:3'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '72b74d24-5792-48da-a887-102100d6e2c0', 'name': 'agent', 'tags': ['graph:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}, 'input': {'some_bytes': None, 'some_byte_array': None, 'dict_with_bytes': None, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}], 'sleep': None}}, 'run_id': '72b74d24-5792-48da-a887-102100d6e2c0', 'name': 'agent', 'tags': ['graph:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.124350+00:00', 'step': 1, 'payload': {'id': '212ed9c2-a454-50c5-a202-12066bbbe7b8', 'name': 'agent', 'result': [['some_bytes', 'c29tZV9ieXRlcw=='], ['some_byte_array', 'c29tZV9ieXRlX2FycmF5'], ['dict_with_bytes', {'more_bytes': 'bW9yZV9ieXRlcw=='}], ['messages', [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]]]}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.124350+00:00', 'step': 1, 'payload': {'id': '212ed9c2-a454-50c5-a202-12066bbbe7b8', 'name': 'agent', 'result': [['some_bytes', 'c29tZV9ieXRlcw=='], ['some_byte_array', 'c29tZV9ieXRlX2FycmF5'], ['dict_with_bytes', {'more_bytes': 'bW9yZV9ieXRlcw=='}], ['messages', [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]]]}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['values', {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: messages/complete...
    [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.124510+00:00', 'step': 1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc91-6a34-8001-26353c117c25', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}, 'metadata': {'source': 'loop', 'step': 1, 'writes': {'agent': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}}}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.124510+00:00', 'step': 1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc91-6a34-8001-26353c117c25', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}, 'metadata': {'source': 'loop', 'step': 1, 'writes': {'agent': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}}}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-06-24T21:34:06.124572+00:00', 'step': 2, 'payload': {'id': '44139125-a1be-57c2-9cb2-19eb62bbaf2f', 'name': 'tool', 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'sleep': None}, 'triggers': ['branch:agent:should_continue:tool']}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task', 'timestamp': '2024-06-24T21:34:06.124572+00:00', 'step': 2, 'payload': {'id': '44139125-a1be-57c2-9cb2-19eb62bbaf2f', 'name': 'tool', 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'sleep': None}, 'triggers': ['branch:agent:should_continue:tool']}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'tool', 'tags': ['graph:step:2'], 'run_id': '91575720-886e-485e-ae2d-d6817e5346bf', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 2, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '91575720-886e-485e-ae2d-d6817e5346bf', 'name': 'tool', 'tags': ['graph:step:2'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 2, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': None, 'tool_call_id': 'tool_call_id'}]}}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}, 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'sleep': None}}, 'run_id': '91575720-886e-485e-ae2d-d6817e5346bf', 'name': 'tool', 'tags': ['graph:step:2'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 2, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.126828+00:00', 'step': 2, 'payload': {'id': '44139125-a1be-57c2-9cb2-19eb62bbaf2f', 'name': 'tool', 'result': [['messages', [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]]]}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.126828+00:00', 'step': 2, 'payload': {'id': '44139125-a1be-57c2-9cb2-19eb62bbaf2f', 'name': 'tool', 'result': [['messages', [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]]]}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['values', {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: messages/complete...
    [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.126966+00:00', 'step': 2, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc97-6a06-8002-8e9ffc1ea75a', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}, 'metadata': {'source': 'loop', 'step': 2, 'writes': {'tool': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}}}}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.126966+00:00', 'step': 2, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc97-6a06-8002-8e9ffc1ea75a', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}, 'metadata': {'source': 'loop', 'step': 2, 'writes': {'tool': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}}}}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-06-24T21:34:06.127034+00:00', 'step': 3, 'payload': {'id': 'f1ccf371-63b3-5268-a837-7f360a93c4ec', 'name': 'agent', 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}], 'sleep': None}, 'triggers': ['tool']}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task', 'timestamp': '2024-06-24T21:34:06.127034+00:00', 'step': 3, 'payload': {'id': 'f1ccf371-63b3-5268-a837-7f360a93c4ec', 'name': 'agent', 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}], 'sleep': None}, 'triggers': ['tool']}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'agent', 'tags': ['graph:step:3'], 'run_id': 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_start', 'data': {'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]]}}, 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'run_id': '0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'e', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: messages/metadata...
    {'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575': {'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}}}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'e', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'n', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'en', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'd', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_end', 'data': {'output': {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, 'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]]}}, 'run_id': '0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'name': 'should_continue', 'tags': ['seq:step:3'], 'run_id': '8af814e9-8136-4aab-acbc-dffc5bcafdfd', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': '__end__', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'run_id': '8af814e9-8136-4aab-acbc-dffc5bcafdfd', 'name': 'should_continue', 'tags': ['seq:step:3'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e', 'name': 'agent', 'tags': ['graph:step:3'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}, 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}], 'sleep': None}}, 'run_id': 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e', 'name': 'agent', 'tags': ['graph:step:3'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.133991+00:00', 'step': 3, 'payload': {'id': 'f1ccf371-63b3-5268-a837-7f360a93c4ec', 'name': 'agent', 'result': [['some_bytes', 'c29tZV9ieXRlcw=='], ['some_byte_array', 'c29tZV9ieXRlX2FycmF5'], ['dict_with_bytes', {'more_bytes': 'bW9yZV9ieXRlcw=='}], ['messages', [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]]]}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.133991+00:00', 'step': 3, 'payload': {'id': 'f1ccf371-63b3-5268-a837-7f360a93c4ec', 'name': 'agent', 'result': [['some_bytes', 'c29tZV9ieXRlcw=='], ['some_byte_array', 'c29tZV9ieXRlX2FycmF5'], ['dict_with_bytes', {'more_bytes': 'bW9yZV9ieXRlcw=='}], ['messages', [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]]]}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['values', {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: messages/complete...
    [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.134190+00:00', 'step': 3, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bca9-6418-8003-8d0d0b06845c', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}, 'metadata': {'source': 'loop', 'step': 3, 'writes': {'agent': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}}}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.134190+00:00', 'step': 3, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bca9-6418-8003-8d0d0b06845c', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}, 'metadata': {'source': 'loop', 'step': 3, 'writes': {'agent': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}}}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'parent_ids': []}
    
    
    
    Receiving new event of type: end...
    None
    
    
    

