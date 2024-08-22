# How to stream debug events

This guide covers how to stream debug events from your graph (`stream_mode="debug"`).

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


Output:

    {'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3',
     'created_at': '2024-06-21T22:10:27.696862+00:00',
     'updated_at': '2024-06-21T22:10:27.696862+00:00',
     'metadata': {}}

Streaming debug events produces responses containing `type` and `timestamp` keys. Debug events correspond to different steps in the graph's execution (e.g. `task`, `task_result`, `checkpoint`).


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

    # stream debug
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input=input,
        stream_mode="debug",
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

    // stream debug
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      "agent",
      {
        input,
        streamMode: "debug"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(f"Receiving new event of type: {chunk.event}...")
      console.log(chunk.data)
      console.log("\n\n")
    }
    ```

Output:

    Receiving new event of type: metadata...
    {'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a'}
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-21T22:11:09.256850+00:00', 'step': -1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3', 'thread_ts': '1ef301b2-9a2e-6bb6-bfff-8423bcf47561', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a'}, 'values': {'messages': []}, 'metadata': {'source': 'input', 'step': -1, 'writes': {'messages': [{'role': 'human', 'content': "What's the weather in SF?"}]}}}}
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-21T22:11:09.259723+00:00', 'step': 0, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3', 'thread_ts': '1ef301b2-9a35-6c86-8000-f4a85315dbeb', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a'}, 'values': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '906529f7-fbf2-41c9-a28c-b1fe8f891e4e', 'example': False}]}, 'metadata': {'source': 'loop', 'step': 0, 'writes': None}}}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-06-21T22:11:09.260021+00:00', 'step': 1, 'payload': {'id': '12ab1026-a551-5f96-9ad3-43424f094774', 'name': 'agent', 'input': {'some_bytes': None, 'some_byte_array': None, 'dict_with_bytes': None, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '906529f7-fbf2-41c9-a28c-b1fe8f891e4e', 'example': False}], 'sleep': None}, 'triggers': ['start:agent']}}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-06-21T22:11:09.267632+00:00', 'step': 1, 'payload': {'id': '12ab1026-a551-5f96-9ad3-43424f094774', 'name': 'agent', 'result': [['some_bytes', 'c29tZV9ieXRlcw=='], ['some_byte_array', 'c29tZV9ieXRlX2FycmF5'], ['dict_with_bytes', {'more_bytes': 'bW9yZV9ieXRlcw=='}], ['messages', [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-54bd965b-734a-4a0a-8d4d-840865054810', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]]]}}
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-21T22:11:09.268469+00:00', 'step': 1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3', 'thread_ts': '1ef301b2-9a4b-60ae-8001-dd378f965bf7', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '906529f7-fbf2-41c9-a28c-b1fe8f891e4e', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-54bd965b-734a-4a0a-8d4d-840865054810', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}, 'metadata': {'source': 'loop', 'step': 1, 'writes': {'agent': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-54bd965b-734a-4a0a-8d4d-840865054810', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}}}}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-06-21T22:11:09.268659+00:00', 'step': 2, 'payload': {'id': '494ad427-fe8d-5654-91e6-50495a2699f5', 'name': 'tool', 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '906529f7-fbf2-41c9-a28c-b1fe8f891e4e', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-54bd965b-734a-4a0a-8d4d-840865054810', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'sleep': None}, 'triggers': ['branch:agent:should_continue:tool']}}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-06-21T22:11:09.272916+00:00', 'step': 2, 'payload': {'id': '494ad427-fe8d-5654-91e6-50495a2699f5', 'name': 'tool', 'result': [['messages', [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '222ed3b8-450f-41cb-ac40-905def3c700a', 'tool_call_id': 'tool_call_id'}]]]}}
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-21T22:11:09.273113+00:00', 'step': 2, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3', 'thread_ts': '1ef301b2-9a56-6832-8002-8ab17e662980', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '906529f7-fbf2-41c9-a28c-b1fe8f891e4e', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-54bd965b-734a-4a0a-8d4d-840865054810', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '222ed3b8-450f-41cb-ac40-905def3c700a', 'tool_call_id': 'tool_call_id'}]}, 'metadata': {'source': 'loop', 'step': 2, 'writes': {'tool': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '222ed3b8-450f-41cb-ac40-905def3c700a', 'tool_call_id': 'tool_call_id'}]}}}}}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-06-21T22:11:09.273192+00:00', 'step': 3, 'payload': {'id': '677de327-99b7-5d97-9bbd-0092abb62d46', 'name': 'agent', 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '906529f7-fbf2-41c9-a28c-b1fe8f891e4e', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-54bd965b-734a-4a0a-8d4d-840865054810', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '222ed3b8-450f-41cb-ac40-905def3c700a', 'tool_call_id': 'tool_call_id'}], 'sleep': None}, 'triggers': ['tool']}}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-06-21T22:11:09.277262+00:00', 'step': 3, 'payload': {'id': '677de327-99b7-5d97-9bbd-0092abb62d46', 'name': 'agent', 'result': [['some_bytes', 'c29tZV9ieXRlcw=='], ['some_byte_array', 'c29tZV9ieXRlX2FycmF5'], ['dict_with_bytes', {'more_bytes': 'bW9yZV9ieXRlcw=='}], ['messages', [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-006e1758-b1ca-4c90-9ff3-d2e75b9ca9a7', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]]]}}
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-21T22:11:09.277519+00:00', 'step': 3, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3', 'thread_ts': '1ef301b2-9a61-6462-8003-1316d9875b7f', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef301b2-9a0c-68d6-bbb1-0763efc8489a'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '906529f7-fbf2-41c9-a28c-b1fe8f891e4e', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-54bd965b-734a-4a0a-8d4d-840865054810', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '222ed3b8-450f-41cb-ac40-905def3c700a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-006e1758-b1ca-4c90-9ff3-d2e75b9ca9a7', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}, 'metadata': {'source': 'loop', 'step': 3, 'writes': {'agent': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-006e1758-b1ca-4c90-9ff3-d2e75b9ca9a7', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}}}}
    
    
    
    Receiving new event of type: end...
    None
    
    
    

