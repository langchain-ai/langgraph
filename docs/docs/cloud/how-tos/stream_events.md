# How to stream events
This guide covers how to stream events from your graph (`stream_mode="events"`). Depending on the use case and user experience of your LangGraph application, your application may process event types differently.

=== "Python"
    
    ```python
    from langgraph_sdk import get_client

    client = get_client(url="whatever-your-deployment-url-is")
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({apiUrl:"whatever-your-deployment-url-is"});
    # create thread
    const thread = await client.threads.create();
    console.log(thread)
    ```


Output:


    {'thread_id': '3f4c64e0-f792-4a5e-aa07-a4404e06e0bd',
     'created_at': '2024-06-24T22:16:29.301522+00:00',
     'updated_at': '2024-06-24T22:16:29.301522+00:00',
     'metadata': {}}



Streaming events produces responses containing an `event` key (in addition to other keys such as `data`). See the LangChain [`Runnable.astream_events()` reference](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events) for all event types.


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

    # stream events
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input=input,
        stream_mode="events",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    # create input
    const input = {
        "messages": [
            {
                "role": "human",
                "content": "What's the weather in SF?",
            }
        ]
    }

    # stream events
    const streamResponse = client.runs.stream(
        thread["thread_id"],
        "agent",
        {
            input: input,
            streamMode: "events"
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
        assistant_id="agent",
        input=input,
        stream_mode="events",
    ):
        if (
            chunk.event == "events" and
            chunk.data["event"] == "on_chat_model_stream"
        ):
            llm_response += chunk.data["data"]["chunk"]["content"]
            print(llm_response)
    ```

=== "Javascript"

    ```js
    var llm_response = "";
    # stream events
    const streamResponse = client.runs.stream(
        thread["thread_id"],
        "agent",
        {
            input: input,
            streamMode: "events"
        }
    );
    for await (const chunk of streamResponse) {
        if (chunk.event === "events" && chunk.data.event === "on_chat_model_stream") {
            llm_response += chunk.data.data.chunk.content;
            console.log(llm_response);
        }
    }
    ```

Output:

    b
    be
    beg
    begi
    begin
    begine
    beginen
    beginend

