# How to share state between threads

By default, state in a graph is scoped to a specific thread. LangGraph also allows you to specify a "scope" for a given key/value pair that exists between threads. This can be useful for storing information that is shared between threads. For instance, you may want to store information about a user's preferences expressed in one thread, and then use that information in another thread.

In this notebook we will go through an example of how to use a graph that has been deployed with shared state.

## Setup

First, make sure that you have a deployed graph that has a shared state key. Your state definition should look something like this:

```python
class AgentState(TypedDict):
    # This is scoped to a user_id, so it will be information specific to each user
    name: Annotated[dict, SharedValue.on("user_id")]
    # ... other state keys ...
```
!!! note "Typing shared state keys"
    Shared state keys MUST be of type dictionary.

Now we can setup our client and an initial thread to run the graph on:

```python
from langgraph_sdk import get_client

client = get_client(url=<DEPLOYMENT_URL>)
# create thread
thread = await client.threads.create()
```

## Usage

Now, let's run the graph on the first thread, and provide it some information about the users name:

```python
input = {"messages": [{"role": "human", "content": "hi my name is isaac"}]}
config = {"configurable": {"model_name": "anthropic","user_id": "123","thread_id": thread["thread_id"]}}
# stream values
async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent", # the graph name
    input=input,
    config=config,
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    print("\n\n")
```

Output:

    Receiving new event of type: metadata...
    {'run_id': '1ef6a2cf-55d7-6440-bcf7-803b544dacc5'}



    Receiving new event of type: values...
    {'messages': [{'content': 'hi my name is isaac', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4dd09a0f-5c60-4012-916c-56e38069847c', 'example': False}]}



    Receiving new event of type: values...
    {'messages': [{'content': 'hi my name is isaac', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4dd09a0f-5c60-4012-916c-56e38069847c', 'example': False}, {'content': [{'text': 'Okay, got it. Let me update your name using the UserName tool:', 'type': 'text', 'index': 0}, {'id': 'toolu_01T8vAVuqxyNhypzHCw16gTe', 'input': {}, 'name': 'UserName', 'type': 'tool_use', 'index': 1, 'partial_json': '{"first_name": "Isaac", "last_name": " "}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-d56c2e72-270f-4038-9eb5-7904ac4ace0e', 'example': False, 'tool_calls': [{'name': 'UserName', 'args': {'first_name': 'Isaac', 'last_name': ' '}, 'id': 'toolu_01T8vAVuqxyNhypzHCw16gTe', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 422, 'output_tokens': 89, 'total_tokens': 511}}]}



    Receiving new event of type: values...
    {'messages': [{'content': 'hi my name is isaac', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4dd09a0f-5c60-4012-916c-56e38069847c', 'example': False}, {'content': [{'text': 'Okay, got it. Let me update your name using the UserName tool:', 'type': 'text', 'index': 0}, {'id': 'toolu_01T8vAVuqxyNhypzHCw16gTe', 'input': {}, 'name': 'UserName', 'type': 'tool_use', 'index': 1, 'partial_json': '{"first_name": "Isaac", "last_name": " "}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-d56c2e72-270f-4038-9eb5-7904ac4ace0e', 'example': False, 'tool_calls': [{'name': 'UserName', 'args': {'first_name': 'Isaac', 'last_name': ' '}, 'id': 'toolu_01T8vAVuqxyNhypzHCw16gTe', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 422, 'output_tokens': 89, 'total_tokens': 511}}, {'content': 'Saved!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': 'a36eb8fd-c0c9-466b-9def-0c29dd3c5289', 'tool_call_id': 'toolu_01T8vAVuqxyNhypzHCw16gTe', 'artifact': None, 'status': 'success'}]}

Now, let's run the graph on a completely different thread (even configuring it to use a different model), and see that it remembered the user's name:

```python
thread = await client.threads.create()
input = {"messages": [{"role": "human", "content": "what is my name?"}]}
config = {"configurable": {"model_name": "openai","user_id": "123","thread_id": thread["thread_id"]}}
# stream values
async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent", 
    input=input,
    config=config,
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    print("\n\n")
```

Output:

    Receiving new event of type: metadata...
    {'run_id': '1ef6a2d4-e554-63c8-9a18-f071a722444b'}



    Receiving new event of type: values...
    {'messages': [{'content': 'what is my name?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '22cd9d43-f025-4779-a804-bb91b15f5060', 'example': False}]}



    Receiving new event of type: values...
    {'messages': [{'content': 'what is my name?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '22cd9d43-f025-4779-a804-bb91b15f5060', 'example': False}, {'content': 'Your first name is Isaac. However, it seems that your last name is currently not set. Would you like to update your last name?', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-b3dfe744-d48f-4e34-8219-2e42e259f6a9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}

As we can see the graph correctly remembered the users first name, and also correctly identified that the last name was not set.