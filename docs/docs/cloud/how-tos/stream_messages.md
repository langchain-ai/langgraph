# How to stream messages from your graph

LangGraph Cloud supports multiple streaming modes. The main ones are:

- `values`: This streaming mode streams back values of the graph. This is the **full state of the graph** after each node is called.
- `updates`: This streaming mode streams back updates to the graph. This is the **update to the state of the graph** after each node is called.
- `messages`: This streaming mode streams back messages - both complete messages (at the end of a node) as well as **tokens** for any messages generated inside a node. This mode is primarily meant for powering chat applications.


This guide covers `stream_mode="messages"`.

In order to use this mode, the state of the graph you are interacting with MUST have a `messages` key that is a list of messages.
E.g., the state should look something like:

=== "Python"

    ```python
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages
    from langchain_core.messages import AnyMessage

    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
    ```


Alternatively, you can use an instance or subclass of `from langgraph.graph import MessagesState` (`MessagesState` is equivalent to the implementation above).

> [!NOTE]
> LangGraph Cloud only supports hosting graphs written in Python at the moment.

With `stream_mode="messages"` two things will be streamed back:

- It outputs messages produced by any chat model called inside (unless tagged in a special way)
- It outputs messages returned from nodes (to allow for nodes to return `ToolMessages` and the like

First let's set up our client and thread:

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

    const client = new Client({ apiUrl:"whatever-your-deployment-url-is" });
    // create thread
    const thread = await client.threads.create();
    console.log(thread)
    ```

Output:

    {'thread_id': 'e1431c95-e241-4d1d-a252-27eceb1e5c86',
     'created_at': '2024-06-21T15:48:59.808924+00:00',
     'updated_at': '2024-06-21T15:48:59.808924+00:00',
     'metadata': {},
     'status': 'idle',
     'config': {}}

Let's also define a helper function for better formatting of the tool calls in messages

=== "Python"

    ```python
    def format_tool_calls(tool_calls):
        if tool_calls:
            formatted_calls = []
            for call in tool_calls:
                formatted_calls.append(
                    f"Tool Call ID: {call['id']}, Function: {call['name']}, Arguments: {call['args']}"
                )
            return "\n".join(formatted_calls)
        return "No tool calls"
    ```

=== "Javascript"

    ```js
    function formatToolCalls(toolCalls) {
      if (toolCalls && toolCalls.length > 0) {
        const formattedCalls = toolCalls.map(call => {
          return `Tool Call ID: ${call.id}, Function: ${call.name}, Arguments: ${call.args}`;
        });
        return formattedCalls.join("\n");
      }
      return "No tool calls";
    }
    ```

Now we can stream by messages, which will return complete messages (at the end of node execution) as well as tokens for any messages generated inside a node:

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    config = {"configurable": {"model_name": "openai"}}

    async for event in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=input,
        config=config,
        stream_mode="messages",
    ):
        if event.event == "metadata":
            print(f"Metadata: Run ID - {event.data['run_id']}")
            print("-" * 50)
        elif event.event == "messages/partial":
            for data_item in event.data:
                if "role" in data_item and data_item["role"] == "user":
                    print(f"Human: {data_item['content']}")
                else:
                    tool_calls = data_item.get("tool_calls", [])
                    invalid_tool_calls = data_item.get("invalid_tool_calls", [])
                    content = data_item.get("content", "")
                    response_metadata = data_item.get("response_metadata", {})

                    if content:
                        print(f"AI: {content}")

                    if tool_calls:
                        print("Tool Calls:")
                        print(format_tool_calls(tool_calls))

                    if invalid_tool_calls:
                        print("Invalid Tool Calls:")
                        print(format_tool_calls(invalid_tool_calls))

                    if response_metadata:
                        finish_reason = response_metadata.get("finish_reason", "N/A")
                        print(f"Response Metadata: Finish Reason - {finish_reason}")
            print("-" * 50)
    ```

=== "Javascript"

    ```js
    const input = {
      "messages": [
        {
          "role": "human",
          "content": "What's the weather in sf",
        }
      ]
    }
    const config = {"configurable": {"model_name": "openai"}}

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      "agent",
      {
        input,
        config,
        streamMode: "messages"
      }
    );
    for await (const event of streamResponse) {
      if (event.event === "metadata") {
        console.log(`Metadata: Run ID - ${event.data.run_id}`);
        console.log("-".repeat(50));
      } else if (event.event === "messages/partial") {
        event.data.forEach(dataItem => {
          if (dataItem.role && dataItem.role === "user") {
            console.log(`Human: ${dataItem.content}`);
          } else {
            const toolCalls = dataItem.tool_calls || [];
            const invalidToolCalls = dataItem.invalid_tool_calls || [];
            const content = dataItem.content || "";
            const responseMetadata = dataItem.response_metadata || {};

            if (content) {
              console.log(`AI: ${content}`);
            }

            if (toolCalls.length > 0) {
              console.log("Tool Calls:");
              console.log(formatToolCalls(toolCalls));
            }

            if (invalidToolCalls.length > 0) {
              console.log("Invalid Tool Calls:");
              console.log(formatToolCalls(invalidToolCalls));
            }

            if (responseMetadata) {
              const finishReason = responseMetadata.finish_reason || "N/A";
              console.log(`Response Metadata: Finish Reason - ${finishReason}`);
            }
          }
        });
        console.log("-".repeat(50));
      }
    }
    ```

Output:

    Metadata: Run ID - 1ef2fe5c-6a1d-6575-bc09-d7832711c17e
    --------------------------------------------------
    Invalid Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: 
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: {}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: {}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: {'query': ''}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: {'query': 'current'}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: {'query': 'current weather'}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: {'query': 'current weather in'}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: {'query': 'current weather in San'}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: {'query': 'current weather in San Francisco'}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: {'query': 'current weather in San Francisco'}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_cg14F20jMBqWYrNgEkdWHwB3, Function: tavily_search_results_json, Arguments: {'query': 'current weather in San Francisco'}
    Response Metadata: Finish Reason - tool_calls
    --------------------------------------------------
    --------------------------------------------------
    AI: The
    --------------------------------------------------
    AI: The current
    --------------------------------------------------
    AI: The current weather
    --------------------------------------------------
    AI: The current weather in
    --------------------------------------------------
    AI: The current weather in San
    --------------------------------------------------
    AI: The current weather in San Francisco
    --------------------------------------------------
    AI: The current weather in San Francisco is
    --------------------------------------------------
    AI: The current weather in San Francisco is over
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F).
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-s
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-south
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 k
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph).
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%,
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9 miles
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9 miles).
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9 miles). The
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9 miles). The UV
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9 miles). The UV index
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9 miles). The UV index is
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9 miles). The UV index is 
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9 miles). The UV index is 3
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9 miles). The UV index is 3.
    --------------------------------------------------
    AI: The current weather in San Francisco is overcast with a temperature of 13.9°C (57.0°F). The wind is blowing from the south-southwest at 6.9 mph (11.2 kph). The humidity is at 81%, and the visibility is 16 km (9 miles). The UV index is 3.
    Response Metadata: Finish Reason - stop
    --------------------------------------------------

