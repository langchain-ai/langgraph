# How to stream messages from your graph

This guide covers how to stream messages from your graph. In order to use this mode, the state of the graph you are interacting with MUST have a `messages` key that is a list of messages.

E.g., the state should look something like:

=== "Python"

    ```python
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages
    from langchain_core.messages import AnyMessage

    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
    ```

=== "Javascript"

    ```js
    import { type BaseMessage } from "@langchain/core/messages";
    import { Annotation, messagesStateReducer } from "@langchain/langgraph";

    export const StateAnnotation = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => [],
    }),
    });
    ```

Alternatively, you can use an instance or subclass of `from langgraph.graph import MessagesState` (`MessagesState` is equivalent to the implementation above). Or in Javascript: `import { MessagesAnnotation } from "@langchain/langgraph";`.

With `stream_mode="messages"` two things will be streamed back:

- It outputs messages produced by any chat model called inside (unless tagged in a special way)
- It outputs messages returned from nodes (to allow for nodes to return `ToolMessages` and the like)

Read more about how the `messages` streaming mode works [here](https://langchain-ai.github.io/langgraph/cloud/concepts/api/#modemessages)

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

    {'thread_id': 'e1431c95-e241-4d1d-a252-27eceb1e5c86',
     'created_at': '2024-06-21T15:48:59.808924+00:00',
     'updated_at': '2024-06-21T15:48:59.808924+00:00',
     'metadata': {},
     'status': 'idle',
     'config': {}}

Let's also define a helper function for better formatting of the tool calls in messages (for CURL we will define a helper script called `process_stream.sh`)

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

=== "CURL"

    ```bash
    # process_stream.sh

    format_tool_calls() {
        echo "$1" | jq -r 'map("Tool Call ID: \(.id), Function: \(.name), Arguments: \(.args)") | join("\n")'
    }

    process_data_item() {
        local data_item="$1"

        if echo "$data_item" | jq -e '.role == "user"' > /dev/null; then
            echo "Human: $(echo "$data_item" | jq -r '.content')"
        else
            local tool_calls=$(echo "$data_item" | jq -r '.tool_calls // []')
            local invalid_tool_calls=$(echo "$data_item" | jq -r '.invalid_tool_calls // []')
            local content=$(echo "$data_item" | jq -r '.content // ""')
            local response_metadata=$(echo "$data_item" | jq -r '.response_metadata // {}')

            if [ -n "$content" ] && [ "$content" != "null" ]; then
                echo "AI: $content"
            fi

            if [ "$tool_calls" != "[]" ]; then
                echo "Tool Calls:"
                format_tool_calls "$tool_calls"
            fi

            if [ "$invalid_tool_calls" != "[]" ]; then
                echo "Invalid Tool Calls:"
                format_tool_calls "$invalid_tool_calls"
            fi

            if [ "$response_metadata" != "{}" ]; then
                local finish_reason=$(echo "$response_metadata" | jq -r '.finish_reason // "N/A"')
                echo "Response Metadata: Finish Reason - $finish_reason"
            fi
        fi
    }

    while IFS=': ' read -r key value; do
        case "$key" in
            event)
                event="$value"
                ;;
            data)
                if [ "$event" = "metadata" ]; then
                    run_id=$(echo "$value" | jq -r '.run_id')
                    echo "Metadata: Run ID - $run_id"
                    echo "------------------------------------------------"
                elif [ "$event" = "messages/partial" ]; then
                    echo "$value" | jq -c '.[]' | while read -r data_item; do
                        process_data_item "$data_item"
                    done
                    echo "------------------------------------------------"
                fi
                ;;
        esac
    done
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

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data "{
    \"assistant_id\": \"agent\",
    \"config\":{\"configurable\":{\"model_name\":\"openai\"}},
    \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"What's the weather in sf\"}]},
    \"stream_mode\": [
    \"messages\"
    ]
    }" | sed 's/\r$//' | ./process_stream.sh
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

