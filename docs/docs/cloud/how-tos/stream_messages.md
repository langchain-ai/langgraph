# How to stream messages from your graph

!!! info "Prerequisites"
    * [Streaming](../../concepts/streaming.md)

This guide covers how to stream messages from your graph. With `stream_mode="messages"`, messages from any chat model invocations inside your graph nodes will be streamed back.

Read more about how the `messages` streaming mode works [here](https://langchain-ai.github.io/langgraph/cloud/concepts/api/#modemessages)

## Setup

First let's set up our client and thread:

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
        'thread_id': 'e1431c95-e241-4d1d-a252-27eceb1e5c86',
        'created_at': '2024-06-21T15:48:59.808924+00:00',
        'updated_at': '2024-06-21T15:48:59.808924+00:00',
        'metadata': {},
        'status': 'idle',
        'config': {},
        'values': None
    }

## Stream graph in messages mode

Now we can stream by messages, which will return complete messages (at the end of node execution) as well as tokens for any messages generated inside a node:

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    config = {"configurable": {"model_name": "openai"}}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id=assistant_id,
        input=input,
        config=config,
        stream_mode="messages",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    const input = {
      messages: [
        {
          role: "human",
          content: "What's the weather in sf",
        }
      ]
    };
    const config = { configurable: { model_name: "openai" } };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        config,
        streamMode: "messages"
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in la\"}]},
       \"stream_mode\": [
         \"messages\"
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
    {"run_id": "1ef971e0-9a84-6154-9047-247b4ce89c4d", "attempt": 1}



    Receiving new event of type: messages/metadata...
    {
      "run-700157a5-df1a-4829-9e7c-1e07a1d934f7": {
        "metadata": {
          "graph_id": "agent",
          "langgraph_node": "agent",
          ...
        }
      }
    }

    ...

    Receiving new event of type: messages/partial...
    [
      {
        "tool_calls": [
          {
            "name": "tavily_search_results_json",
            "args": {
              "query": "weather"
            },
            "id": "toolu_01RJGmVJtTxccoHHixGkGqaC",
            "type": "tool_call"
          }
        ],
      }
    ]



    Receiving new event of type: messages/partial...
    [
      {
        "type": "ai",
        "tool_calls": [
          {
            "name": "tavily_search_results_json",
            "args": {
              "query": "weather in "
            },
            "id": "toolu_01RJGmVJtTxccoHHixGkGqaC",
            "type": "tool_call"
          }
        ],
        ...
      }
    ]

    ...

    Receiving new event of type: messages/partial...
    [
      {
        "type": "ai",
        "tool_calls": [
          {
            "name": "tavily_search_results_json",
            "args": {
              "query": "weather in san francisco"
            },
            "id": "toolu_01RJGmVJtTxccoHHixGkGqaC",
            "type": "tool_call"
          }
        ],
        ...
      }
    ]



    Receiving new event of type: messages/metadata...
    {
      "aa162b98-433d-4e3c-b204-0d41a6694156": {
        "metadata": {
          "graph_id": "agent",
          "langgraph_node": "action",
          ...
        }
      }
    }



    Receiving new event of type: messages/complete...
    [
      {
        "content": "[{\"url\": \"https://www.weatherapi.com/\", \"content\": \"{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1730334046, 'localtime': '2024-10-30 17:20'}, 'current': {'last_updated_epoch': 1730333700, 'last_updated': '2024-10-30 17:15', 'temp_c': 12.3, 'temp_f': 54.2, 'is_day': 1, 'condition': {'text': 'Partly Cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 9.6, 'wind_kph': 15.5, 'wind_degree': 238, 'wind_dir': 'WSW', 'pressure_mb': 1021.0, 'pressure_in': 30.15, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 93, 'cloud': 57, 'feelslike_c': 11.2, 'feelslike_f': 52.2, 'windchill_c': 11.2, 'windchill_f': 52.2, 'heatindex_c': 12.3, 'heatindex_f': 54.2, 'dewpoint_c': 11.2, 'dewpoint_f': 52.1, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 0.5, 'gust_mph': 12.9, 'gust_kph': 20.8}}\"}]",
        "type": "tool",
        "name": "tavily_search_results_json",
        "tool_call_id": "toolu_01RJGmVJtTxccoHHixGkGqaC",
      }
    ]


    Receiving new event of type: messages/metadata...
    {
      "run-f92646d2-6b13-4648-90c7-0280766bfaf2": {
        "metadata": {
          "graph_id": "agent",
          "langgraph_node": "agent",
          ...
        }
      }
    }



    Receiving new event of type: messages/partial...
    [
      {
        "content": [
          {
            "text": "\n\nThe search",
            "type": "text",
            "index": 0
          }
        ],
        "type": "ai",
        ...
      }
    ]



    Receiving new event of type: messages/partial...
    [
      {
        "content": [
          {
            "text": "\n\nThe search results provide",
            "type": "text",
            "index": 0
          }
        ],
        "type": "ai",
        ...
      }
    ]

    ...

    Receiving new event of type: messages/partial...
    [
      {
        "content": [
          {
            "text": "\n\nThe search results provide the current weather conditions in San Francisco. According to the data, as of 5:20pm on October 30, 2024, the weather in San Francisco is partly cloudy with a temperature of 54\\u00b0F (12\\u00b0C). The wind is blowing from the west-southwest at around 10 mph (15 km/h). The humidity is high at 93% and visibility is 6 miles (10 km). Overall, it seems to be a cool, partly cloudy day with moderate winds in San Francisco.",
            "type": "text",
            "index": 0
          }
        ],
        "type": "ai",
        ...
      }
    ]