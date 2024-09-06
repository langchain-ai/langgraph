# How to stream state updates of your graph

<div class="admonition tip">
    <p class="admonition-title">Setup <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how LangSmith can help you <a href="https://docs.smith.langchain.com
        ">here</a>. 
    </p>
</div>    

This guide covers how to use `stream_mode="updates"` for your graph, which will stream the updates to the graph state that are made after each node is executed. This differs from using `stream_mode="values"`: instead of streaming the entire value of the state at each superstep, it only streams the updates from each of the nodes that made an update to the state at that superstep. Read [this conceptual guide](https://langchain-ai.github.io/langgraph/concepts/low_level/#stream-and-astream) to learn more.

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
      'thread_id': '979e3c89-a702-4882-87c2-7a59a250ce16',
      'created_at': '2024-06-21T15:22:07.453100+00:00',
      'updated_at': '2024-06-21T15:22:07.453100+00:00',
      'metadata': {},
      'status': 'idle',
      'config': {},
      'values': None 
    }

## Stream graph in updates mode

Now we can stream by updates, which outputs updates made to the state by each node after it has executed:


=== "Python"

    ```python
    input = {
        "messages": [
            {
                "role": "user",
                "content": "what's the weather in la"
            }
        ]
    }
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
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
          content: "What's the weather in la"
        }
      ]
    };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        streamMode: "updates"
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"What's the weather in la\"}]},
       \"stream_mode\": [
         \"updates\"
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
    {'run_id': 'cfc96c16-ed9a-44bd-b5bb-c30e3c0725f0'}
    
    
    
    Receiving new event of type: data...
    {'agent': {'messages': [{'content': [{'id': 'toolu_0148tMmDK51iLQfG1yaNwRHM', 'input': {'query': 'weather in los angeles'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-1a9d32b0-7007-4a36-abde-8df812a0ed94', 'example': False, 'tool_calls': [{'name': 'tavily_search_results_json', 'args': {'query': 'weather in los angeles'}, 'id': 'toolu_0148tMmDK51iLQfG1yaNwRHM'}], 'invalid_tool_calls': []}]}}
    
    
    
    Receiving new event of type: data...
    {'action': {'messages': [{'content': '[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'Los Angeles\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 34.05, \'lon\': -118.24, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1716062239, \'localtime\': \'2024-05-18 12:57\'}, \'current\': {\'last_updated_epoch\': 1716061500, \'last_updated\': \'2024-05-18 12:45\', \'temp_c\': 18.9, \'temp_f\': 66.0, \'is_day\': 1, \'condition\': {\'text\': \'Overcast\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/day/122.png\', \'code\': 1009}, \'wind_mph\': 2.2, \'wind_kph\': 3.6, \'wind_degree\': 10, \'wind_dir\': \'N\', \'pressure_mb\': 1017.0, \'pressure_in\': 30.02, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 65, \'cloud\': 100, \'feelslike_c\': 18.9, \'feelslike_f\': 66.0, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 6.0, \'gust_mph\': 7.5, \'gust_kph\': 12.0}}"}]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'tavily_search_results_json', 'id': 'a36e8cd1-0e96-4417-9c15-f10a945d2b42', 'tool_call_id': 'toolu_0148tMmDK51iLQfG1yaNwRHM'}]}}
    
    
    
    Receiving new event of type: data...
    {'agent': {'messages': [{'content': 'The weather in Los Angeles is currently overcast with a temperature of around 66°F (18.9°C). There are light winds from the north at around 2-3 mph. The humidity is 65% and visibility is good at 9 miles. Overall, mild spring weather conditions in LA.', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-d5c1c2f0-b12d-41ce-990b-f36570e7483d', 'example': False, 'tool_calls': [], 'invalid_tool_calls': []}]}}
    
    
    
    Receiving new event of type: end...
    None