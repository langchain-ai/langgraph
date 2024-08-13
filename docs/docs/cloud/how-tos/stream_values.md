# How to stream full state of your graph

LangGraph Cloud supports multiple streaming modes. The main ones are:

- `values`: This streaming mode streams back values of the graph. This is the **full state of the graph** after each node is called.
- `updates`: This streaming mode streams back updates to the graph. This is the **update to the state of the graph** after each node is called.
- `messages`: This streaming mode streams back messages - both complete messages (at the end of a node) as well as **tokens** for any messages generated inside a node. This mode is primarily meant for powering chat applications.


This guide covers `stream_mode="values"`.

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

    const client = new Client({ apiUrl: "whatever-your-deployment-url-is" });
    // create thread
    const thread = await client.threads.create();
    console.log(thread)
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url whatever-your-deployment-url-is/threads \
      --header 'Content-Type: application/json' \
      --data '{
        "metadata": {}
      }'
    ```

Output:

    {'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4',
     'created_at': '2024-06-24T21:30:07.980789+00:00',
     'updated_at': '2024-06-24T21:30:07.980789+00:00',
     'metadata': {},
     'status': 'idle',
     'config': {}}

Now we can stream by values, which streams the full state of the graph after each node has finished executing:

=== "Python"

    ```python
    input = {"messages": [{"role": "human", "content": "what's the weather in la"}]}

    # stream values
    async for chunk in client.runs.stream(
        thread["thread_id"],
        "agent", 
        input=input,
        stream_mode="values"
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    const input = {"messages": [{"role": "human", "content": "what's the weather in la"}]}

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      "agent",
      {
        input,
        streamMode: "values"
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
     --url whatever-your-deployment-url-is/threads/_YOUR_THREAD_ID_/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in la\"}]},
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
    {'run_id': 'f08791ce-0a3d-44e0-836c-ff62cd2e2786'}
    
    
    
    Receiving new event of type: values...
    {'messages': [{'role': 'human', 'content': 'what's the weather in la'}]}
    
    
    
    Receiving new event of type: values...
    {'messages': [{'content': 'what's the weather in la', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'faa15565-8823-4aa1-87af-e21b40526fae', 'example': False}, {'content': [{'id': 'toolu_01E5mSaZWm5rWJnCqmt63v4g', 'input': {'query': 'weather in los angeles'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-3fe1db7a-6b8d-4d83-ba07-8657190ad811', 'example': False, 'tool_calls': [{'name': 'tavily_search_results_json', 'args': {'query': 'weather in los angeles'}, 'id': 'toolu_01E5mSaZWm5rWJnCqmt63v4g'}], 'invalid_tool_calls': []}]}
    
    
    
    Receiving new event of type: values...
    {'messages': [{'content': 'what's the weather in la', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'faa15565-8823-4aa1-87af-e21b40526fae', 'example': False}, {'content': [{'id': 'toolu_01E5mSaZWm5rWJnCqmt63v4g', 'input': {'query': 'weather in los angeles'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-3fe1db7a-6b8d-4d83-ba07-8657190ad811', 'example': False, 'tool_calls': [{'name': 'tavily_search_results_json', 'args': {'query': 'weather in los angeles'}, 'id': 'toolu_01E5mSaZWm5rWJnCqmt63v4g'}], 'invalid_tool_calls': []}, {'content': '[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'Los Angeles\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 34.05, \'lon\': -118.24, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1716310320, \'localtime\': \'2024-05-21 9:52\'}, \'current\': {\'last_updated_epoch\': 1716309900, \'last_updated\': \'2024-05-21 09:45\', \'temp_c\': 16.7, \'temp_f\': 62.1, \'is_day\': 1, \'condition\': {\'text\': \'Overcast\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/day/122.png\', \'code\': 1009}, \'wind_mph\': 8.1, \'wind_kph\': 13.0, \'wind_degree\': 250, \'wind_dir\': \'WSW\', \'pressure_mb\': 1015.0, \'pressure_in\': 29.97, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 65, \'cloud\': 100, \'feelslike_c\': 16.7, \'feelslike_f\': 62.1, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 5.0, \'gust_mph\': 12.5, \'gust_kph\': 20.2}}"}]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'tavily_search_results_json', 'id': '0d5dab31-5ff8-4ae2-a560-bc4bcba7c9d7', 'tool_call_id': 'toolu_01E5mSaZWm5rWJnCqmt63v4g'}]}
    
    
    
    Receiving new event of type: values...
    {'messages': [{'content': 'what's the weather in la', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'faa15565-8823-4aa1-87af-e21b40526fae', 'example': False}, {'content': [{'id': 'toolu_01E5mSaZWm5rWJnCqmt63v4g', 'input': {'query': 'weather in los angeles'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-3fe1db7a-6b8d-4d83-ba07-8657190ad811', 'example': False, 'tool_calls': [{'name': 'tavily_search_results_json', 'args': {'query': 'weather in los angeles'}, 'id': 'toolu_01E5mSaZWm5rWJnCqmt63v4g'}], 'invalid_tool_calls': []}, {'content': '[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'Los Angeles\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 34.05, \'lon\': -118.24, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1716310320, \'localtime\': \'2024-05-21 9:52\'}, \'current\': {\'last_updated_epoch\': 1716309900, \'last_updated\': \'2024-05-21 09:45\', \'temp_c\': 16.7, \'temp_f\': 62.1, \'is_day\': 1, \'condition\': {\'text\': \'Overcast\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/day/122.png\', \'code\': 1009}, \'wind_mph\': 8.1, \'wind_kph\': 13.0, \'wind_degree\': 250, \'wind_dir\': \'WSW\', \'pressure_mb\': 1015.0, \'pressure_in\': 29.97, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 65, \'cloud\': 100, \'feelslike_c\': 16.7, \'feelslike_f\': 62.1, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 5.0, \'gust_mph\': 12.5, \'gust_kph\': 20.2}}"}]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'tavily_search_results_json', 'id': '0d5dab31-5ff8-4ae2-a560-bc4bcba7c9d7', 'tool_call_id': 'toolu_01E5mSaZWm5rWJnCqmt63v4g'}, {'content': 'Based on the weather API results, the current weather in Los Angeles is overcast with a temperature of around 62°F (17°C). There are light winds from the west-southwest around 8-13 mph. The humidity is 65% and visibility is good at 9 miles. Overall, mild spring weather conditions in LA.', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-4d6d4c23-5aad-4042-b0d9-19407a9e08e3', 'example': False, 'tool_calls': [], 'invalid_tool_calls': []}]}
    
    
    
    Receiving new event of type: end...
    None
    
    
    


If we want to just get the final result, we can use this endpoint and just keep track of the last value we received


=== "Python"

    ```python
    final_answer = None
    async for chunk in client.runs.stream(
        thread["thread_id"],
        "agent",
        input=input,
        stream_mode="values"
    ):
        if chunk.event == "values":
            final_answer = chunk.data
    ```

=== "Javascript"

    ```js
    let finalAnswer;
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      "agent",
      {
        input,
        streamMode: "values"
      }
    );
    for await (const chunk of streamResponse) {
      finalAnswer = chunk.data;
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url whatever-your-deployment-url-is/threads/_YOUR_THREAD_ID_/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in la\"}]},
       \"stream_mode\": [
         \"updates\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^data:/ { 
         sub(/^data: /, "", $0)   
         data_content = $0          
     }    
     END {                                               
         if (data_content != "") {
             print data_content
         }
     }         
     '
    ```


Output:

    {'messages': [{'content': 'what's the weather in la',
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': 'e78c2f94-d810-42fc-a399-11f6bb1b1092',
       'example': False},
      {'content': [{'id': 'toolu_01SBMoAGr4U9x3ibztm2UUom',
         'input': {'query': 'weather in los angeles'},
         'name': 'tavily_search_results_json',
         'type': 'tool_use'}],
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'ai',
       'name': None,
       'id': 'run-80767ab8-09fc-40ec-9e45-657ddef5e0b1',
       'example': False,
       'tool_calls': [{'name': 'tavily_search_results_json',
         'args': {'query': 'weather in los angeles'},
         'id': 'toolu_01SBMoAGr4U9x3ibztm2UUom'}],
       'invalid_tool_calls': []},
      {'content': '[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'Los Angeles\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 34.05, \'lon\': -118.24, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1716310320, \'localtime\': \'2024-05-21 9:52\'}, \'current\': {\'last_updated_epoch\': 1716309900, \'last_updated\': \'2024-05-21 09:45\', \'temp_c\': 16.7, \'temp_f\': 62.1, \'is_day\': 1, \'condition\': {\'text\': \'Overcast\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/day/122.png\', \'code\': 1009}, \'wind_mph\': 8.1, \'wind_kph\': 13.0, \'wind_degree\': 250, \'wind_dir\': \'WSW\', \'pressure_mb\': 1015.0, \'pressure_in\': 29.97, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 65, \'cloud\': 100, \'feelslike_c\': 16.7, \'feelslike_f\': 62.1, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 5.0, \'gust_mph\': 12.5, \'gust_kph\': 20.2}}"}]',
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'tool',
       'name': 'tavily_search_results_json',
       'id': 'af25e94a-c119-48c3-bbd3-096e42f472ac',
       'tool_call_id': 'toolu_01SBMoAGr4U9x3ibztm2UUom'},
      {'content': 'Based on the weather API results, the current weather in Los Angeles is overcast with a temperature of around 62°F (17°C). There are light winds from the west-southwest around 8-13 mph. The humidity is 65% and visibility is good at 9 miles. Overall, mild spring weather conditions in LA.',
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'ai',
       'name': None,
       'id': 'run-b90f0037-e56a-4f3b-ad92-00d10d079a9e',
       'example': False,
       'tool_calls': [],
       'invalid_tool_calls': []}]}