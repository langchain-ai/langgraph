## Rollback

This guide assumes knowledge of what double-texting is, which you can learn about in the [double-texting conceptual guide][double-texting].

The guide covers the `rollback` option for double texting, which interrupts the prior run of the graph and starts a new one with the double-text. This option is very similar to the `interrupt` option, but in this case the first run is completely deleted from the database and cannot be restarted. Below is a quick example of using the `rollback` option.

First, we will define a quick helper function for printing out JS model outputs (you can skip this if using Python):

```js
function prettyPrint(m) {
  const padded = " " + m['type'] + " ";
  const sepLen = Math.floor((80 - padded.length) / 2);
  const sep = "=".repeat(sepLen);
  const secondSep = sep + (padded.length % 2 ? "=" : "");
  
  console.log(`${sep}${padded}${secondSep}`);
  console.log("\n\n");
  console.log(m.content);
}
```

Now, let's import our required packages and instantiate our client, assistant, and thread.

=== "Python"

    ```python
    import asyncio

    import httpx
    from langchain_core.messages import convert_to_messages
    from langgraph_sdk import get_client

    client = get_client(url="whatever-your-deployment-url-is")
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl:"whatever-your-deployment-url-is" });
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

Now let's run a thread with the multitask parameter set to "rollback":

=== "Python"

    ```python
    # the first run will be rolled back
    rolled_back_run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "human", "content": "what's the weather in sf?"}]},
    )
    await asyncio.sleep(2)
    run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "human", "content": "what's the weather in nyc?"}]},
        multitask_strategy="rollback",
    )
    # wait until the second run completes
    await client.runs.join(thread["thread_id"], run["run_id"])
    ```

=== "Javascript"

    ```js
    // the first run will be interrupted
    let rolledBackRun = await client.runs.create(
      thread["thread_id"],
      assistantId,
      { input: { messages: [{ role: "human", content: "what's the weather in sf?" }] } }
    );
    await new Promise(resolve => setTimeout(resolve, 2000)); 

    let run = await client.runs.create(
      thread["thread_id"],
      assistant_id,
      { 
        input: { messages: [{ role: "human", content: "what's the weather in nyc?" }] },
        multitaskStrategy: "rollback" 
      }
    );

    // wait until the second run completes
    await client.runs.join(thread["thread_id"], run["run_id"]);
    ```

We can see that the thread has data only from the second run

=== "Python"

    ```python
    state = await client.threads.get_state(thread["thread_id"])

    for m in convert_to_messages(state["values"]["messages"]):
        m.pretty_print()
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread["thread_id"]);

    for (const m of state['values']['messages']) {
      prettyPrint(m);
    }
    ```

Output:

    ================================[1m Human Message [0m=================================
    
    what's the weather in nyc?
    ==================================[1m Ai Message [0m==================================
    
    [{'id': 'toolu_01JzPqefao1gxwajHQ3Yh3JD', 'input': {'query': 'weather in nyc'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01JzPqefao1gxwajHQ3Yh3JD)
     Call ID: toolu_01JzPqefao1gxwajHQ3Yh3JD
      Args:
        query: weather in nyc
    =================================[1m Tool Message [0m=================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'New York', 'region': 'New York', 'country': 'United States of America', 'lat': 40.71, 'lon': -74.01, 'tz_id': 'America/New_York', 'localtime_epoch': 1718734479, 'localtime': '2024-06-18 14:14'}, 'current': {'last_updated_epoch': 1718733600, 'last_updated': '2024-06-18 14:00', 'temp_c': 29.4, 'temp_f': 84.9, 'is_day': 1, 'condition': {'text': 'Sunny', 'icon': '//cdn.weatherapi.com/weather/64x64/day/113.png', 'code': 1000}, 'wind_mph': 2.2, 'wind_kph': 3.6, 'wind_degree': 158, 'wind_dir': 'SSE', 'pressure_mb': 1025.0, 'pressure_in': 30.26, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 63, 'cloud': 0, 'feelslike_c': 31.3, 'feelslike_f': 88.3, 'windchill_c': 28.3, 'windchill_f': 82.9, 'heatindex_c': 29.6, 'heatindex_f': 85.3, 'dewpoint_c': 18.4, 'dewpoint_f': 65.2, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 7.0, 'gust_mph': 16.5, 'gust_kph': 26.5}}"}]
    ==================================[1m Ai Message [0m==================================
    
    The weather API results show that the current weather in New York City is sunny with a temperature of around 85°F (29°C). The wind is light at around 2-3 mph from the south-southeast. Overall it looks like a nice sunny summer day in NYC.


Verify that the original, rolled back run was deleted

=== "Python"

    ```python
    try:
        await client.runs.get(thread["thread_id"], rolled_back_run["run_id"])
    except httpx.HTTPStatusError as _:
        print("Original run was correctly deleted")
    ```

=== "Javascript"

    ```js
    try {
      await client.runs.get(thread["thread_id"], rolledBackRun["run_id"]);
    } catch (e) {
      console.log("Original run was correctly deleted");
    }
    ```

Output:

    Original run was correctly deleted

