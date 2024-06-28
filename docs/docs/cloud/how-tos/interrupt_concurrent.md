## Interrupt

This notebook assumes knowledge of what double-texting is, which you can learn about in the [double-texting conceptual guide](https://langchain-ai.github.io/langgraph/cloud/concepts/#double-texting).

The guide covers the `interrupt` option for double texting, which interrupts the prior run of the graph and starts a new one with the double-text. This option does not delete the first run, but rather keeps it in the database but sets its status to `interrupted`. Below is a quick example of using the `interrupt` option.

First, let's import our required packages and instantiate our client, assistant, and thread.


=== "Python"

    ```python
    import asyncio

    from langchain_core.messages import convert_to_messages
    from langgraph_sdk import get_client

    client = get_client(url="whatever-your-deployment-url-is")
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    import { coerceMessageLikeToMessage } from "@langchain/core/messages"

    const client = new Client({apiUrl:"whatever-your-deployment-url-is"});
    const assistant_id = "agent";
    const thread = await client.threads.create();
    ```

Now we can start our two runs and join the second on euntil it has completed:

=== "Python"

    ```python
    # the first run will be interrupted
    interrupted_run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "human", "content": "what's the weather in sf?"}]},
    )
    await asyncio.sleep(2)
    run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "human", "content": "what's the weather in nyc?"}]},
        multitask_strategychrom="interrupt",
    )
    # wait until the second run completes
    await client.runs.join(thread["thread_id"], run["run_id"])
    ```

=== "Javascript"

    ```js
    // the first run will be interrupted
    let interrupted_run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        { input: { messages: [{ role: "human", content: "what's the weather in sf?" }] } }
    );
    await new Promise(resolve => setTimeout(resolve, 2000)); 

    let run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        { 
            input: { messages: [{ role: "human", content: "what's the weather in nyc?" }] },
            multitaskStrategy: "interrupt" 
        }
    );

    // wait until the second run completes
    await client.runs.join(thread["thread_id"], run["run_id"]);
    ```

We can see that the thread has partial data from the first run + data from the second run


=== "Python"

    ```python
    state = await client.threads.get_state(thread["thread_id"])

    for m in convert_to_messages(state["values"]["messages"]):
        m.pretty_print()
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread["thread_id"]);

    const baseMessages = state["values"]["messages"].map((message) =>
        coerceMessageLikeToMessage(message);
    );

    for (const m in baseMessages) {
        let padded = " " + m._getType() + " ";
        let sepLen = Math.floor((80 - padded.length) / 2);
        let sep = "=".repeat(sepLen);
        let secondSep = sep + (padded.length % 2 ? "=" : "");
        
        console.log(`${sep}${padded}${secondSep}`);
        console.log("\n\n");
        console.log(m.content);
    }
    ```

Output:

    ================================[1m Human Message [0m=================================
    
    what's the weather in sf?
    ==================================[1m Ai Message [0m==================================
    
    [{'id': 'toolu_01MjNtVJwEcpujRGrf3x6Pih', 'input': {'query': 'weather in san francisco'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01MjNtVJwEcpujRGrf3x6Pih)
     Call ID: toolu_01MjNtVJwEcpujRGrf3x6Pih
      Args:
        query: weather in san francisco
    =================================[1m Tool Message [0m=================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.wunderground.com/hourly/us/ca/san-francisco/KCASANFR2002/date/2024-6-18", "content": "High 64F. Winds W at 10 to 20 mph. A few clouds from time to time. Low 49F. Winds W at 10 to 20 mph. Temp. San Francisco Weather Forecasts. Weather Underground provides local & long-range weather ..."}]
    ================================[1m Human Message [0m=================================
    
    what's the weather in nyc?
    ==================================[1m Ai Message [0m==================================
    
    [{'id': 'toolu_01KtE1m1ifPLQAx4fQLyZL9Q', 'input': {'query': 'weather in new york city'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01KtE1m1ifPLQAx4fQLyZL9Q)
     Call ID: toolu_01KtE1m1ifPLQAx4fQLyZL9Q
      Args:
        query: weather in new york city
    =================================[1m Tool Message [0m=================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.accuweather.com/en/us/new-york/10021/june-weather/349727", "content": "Get the monthly weather forecast for New York, NY, including daily high/low, historical averages, to help you plan ahead."}]
    ==================================[1m Ai Message [0m==================================
    
    The search results provide weather forecasts and information for New York City. Based on the top result from AccuWeather, here are some key details about the weather in NYC:
    
    - This is a monthly weather forecast for New York City for the month of June.
    - It includes daily high and low temperatures to help plan ahead.
    - Historical averages for June in NYC are also provided as a reference point.
    - More detailed daily or hourly forecasts with precipitation chances, humidity, wind, etc. can be found by visiting the AccuWeather page.
    
    So in summary, the search provides a convenient overview of the expected weather conditions in New York City over the next month to give you an idea of what to prepare for if traveling or making plans there. Let me know if you need any other details!


Verify that the original, interrupted run was interrupted

=== "Python"

    ```python
    print((await client.runs.get(thread["thread_id"], interrupted_run["run_id"]))["status"])
    ```

=== "Javascript"

    ```js
    console.log((await client.runs.get(thread['thread_id'], interrupted_run["run_id"]))["status"])
    ```

Output:

    'interrupted'

