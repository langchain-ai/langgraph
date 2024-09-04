## Reject

This guide assumes knowledge of what double-texting is, which you can learn about in the [double-texting conceptual guide][double-texting].

The guide covers the `reject` option for double texting, which rejects the new run of the graph by throwing an error and continues with the original run until completion. Below is a quick example of using the `reject` option.

First, we will define a quick helper function for printing out JS and CURL model outputs (you can skip this if using Python):

=== "Javascript":

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

=== "CURL"

    ```bash
    # PLACE THIS IN A FILE CALLED pretty_print.sh
    pretty_print() {
      local type="$1"
      local content="$2"
      local padded=" $type "
      local total_width=80
      local sep_len=$(( (total_width - ${#padded}) / 2 ))
      local sep=$(printf '=%.0s' $(eval "echo {1.."${sep_len}"}"))
      local second_sep=$sep
      if (( (total_width - ${#padded}) % 2 )); then
        second_sep="${second_sep}="
      fi

      echo "${sep}${padded}${second_sep}"
      echo
      echo "$content"
    }
    ```

Now, let's import our required packages and instantiate our client, assistant, and thread.

=== "Python"

    ```python
    import httpx
    from langchain_core.messages import convert_to_messages
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json'
    ```

Now we can run a thread and try to run a second one with the "reject" option, which should fail since we have already started a run:


=== "Python"

    ```python
    run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "human", "content": "what's the weather in sf?"}]},
    )
    try:
        await client.runs.create(
            thread["thread_id"],
            assistant_id,
            input={
                "messages": [{"role": "human", "content": "what's the weather in nyc?"}]
            },
            multitask_strategy="reject",
        )
    except httpx.HTTPStatusError as e:
        print("Failed to start concurrent run", e)
    ```

=== "Javascript"

    ```js
    const run = await client.runs.create(
      thread["thread_id"],
      assistantId,
      input={"messages": [{"role": "human", "content": "what's the weather in sf?"}]},
    );
    
    try {
      await client.runs.create(
        thread["thread_id"],
        assistantId,
        { 
          input: {"messages": [{"role": "human", "content": "what's the weather in nyc?"}]},
          multitask_strategy:"reject"
        },
      );
    } catch (e) {
      console.error("Failed to start concurrent run", e);
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOY<ENT_URL>>/threads/<THREAD_ID>/runs \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what\'s the weather in sf?\"}]},
    }" && curl --request POST \
    --url <DEPLOY<ENT_URL>>/threads/<THREAD_ID>/runs \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what\'s the weather in nyc?\"}]},
      \"multitask_strategy\": \"reject\"
    }" || { echo "Failed to start concurrent run"; echo "Error: $?" >&2; }
    ```

Output:

    Failed to start concurrent run Client error '409 Conflict' for url 'http://localhost:8123/threads/f9e7088b-8028-4e5c-88d2-9cc9a2870e50/runs'
    For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/409


We can verify that the original thread finished executing:

=== "Python"

    ```python
    # wait until the original run completes
    await client.runs.join(thread["thread_id"], run["run_id"])

    state = await client.threads.get_state(thread["thread_id"])

    for m in convert_to_messages(state["values"]["messages"]):
        m.pretty_print()
    ```

=== "Javascript"

    ```js
    await client.runs.join(thread["thread_id"], run["run_id"]);

    const state = await client.threads.getState(thread["thread_id"]);

    for (const m of state["values"]["messages"]) {
      prettyPrint(m);
    }
    ```

=== "CURL"

    ```bash
    source pretty_print.sh && curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>/join && \
    curl --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | \
    jq -c '.values.messages[]' | while read -r element; do
        type=$(echo "$element" | jq -r '.type')
        content=$(echo "$element" | jq -r '.content | if type == "array" then tostring else . end')
        pretty_print "$type" "$content"
    done
    ```

Output:

    ================================[1m Human Message [0m=================================
    
    what's the weather in sf?
    ==================================[1m Ai Message [0m==================================
    
    [{'id': 'toolu_01CyewEifV2Kmi7EFKHbMDr1', 'input': {'query': 'weather in san francisco'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01CyewEifV2Kmi7EFKHbMDr1)
     Call ID: toolu_01CyewEifV2Kmi7EFKHbMDr1
      Args:
        query: weather in san francisco
    =================================[1m Tool Message [0m=================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.accuweather.com/en/us/san-francisco/94103/june-weather/347629", "content": "Get the monthly weather forecast for San Francisco, CA, including daily high/low, historical averages, to help you plan ahead."}]
    ==================================[1m Ai Message [0m==================================
    
    According to the search results from Tavily, the current weather in San Francisco is:
    
    The average high temperature in San Francisco in June is around 65°F (18°C), with average lows around 54°F (12°C). June tends to be one of the cooler and foggier months in San Francisco due to the marine layer of fog that often blankets the city during the summer months.
    
    Some key points about the typical June weather in San Francisco:
    
    - Mild temperatures with highs in the 60s F and lows in the 50s F
    - Foggy mornings that often burn off to sunny afternoons
    - Little to no rainfall, as June falls in the dry season
    - Breezy conditions, with winds off the Pacific Ocean
    - Layers are recommended for changing weather conditions
    
    So in summary, you can expect mild, foggy mornings giving way to sunny but cool afternoons in San Francisco this time of year. The marine layer keeps temperatures moderate compared to other parts of California in June.

