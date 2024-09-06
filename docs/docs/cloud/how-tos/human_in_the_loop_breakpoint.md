# How to Add Breakpoints
 

When creating LangGraph agents, it is often nice to add a human-in-the-loop component.
This can be helpful when giving them access to tools.
Often in these situations you may want to manually approve an action before taking.

This can be in several ways, but the primary supported way is to add an "interrupt" before a node is executed.
This interrupts execution at that node.
You can then resume from that spot to continue.  

## Setup

### Code for your graph

In this how-to we use a simple ReAct style hosted graph (you can see the full code for defining it [here](../../how-tos/human_in_the_loop/breakpoints.ipynb)). The important thing is that there are two nodes (one named `agent` that calls the LLM, and one named `action` that calls the tool), and a routing function from `agent` that determines whether to call `action` next or just end the graph run (the `action` node always calls the `agent` node after execution).

### SDK Initialization


=== "Python"

    ```python
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
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

## Adding a breakpoint

We now want to add a breakpoint in our graph run, which we will do before a tool is called.
We can do this by adding `interrupt_before=["action"]`, which tells us to interrupt before calling the action node.
We can do this either when compiling the graph or when kicking off a run.
Here we will do it when kicking of a run, if you would like to to do it at compile time you need to edit the python file where your graph is defined and add the `interrupt_before` parameter when you call `.compile`.

First let's access our hosted LangGraph instance through the SDK:

And, now let's compile it with a breakpoint before the tool node:

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
        interrupt_before=["action"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```
=== "Javascript"

    ```js
    const input = { messages: [{ role: "human", content: "what's the weather in sf" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
        interruptBefore: ["action"]
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf\"}]},
       \"interrupt_before\": [\"action\"],
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
    {'run_id': '3b77ef83-687a-4840-8858-0371f91a92c3'}
    
    
    
    Receiving new event of type: data...
    {'agent': {'messages': [{'content': [{'id': 'toolu_01HwZqM1ptX6E15A5LAmyZTB', 'input': {'query': 'weather in san francisco'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-e5d17791-4d37-4ad2-815f-a0c4cba62585', 'example': False, 'tool_calls': [{'name': 'tavily_search_results_json', 'args': {'query': 'weather in san francisco'}, 'id': 'toolu_01HwZqM1ptX6E15A5LAmyZTB'}], 'invalid_tool_calls': []}]}}
    
    
    
    Receiving new event of type: end...
    None
    
    
    

