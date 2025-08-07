# Time travel using Server API

LangGraph provides the [**time travel**](../../concepts/time-travel.md) functionality to resume execution from a prior checkpoint, either replaying the same state or modifying it to explore alternatives. In all cases, resuming past execution produces a new fork in the history.

To time travel using the LangGraph Server API (via the LangGraph SDK):

1. **Run the graph** with initial inputs using [LangGraph SDK](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/)'s @[`client.runs.wait`][client.runs.wait] or @[`client.runs.stream`][client.runs.stream] APIs.
2. **Identify a checkpoint in an existing thread**: Use @[`client.threads.get_history`][client.threads.get_history] method to retrieve the execution history for a specific `thread_id` and locate the desired `checkpoint_id`.
   Alternatively, set a [breakpoint](./human_in_the_loop_breakpoint.md) before the node(s) where you want execution to pause. You can then find the most recent checkpoint recorded up to that breakpoint.
3. **(Optional) modify the graph state**: Use the @[`client.threads.update_state`][client.threads.update_state] method to modify the graph’s state at the checkpoint and resume execution from alternative state.
4. **Resume execution from the checkpoint**: Use the @[`client.runs.wait`][client.runs.wait] or @[`client.runs.stream`][client.runs.stream] APIs with an input of `None` and the appropriate `thread_id` and `checkpoint_id`.

## Use time travel in a workflow

??? example "Example graph"

    ```python
    from typing_extensions import TypedDict, NotRequired
    from langgraph.graph import StateGraph, START, END
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import InMemorySaver

    class State(TypedDict):
        topic: NotRequired[str]
        joke: NotRequired[str]

    llm = init_chat_model(
        "anthropic:claude-3-7-sonnet-latest",
        temperature=0,
    )

    def generate_topic(state: State):
        """LLM call to generate a topic for the joke"""
        msg = llm.invoke("Give me a funny topic for a joke")
        return {"topic": msg.content}

    def write_joke(state: State):
        """LLM call to write a joke based on the topic"""
        msg = llm.invoke(f"Write a short joke about {state['topic']}")
        return {"joke": msg.content}

    # Build workflow
    builder = StateGraph(State)

    # Add nodes
    builder.add_node("generate_topic", generate_topic)
    builder.add_node("write_joke", write_joke)

    # Add edges to connect nodes
    builder.add_edge(START, "generate_topic")
    builder.add_edge("generate_topic", "write_joke")

    # Compile
    graph = builder.compile()
    ```

### 1. Run the graph

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)

    # Using the graph deployed with the name "agent"
    assistant_id = "agent"

    # create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Run the graph
    result = await client.runs.wait(
        thread_id,
        assistant_id,
        input={}
    )
    ```

=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

    // Using the graph deployed with the name "agent"
    const assistantID = "agent";

    // create a thread
    const thread = await client.threads.create();
    const threadID = thread["thread_id"];

    // Run the graph
    const result = await client.runs.wait(
      threadID,
      assistantID,
      { input: {}}
    );
    ```

=== "cURL"

    Create a thread:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads \
    --header 'Content-Type: application/json' \
    --data '{}'
    ```

    Run the graph:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {}
    }"
    ```

### 2. Identify a checkpoint

=== "Python"

    ```python
    # The states are returned in reverse chronological order.
    states = await client.threads.get_history(thread_id)
    selected_state = states[1]
    print(selected_state)
    ```

=== "JavaScript"

    ```js
    // The states are returned in reverse chronological order.
    const states = await client.threads.getHistory(threadID);
    const selectedState = states[1];
    console.log(selectedState);
    ```

=== "cURL"

    ```bash
    curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/history \
    --header 'Content-Type: application/json'
    ```

### 3. Update the state (optional)

`update_state` will create a new checkpoint. The new checkpoint will be associated with the same thread, but a new checkpoint ID.

=== "Python"

    ```python
    new_config = await client.threads.update_state(
        thread_id,
        {"topic": "chickens"},
        # highlight-next-line
        checkpoint_id=selected_state["checkpoint_id"]
    )
    print(new_config)
    ```

=== "JavaScript"

    ```js
    const newConfig = await client.threads.updateState(
      threadID,
      {
        values: { "topic": "chickens" },
        checkpointId: selectedState["checkpoint_id"]
      }
    );
    console.log(newConfig);
    ```

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"checkpoint_id\": <CHECKPOINT_ID>,
      \"values\": {\"topic\": \"chickens\"}
    }"
    ```

### 4. Resume execution from the checkpoint

=== "Python"

    ```python
    await client.runs.wait(
        thread_id,
        assistant_id,
        # highlight-next-line
        input=None,
        # highlight-next-line
        checkpoint_id=new_config["checkpoint_id"]
    )
    ```

=== "JavaScript"

    ```js
    await client.runs.wait(
      threadID,
      assistantID,
      {
        // highlight-next-line
        input: null,
        // highlight-next-line
        checkpointId: newConfig["checkpoint_id"]
      }
    );
    ```

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"checkpoint_id\": <CHECKPOINT_ID>
    }"
    ```

## Learn more

- [**LangGraph time travel guide**](../../how-tos/human_in_the_loop/time-travel.md): learn more about using time travel in LangGraph.