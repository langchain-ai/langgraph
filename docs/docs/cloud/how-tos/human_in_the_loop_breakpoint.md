# Set breakpoints using Server API

[Breakpoints](../../concepts/breakpoints.md) pause graph execution at defined points and let you step through each stage. They use LangGraph's [**persistence layer**](../../concepts/persistence.md), which saves the graph state after each step.

With breakpoints, you can inspect the graph's state and node inputs at any point. Execution pauses indefinitely until you resume, as the checkpointer preserves the state.

!!! tip

    For conceptual information on breakpoints, see [Breakpoints](../../concepts/breakpoints.md).

## Set static breakpoints

Static breakpoints are triggered either before or after a node executes. You can set static breakpoints by specifying `interrupt_before` and `interrupt_after` at compile time or run time.

=== "Compile time"

    ```python
    # highlight-next-line
    graph = graph_builder.compile( # (1)!
        # highlight-next-line
        interrupt_before=["node_a"], # (2)!
        # highlight-next-line
        interrupt_after=["node_b", "node_c"], # (3)!
    )
    ```

    1. The breakpoints are set during `compile` time.
    2. `interrupt_before` specifies the nodes where execution should pause before the node is executed.
    3. `interrupt_after` specifies the nodes where execution should pause after the node is executed.

=== "Run time"

    === "Python"

        ```python
        # highlight-next-line
        await client.runs.wait( # (1)!
            thread_id,
            assistant_id,
            inputs=inputs,
            # highlight-next-line
            interrupt_before=["node_a"], # (2)!
            # highlight-next-line
            interrupt_after=["node_b", "node_c"] # (3)!
        )
        ```

        1. `client.runs.wait` is called with the `interrupt_before` and `interrupt_after` parameters. This is a run-time configuration and can be changed for every invocation.
        2. `interrupt_before` specifies the nodes where execution should pause before the node is executed.
        3. `interrupt_after` specifies the nodes where execution should pause after the node is executed.

    === "JavaScript"

        ```js
        // highlight-next-line
        await client.runs.wait( // (1)!
          threadID,
          assistantID,
          {
            input: input,
            // highlight-next-line
            interruptBefore: ["node_a"], // (2)!
            // highlight-next-line
            interruptAfter: ["node_b", "node_c"] // (3)!
          }
        )
        ```

        1. `client.runs.wait` is called with the `interruptBefore` and `interruptAfter` parameters. This is a run-time configuration and can be changed for every invocation.
        2. `interruptBefore` specifies the nodes where execution should pause before the node is executed.
        3. `interruptAfter` specifies the nodes where execution should pause after the node is executed.

    === "cURL"

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"interrupt_before\": [\"node_a\"],
          \"interrupt_after\": [\"node_b\", \"node_c\"],
          \"input\": <INPUT>
        }"
        ```

## Example

This example shows how to add **static** breakpoints. See [Use breakpoints](../../how-tos/human_in_the_loop/breakpoints.md) for more options on adding breakpoints.

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)

    # Using the graph deployed with the name "agent"
    assistant_id = "agent"

    # create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Run the graph until the breakpoint
    result = await client.runs.wait(
        thread_id,
        assistant_id,
        input=inputs   # (1)!
    )

    # Resume the graph
    await client.runs.wait(
        thread_id,
        assistant_id,
        input=None   # (2)!
    )
    ```

    1. The graph is run until the first breakpoint is hit.
    2. The graph is resumed by passing in `None` for the input. This will run the graph until the next breakpoint is hit.

=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

    // Using the graph deployed with the name "agent"
    const assistantID = "agent";

    // create a thread
    const thread = await client.threads.create();
    const threadID = thread["thread_id"];

    // Run the graph until the breakpoint
    const result = await client.runs.wait(
      threadID,
      assistantID,
      { input: input }   // (1)!
    );

    // Resume the graph
    await client.runs.wait(
      threadID,
      assistantID,
      { input: null }   // (2)!
    );
    ```

    1. The graph is run until the first breakpoint is hit.
    2. The graph is resumed by passing in `null` for the input. This will run the graph until the next breakpoint is hit.

=== "cURL"

    Create a thread:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads \
    --header 'Content-Type: application/json' \
    --data '{}'
    ```

    Run the graph until the breakpoint:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": <INPUT>
    }"
    ```

    Resume the graph:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\"
    }"
    ```