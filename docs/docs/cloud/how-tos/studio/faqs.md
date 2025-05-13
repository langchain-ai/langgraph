# LangGraph Studio FAQs

## Why is my project failing to start?

A project may fail to start if the configuration file is defined incorrectly, or if required environment variables are missing. See [here](../../reference/cli.md#configuration-file) for how your configuration file should be defined.

## How does interrupt work?

When you select the `Interrupts` dropdown and select a node to interrupt the graph will pause execution before and after (unless the node goes straight to `END`) that node has run. This means that you will be able to both edit the state before the node is ran and the state after the node has ran. This is intended to allow developers more fine-grained control over the behavior of a node and make it easier to observe how the node is behaving. You will not be able to edit the state after the node has ran if the node is the final node in the graph.

For more information on interrupts and human in the loop, see [here](../../../concepts/human_in_the_loop.md).

## Why are extra edges showing up in my graph?

If you don't define your conditional edges carefully, you might notice extra edges appearing in your graph. This is because without proper definition, LangGraph Studio assumes the conditional edge could access all other nodes. In order for this to not be the case, you need to be explicit about how you define the nodes the conditional edge routes to. There are two ways you can do this:

### Solution 1: Include a path map

The first way to solve this is to add path maps to your conditional edges. A path map is just a dictionary or array that maps the possible outputs of your router function with the names of the nodes that each output corresponds to. The path map is passed as the third argument to the `add_conditional_edges` function like so:

=== "Python"

    ```python
    graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
    ```

=== "Javascript"

    ```ts
    graph.addConditionalEdges("node_a", routingFunction, { true: "node_b", false: "node_c" });
    ```

In this case, the routing function returns either True or False, which map to `node_b` and `node_c` respectively.

### Solution 2: Update the typing of the router (Python only)

Instead of passing a path map, you can also be explicit about the typing of your routing function by specifying the nodes it can map to using the `Literal` python definition. Here is an example of how to define a routing function in that way:

```python
def routing_function(state: GraphState) -> Literal["node_b","node_c"]:
    if state['some_condition'] == True:
        return "node_b"
    else:
        return "node_c"
```


## Why is my graph taking so long to startup?

The LangGraph Studio interacts with a local LangGraph API server. To stay aligned with ongoing updates, the LangGraph API requires regular rebuilding. As a result, you may occasionally experience slight delays when starting up your project.

## Can I use Studio without using LangSmith?
By default, LangGraph Studio is accessed from the LangSmith UI, within the LangGraph Platform Deployments tab. For local development, if you do not wish to have data traced to Langsmith, simply set `LANGSMITH_TRACING=false` in your application's `.env` file. With tracing disabled, no data will leave your local server.