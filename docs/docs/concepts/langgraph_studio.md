# LangGraph Studio

!!! info "Prerequisites"

    - [LangGraph Platform](./langgraph_platform.md)
    - [LangGraph Server](./langgraph_server.md)

LangGraph Studio offers a new way to develop LLM applications by providing a specialized agent IDE that enables visualization, interaction, and debugging of complex agentic applications.

With visual graphs and the ability to edit state, you can better understand agent workflows and iterate faster. LangGraph Studio integrates with LangSmith allowing you to collaborate with teammates to debug failure modes.

![](img/lg_studio.png)

## Features

The key features of LangGraph Studio are:

- Visualize your graphs
- Test your graph by running it from the UI
- Debug your agent by [modifying its state and rerunning](human_in_the_loop.md)
- Create and manage [assistants](assistants.md)
- View and manage [threads](persistence.md#threads)
- View and manage [long term memory](memory.md)
- Add node input/outputs to [LangSmith](https://smith.langchain.com/) datasets for testing

## Getting started

There are two ways to connect your LangGraph app with the studio:

### Deployed Application

If you have deployed your LangGraph application on LangGraph Platform, you can access the studio as part of that deployment. To do so, navigate to the deployment in LangGraph Platform within the LangSmith UI and click the "LangGraph Studio" button.

### Local Development Server

If you have a LangGraph application that is [running locally in-memory](../tutorials/langgraph-platform/local-server.md), you can connect it to LangGraph Studio in the browser within LangSmith.

By default, starting the local server with `langgraph dev` will run the server at `http://127.0.0.1:2024` and automatically open Studio in your browser. However, you can also manually connect to Studio by either:

1. In LangGraph Platform, clicking the "LangGraph Studio" button and entering the server URL in the dialog that appears.

   or

2. Navigating to the URL in your browser:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

## Related

For more information please see the following:

- [LangGraph Studio how-to guides](../how-tos/index.md#langgraph-studio)
- [LangGraph CLI Documentation](../cloud/reference/cli.md)

## LangGraph Studio FAQs

### Why is my project failing to start?

A project may fail to start if the configuration file is defined incorrectly, or if required environment variables are missing. See [here](../cloud/reference/cli.md#configuration-file) for how your configuration file should be defined.

### How does interrupt work?

When you select the `Interrupts` dropdown and select a node to interrupt the graph will pause execution before and after (unless the node goes straight to `END`) that node has run. This means that you will be able to both edit the state before the node is ran and the state after the node has ran. This is intended to allow developers more fine-grained control over the behavior of a node and make it easier to observe how the node is behaving. You will not be able to edit the state after the node has ran if the node is the final node in the graph.

For more information on interrupts and human in the loop, see [here](./human_in_the_loop.md).

### Why are extra edges showing up in my graph?

If you don't define your conditional edges carefully, you might notice extra edges appearing in your graph. This is because without proper definition, LangGraph Studio assumes the conditional edge could access all other nodes. In order for this to not be the case, you need to be explicit about how you define the nodes the conditional edge routes to. There are two ways you can do this:

#### Solution 1: Include a path map

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

#### Solution 2: Update the typing of the router (Python only)

Instead of passing a path map, you can also be explicit about the typing of your routing function by specifying the nodes it can map to using the `Literal` python definition. Here is an example of how to define a routing function in that way:

```python
def routing_function(state: GraphState) -> Literal["node_b","node_c"]:
    if state['some_condition'] == True:
        return "node_b"
    else:
        return "node_c"
```

### Studio Desktop FAQs

!!! warning "Deprecation Warning"
    In order to support a wider range of platforms and users, we now recommend following the above instructions to connect to LangGraph Studio using the development server instead of the desktop app.

The LangGraph Studio Desktop App is a standalone application that allows you to connect to your LangGraph application and visualize and interact with your graph. It is available for MacOS only and requires Docker to be installed.

#### Why is my project failing to start?

In addition to the reasons listed above, for the desktop app there are a few more reasons that your project might fail to start:

!!! Important "Note "

    LangGraph Studio Desktop automatically populates `LANGCHAIN_*` environment variables for license verification and tracing, regardless of the contents of the `.env` file. All other environment variables defined in `.env` will be read as normal.

##### Docker issues

LangGraph Studio (desktop) requires Docker Desktop version 4.24 or higher. Please make sure you have a version of Docker installed that satisfies that requirement and also make sure you have the Docker Desktop app up and running before trying to use LangGraph Studio. In addition, make sure you have docker-compose updated to version 2.22.0 or higher.

##### Incorrect data region

If you receive a license verification error when attempting to start the LangGraph Server, you may be logged into the incorrect LangSmith data region. Ensure that you're logged into the correct LangSmith data region and ensure that the LangSmith account has access to LangGraph platform.

1. In the top right-hand corner, click the user icon and select `Logout`.
1. At the login screen, click the `Data Region` dropdown menu and select the appropriate data region. Then click `Login to LangSmith`.

### How do I reload the app?

If you would like to reload the app, don't use Command+R as you might normally do. Instead, close and reopen the app for a full refresh.

### How does automatic rebuilding work?

One of the key features of LangGraph Studio is that it automatically rebuilds your image when you change the source code. This allows for a super fast development and testing cycle which makes it easy to iterate on your graph. There are two different ways that LangGraph rebuilds your image: either by editing the image or completely rebuilding it.

#### Rebuilds from source code changes

If you modified the source code only (no configuration or dependency changes!) then the image does not require a full rebuild, and LangGraph Studio will only update the relevant parts. The UI status in the bottom left will switch from `Online` to `Stopping` temporarily while the image gets edited. The logs will be shown as this process is happening, and after the image has been edited the status will change back to `Online` and you will be able to run your graph with the modified code!

#### Rebuilds from configuration or dependency changes

If you edit your graph configuration file (`langgraph.json`) or the dependencies (either `pyproject.toml` or `requirements.txt`) then the entire image will be rebuilt. This will cause the UI to switch away from the graph view and start showing the logs of the new image building process. This can take a minute or two, and once it is done your updated image will be ready to use!

### Why is my graph taking so long to startup?

The LangGraph Studio interacts with a local LangGraph API server. To stay aligned with ongoing updates, the LangGraph API requires regular rebuilding. As a result, you may occasionally experience slight delays when starting up your project.
