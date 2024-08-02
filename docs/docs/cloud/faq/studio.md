# Studio FAQs

## Why is my project failing to start?

There are a few reasons that your project might fail to start, here are some of the most common ones.

### Docker issues

LangGraph Studio requires Docker Desktop version 4.24 or higher. Please make sure you have a version of Docker installed that satisfies that requirement and also make sure you have the Docker Desktop app up and running before trying to use LangGraph Studio. In addition, make sure you have docker-compose updated to version 2.22.0 or higher.

### Configuration or environment issues

Another reason your project might fail to start is because your configuration file is defined incorrectly, or you are missing required environment variables. 

## How does interrupt work?

When you select the `Interrupts` dropdown and select a node to interrupt the graph will pause execution before and after (unless the node goes straight to `END`) that node has run. This means that you will be able to both edit the input to the node and the output of the node. This is intended to allow developers more fine-grained control over the behavior of a node and make it easier to observe how the node is behaving. You will not be able to edit the output if the node is the final node in the graph.

## How do I reload the app?

If you would like to reload the app, don't use Command+R as you might normally do. Instead, close and reopen the app for a full refresh.

## How does automatic rebuilding work?

One of the best features of LangGraph Studio is that it automatically rebuilds your image when you change the source code. This allows for a super fast development and testing cycle which makes it easy to iterate on your graph. There are two different ways that LangGraph rebuilds your image: either by editing the image or completely rebuilding it.

### Rebuilds from source code changes

If you modified the source code only (no configuration or dependency changes!) then the image does not require a full rebuild, and LangGraph Studio will only update the relevant parts. The UI status in the bottom left will switch from `Online` to `Stopping` temporarily while the image gets edited. The logs will be shown as this process is happening, and after the image has been edited the status will change back to `Online` and you will be able to run your graph with the modified code!


### Rebuilds from configuration or dependency changes

If you edit your graph configuration file (`langgraph.json`) or the dependencies (either `pyproject.toml` or `requirements.txt`) then the entire image will be rebuilt. This will cause the UI to switch away from the graph view and start showing the logs of the new image building process. This can take a minute or two, and once it is done your updated image will be ready to use!

## Why is my graph taking so long to startup?

The LangGraph studio works by making API calls to the LangGraph API, and this API must be rebuilt regularly to keep up with the changes made to it. For this reason, you might notice that occasionally your graphs take a little longer to startup. 

## Why are extra edges showing up in my graph?

If you don't define your conditional edges carefully, you might notice extra edges appearing in your graph. This is because without proper definition, LangGraph Studio assumes the conditional edge could access all other nodes. In order for this to not be the case, you need to be explicit about how you define conditional edges. There are two ways you can do this:

### Solution 1: Include a path map

The first way to solve this is to add path maps to your conditional edges. A path map is just a dictionary that maps the possible outputs of your router function with the names of the nodes that each output corresponds to. The path map is passed as the third argument to the `add_conditional_edges` function like so:

```python
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
```

In this case, the routing function returns either True or False, which map to `node_b` and `node_c` respectively.

### Solution 2: Update the typing of the router

Instead of passing a path map, you can also be explicit about the typing of your routing function by specifying the nodes it can map to using the `Literal` python definition. Here is an example of how to define a routing function in that way:

```python
def routing_function(state: GraphState) -> Literal["node_b","node_c"]:
    if state['some_condition'] == True:
        return "node_a"
    else:
        return "node_b"
```

