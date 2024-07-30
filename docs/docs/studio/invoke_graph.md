# How to invoke graph

LangGraph Studio lets you run your graph with different inputs and configurations.

## Start a new run

To start a new run:

1. In the dropdown menu (top-left corner of the left-hand pane), select a graph. In our example the graph is called `agent`. The list of graphs corresponds to the `graphs` keys in your `langgraph.json` configuration.
1. In the bottom of the left-hand pane, edit the `Input` section.
1. Click `Submit` to invoke the selected graph.
1. View output of the invocation in the right-hand pane.

The following video shows how to start a new run:

<video controls allowfullscreen="true" poster="../img/graph_video_poster.png">
    <source src="../img/graph_invoke.mp4" type="video/mp4">
</video>

## Configure graph run

To change configuration for a given graph run, press `Configurable` button in the `Input` section. Then click `Submit` to invoke the graph.

!!! note Note
    In order for the `Configurable` menu to be visible, make sure to specify `config_schema` when creating `StateGraph`. You can read more about how to add config schema to your graph [here](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/configuration_cloud/).
   
The following video shows how to edit configuration and start a new run:

<video controls allowfullscreen="true" poster="../img/graph_video_poster.png">
    <source src="../img/graph_config.mp4" type="video/mp4">
</video>