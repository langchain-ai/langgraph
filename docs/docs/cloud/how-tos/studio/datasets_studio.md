# Adding nodes as dataset examples in Studio

In LangGraph Studio you can create dataset examples from the thread history in the right-hand pane. This can be especially useful when you want to evaluate intermediate steps of the agent.

1. Click on the `Add to Dataset` button to enter the dataset mode.
1. Select nodes which you want to add to dataset.
1. Select the target dataset to create the example in.

You can edit the example payload before sending it to the dataset, which is useful if you need to make changes to conform the example to the dataset schema.

Finally, you can customise the target dataset by clicking on the `Settings` button.

See [Evaluating intermediate steps](https://docs.smith.langchain.com/evaluation/how_to_guides/langgraph#evaluating-intermediate-steps) for more details on how to evaluate intermediate steps.

<video controls allowfullscreen="true" poster="../img/studio_datasets.jpg">
    <source src="https://langgraph-docs-assets.pages.dev/studio_datasets.mp4" type="video/mp4">
</video>
