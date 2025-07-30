# Add node to dataset

This guide shows how to add examples to [LangSmith datasets](https://docs.smith.langchain.com/evaluation/how_to_guides#dataset-management) from nodes in the thread log. This is useful to evaluate individual steps of the agent.

1. Select a thread.
2. Click on the `Add to Dataset` button.
3. Select nodes whose input/output you want to add to a dataset.
4. For each selected node, select the target dataset to create the example in. By default a dataset for the specific assistant and node will be selected. If this dataset does not yet exist, it will be created.
5. Edit the example's input/output as needed before adding it to the dataset.
6. Select "Add to dataset" at the bottom of the page to add all selected nodes to their respective datasets.

See [Evaluating intermediate steps](https://docs.smith.langchain.com/evaluation/how_to_guides/langgraph#evaluating-intermediate-steps) for more details on how to evaluate intermediate steps.
