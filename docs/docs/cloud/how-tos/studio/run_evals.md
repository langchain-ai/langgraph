# Run experiments over a dataset

LangGraph Studio supports evaluations by allowing you to run your assistant over a pre-defined LangSmith dataset. This enables you to understand how your application performs over a variety of inputs, compare the results to reference outputs, and score the results using evaluators.

This guide shows you how to run an experiment in Studio.

!!! info "Note"

    This feature is supported for all applications deployed on LangGraph Platform, or locally running Python applications. Support for the local JS server is coming soon.

1. To start, ensure you have an existing LangSmith dataset. For information on how to create a dataset, see [here](https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application#set-up-your-dataset).

2. You can additionally choose to attach evaluators to the created dataset. Evaluators are functions that score how well your application performs on a given example. They get triggered at the end of the experiment and run against the experiment results. For more information on evaluators, see [here](https://docs.smith.langchain.com/evaluation/concepts#evaluators).

3. Click "Run experiment" in the top right corner of the Studio page.

4. Select the desired dataset (or dataset split) and click start.

5. All of the inputs in the dataset will now be run against the active graph and assistant.

6. Monitor the experiment progress in the top right corner. Click the arrow icon button at any time to view the experiment results.

## Troubleshooting

### Run experiment button disabled

#### Deployed application

If your application is deployed on LangGraph Platform, make a new revision to enable this feature.

#### Local development server

If you are running your application using the local server, make sure you upgrade to the latest version of the `langgraph-cli`. Additionally, you have tracing enabled via the `LANGSMITH_API_KEY` in your project's `.env` file. For local servers, this is not yet supported in JavaScript.

### Evaluator results missing

When you run your experiment, the evaluators get scheduled for execution in a queue. If you don't see evaluator results for the experiment, it likely means that these are still in the queue. Continue to check the experiment for updated results.
