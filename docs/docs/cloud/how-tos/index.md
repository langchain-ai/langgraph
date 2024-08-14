---
hide:
  - toc
---

# How-to Guides

Welcome to the LangGraph Cloud how-to guides! These guides provide practical, step-by-step instructions for accomplishing key tasks in LangGraph Cloud.

## Deployment

LangGraph Cloud gives you best in class observability, testing, and hosting services. Read more about them in these how to guides:

- [How to set up app for deployment (requirements.txt)](../deployment/setup.md)
- [How to set up app for deployment (pyproject.toml)](../deployment/setup_pyproject.md)
- [How to test locally](../deployment/test_locally.md)
- [How to deploy to LangGraph cloud](../deployment/cloud.md)
- [How to self-host](../deployment/self_hosted.md)


## Streaming

Streaming the results of your LLM application is vital for ensuring a good user experience, especially when your graph may call multiple models and take a long time to fully complete a run. Read about how to stream values from your graph in these how to guides:

- [How to stream values](./stream_values.md)
- [How to stream updates](./stream_updates.md)
- [How to stream messages](./stream_messages.md)
- [How to stream events](./stream_events.md)
- [How to stream in debug mode](./stream_debug.md)
- [How to stream multiple modes](./stream_multiple.md)

## Double-texting

Graph execution can take a while, and sometimes users may change their mind about the input they wanted to send before their original input has finished running. For example, a user might notice a typo in their original request and will edit the prompt and resend it. Deciding what to do in these cases is important for ensuring a smooth user experience and preventing your graphs from behaving in unexpected ways. The following how-to guides provide information on the various options LangGraph Cloud gives you for dealing with double-texting:

- [How to use the interrupt option](./interrupt_concurrent.md)
- [How to use the rollback option](./rollback_concurrent.md)
- [How to use the reject option](./reject_concurrent.md)
- [How to use the enqueue option](./enqueue_concurrent.md)

## Human-in-the-loop

When creating complex graphs, leaving every decision up to the LLM can be dangerous, especially when the decisions involve invoking certain tools or accessing specific documents. To remedy this, LangGraph allows you to insert human-in-the-loop behavior to ensure your graph does not have undesired outcomes. Read more about the different ways you can add human-in-the-loop capabilities to your LangGraph Cloud projects in these how-to guides:

- [How to add a breakpoint](./human_in_the_loop_breakpoint.md)
- [How to wait for user input](./human_in_the_loop_user_input.md)
- [How to edit graph state](./human_in_the_loop_edit_state.md)
- [How to replay and branch from prior states](./human_in_the_loop_time_travel.md)
- [How to review tool calls](./human_in_the_loop_review_tool_calls.md)

## LangGraph Studio

LangGraph Studio is a built-in UI for visualizing, testing, and debugging your agents.

- [How to enter LangGraph Studio](./test_deployment.md)
- [How to enter LangGraph Studio for local deployment](./test_local_deployment.md)
- [How to test your graph in LangGraph Studio](./invoke_studio.md)
- [Interact with threads in LangGraph Studio](./threads_studio.md)

## Different Types of Runs:

LangGraph Cloud supports multiple types of runs besides streaming runs.

- [How to run an agent in the background](cloud_examples/background_run.ipynb)
- [How to run multiple agents in the same thread](cloud_examples/same-thread.ipynb)
- [How to create cron jobs](cloud_examples/cron_jobs.ipynb)
- [How to create stateless runs](cloud_examples/stateless_runs.ipynb)

## Other 

Other guides that may prove helpful!

- [How to configure agents](cloud_examples/configuration_cloud.ipynb)
- [How to convert LangGraph calls to LangGraph cloud calls](cloud_examples/langgraph_to_langgraph_cloud.ipynb)
- [How to integrate webhooks](cloud_examples/webhooks.ipynb)
- [How to copy threads](./copy_threads.md)