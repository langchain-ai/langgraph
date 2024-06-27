---
hide:
  - toc
---

# How-to Guides

Welcome to the LangGraph Cloud how-to guides! These guides provide practical, step-by-step instructions for accomplishing key tasks in LangGraph Cloud.

## Deployment

LangGraph Cloud gives you best in class observability, testing, and hosting services. Read more about them in these how to guides:

- [How to set up app for deployment](https://langchain-ai.github.io/langgraph/cloud/deployment/setup/)
- [How to deploy to LangGraph cloud](https://langchain-ai.github.io/langgraph/cloud/deployment/cloud/)
- [How to self-host](https://langchain-ai.github.io/langgraph/cloud/deployment/self_hosted/)

## Streaming

Streaming the results of your LLM application is vital for ensuring a good user experience, especially when your graph may call multiple models and take a long time to fully complete a run. Read about how to stream values from your graph in these how to guides:

- [How to stream values](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/stream_values/)
- [How to stream updates](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/stream_updates/)
- [How to stream messages](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/stream_messages/)
- [How to stream events](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/stream_events/)
- [How to stream in debug mode](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/stream_debug/)
- [How to stream multiple modes](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/stream_multiple/)

## Double-texting

Graph execution can take a while, and sometimes users may change their mind about the input they wanted to send before their original input has finished running. For example, a user might notice a typo in their original request and will edit the prompt and resend it. Deciding what to do in these cases is important for ensuring a smooth user experience and preventing your graphs from behaving in unexpected ways. The following how-to guides provide information on the various options LangGraph Cloud gives you for dealing with double-texting:

- [How to use the interrupt option](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/interrupt_concurrent/)
- [How to use the rollback option](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/rollback_concurrent/)
- [How to use the reject option](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/reject_concurrent/)
- [How to use the rnqueue option](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/enqueue_concurrent/)

## Human-in-the-loop

When creating complex graphs, leaving every decision up to the LLM can be dangerous, especially when the decisions involve invoking certain tools or accessing specific documents. To remedy this, LangGraph allows you to insert human-in-the-loop behavior to ensure your graph does not have undesired outcomes. Read more about the different ways you can add human-in-the-loop capabilities to your LangGraph Cloud projects in these how-to guides:

- [How to add a breakpoint](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/human_in_the_loop_breakpoint/)
- [How to wait for user input](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/human_in_the_loop_user_input/)
- [How to edit graph state](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/human_in_the_loop_edit_state/)
- [How to replay and branch from prior states](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/human_in_the_loop_time_travel/)

## LangGraph Studio

- [Test Cloud Deployment](https://langchain-ai.github.io/langgraph/cloud/how-tos/test_deployment/)
- [Invoke graph in LangGraph Studio](https://langchain-ai.github.io/langgraph/cloud/how-tos/invoke_studio/)
- [Interact with threads in LangGraph Studio](https://langchain-ai.github.io/langgraph/cloud/how-tos/threads_studio/)

## And more!

The four sections above don't cover everything that is possible with LangGraph cloud - make sure to check out our other how-to guides to learn even more!