# How to Test with LangGraph Studio

LangGraph applications can be tested with LangGraph Studio. LangGraph Studio is a robust UI for testing and exercising functionality of the graphs in a LangGraph application. The LangGraph Studio UI connects directly to a LangGraph Cloud deployments or to a local LangGraph API instance.

The LangGraph Studio UI is available within <a href="https://www.langchain.com/langsmith" target="_blank">LangSmith</a>. To test a LangGraph application, navigate to the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>.

## Test Cloud Deployment

The LangGraph Studio UI connects directly to LangGraph Cloud deployments.

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `Deployments`. The `Deployments` view contains a list of existing LangGraph Cloud deployments.
1. Select an existing deployment to test with LangGraph Studio.
1. In the top-right corner, select `Open LangGraph Studio`.
1. [Invoke an assistant](#invoke-assistant) or [view an existing thread](#view-thread).

## Test Local Instance

The LangGraph Studio UI connects directly to local LangGraph API instances. This is helpful for quickly iterating and testing during the development process.

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. Find the hostname of the local LangGraph API instance. For example, `http://localhost:9123`.
1. Navigate to the URL `/studio/thread/` and set the `baseUrl` query parameter to the hostname. For example, `https://smith.langchain.com/studio/thread?baseUrl=http://localhost:9123`.
1. [Invoke an assistant](#invoke-assistant) or [view an existing thread](#view-thread).

## Invoke Assistant

1. The LangGraph Studio UI displays a visualization of the selected assistant.
    1. In the top-right dropdown menu of the left-hand pane, select an assistant.
    1. In the bottom of the left-hand pane, edit the `Input` and `Configure` the assistant.
    1. Select `Submit` to invoke the selected assistant.
1. View output of the invocation in the right-hand pane.

## View Thread

1. In the top of the right-hand pane, select the `New Thread` dropdown menu to view existing threads.
1. View the state of the thread (i.e. the output) in the right-hand pane.
1. To create a new thread, select `+ New Thread`.

## Edit Thread State

The LangGraph Studio UI contains features for editing thread state. Explore these features in the right-hand pane. Select the `Edit` icon, modify the desired state, and then select `Fork` to invoke the assistant with the updated state.