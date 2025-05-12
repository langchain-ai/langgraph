# INVALID_CHAT_HISTORY

This error is raised in the prebuilt [create_react_agent][langgraph.prebuilt.chat_agent_executor.create_react_agent] when the `call_model` graph node receives a malformed list of messages. Specifically, it is malformed when there are `AIMessages` with `tool_calls` (LLM requesting to call a tool) that do not have a corresponding `ToolMessage` (result of a tool invocation to return to the LLM).

There could be a few reasons you're seeing this error:

1. You manually passed a malformed list of messages when invoking the graph, e.g. `graph.invoke({'messages': [AIMessage(..., tool_calls=[...])]})`
2. The graph was interrupted before receiving updates from the `tools` node (i.e. a list of ToolMessages)
and you invoked it with a an input that is not None or a ToolMessage,
e.g. `graph.invoke({'messages': [HumanMessage(...)]}, config)`.
    This interrupt could have been triggered in one of the following ways:
     - You manually set `interrupt_before = ['tools']` in `create_react_agent`
     - One of the tools raised an error that wasn't handled by the [ToolNode][langgraph.prebuilt.tool_node.ToolNode] (`"tools"`)

## Troubleshooting

To resolve this, you can do one of the following:

1. Don't invoke the graph with a malformed list of messages
2. In case of an interrupt (manual or due to an error) you can:

    - provide ToolMessages that match existing tool calls and call `graph.invoke({'messages': [ToolMessage(...)]})`.
    **NOTE**: this will append the messages to the history and run the graph from the START node.
    - manually update the state and resume the graph from the interrupt:

        1. get the list of most recent messages from the graph state with `graph.get_state(config)`
        2. modify the list of messages to either remove unanswered tool calls from AIMessages
or add ToolMessages with tool_call_ids that match unanswered tool calls
        3. call `graph.update_state(config, {'messages': ...})` with the modified list of messages
        4. resume the graph, e.g. call `graph.invoke(None, config)`
