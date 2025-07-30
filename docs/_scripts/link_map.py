"""Link mapping for cross-reference resolution across different scopes.

This module provides link mappings for different language/framework scopes
to resolve @[link_name] references to actual URLs.
"""

# Python-specific link mappings
# Python-specific link mappings
PYTHON_LINK_MAP = {
    "StateGraph": "reference/graphs/#langgraph.graph.StateGraph",
    "add_conditional_edges": "reference/graphs/#langgraph.graph.state.StateGraph.add_conditional_edges",
    "add_edge": "reference/graphs/#langgraph.graph.state.StateGraph.add_edge",
    "add_node": "reference/graphs/#langgraph.graph.state.StateGraph.add_node",
    "add_messages": "reference/graphs/#langgraph.graph.message.add_messages",
    "ToolNode": "reference/agents/#langgraph.prebuilt.tool_node.ToolNode",
    "CompiledStateGraph.astream": "reference/graphs/#langgraph.graph.state.CompiledStateGraph.astream",
    "Pregel.astream": "reference/pregel/#langgraph.pregel.Pregel.astream",
    "AsyncPostgresSaver": "reference/checkpoints/#langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
    "AsyncSqliteSaver": "reference/checkpoints/#langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver",
    "BaseCheckpointSaver": "reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver",
    "BaseStore": "reference/store/#langgraph.store.base.BaseStore",
    "BaseStore.put": "reference/store/#langgraph.store.base.BaseStore.put",
    "BinaryOperatorAggregate": "reference/pregel/#langgraph.pregel.Pregel--advanced-channels-context-and-binaryoperatoraggregate",
    "CipherProtocol": "reference/checkpoints/#langgraph.checkpoint.serde.base.CipherProtocol",
    "client.runs.stream": "cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.RunsClient.stream",
    "client.runs.wait": "cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.RunsClient.wait",
    "client.threads.get_history": "cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.ThreadsClient.get_history",
    "client.threads.update_state": "cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.ThreadsClient.update_state",
    "Command": "reference/types/#langgraph.types.Command",
    "CompiledStateGraph": "reference/graphs/#langgraph.graph.state.CompiledStateGraph",
    "create_react_agent": "reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent",
    "create_supervisor": "reference/supervisor/#langgraph_supervisor.supervisor.create_supervisor",
    "EncryptedSerializer": "reference/checkpoints/#langgraph.checkpoint.serde.encrypted.EncryptedSerializer",
    "entrypoint.final": "reference/func/#langgraph.func.entrypoint.final",
    "entrypoint": "reference/func/#langgraph.func.entrypoint",
    "from_pycryptodome_aes": "reference/checkpoints/#langgraph.checkpoint.serde.encrypted.EncryptedSerializer.from_pycryptodome_aes",
    # "getContextVariable": "<insert-ref>",
    "get_state_history": "reference/graphs/#langgraph.graph.state.CompiledStateGraph.get_state_history",
    "get_stream_writer": "reference/config/#langgraph.config.get_stream_writer",
    "HumanInterrupt": "reference/prebuilt/#langgraph.prebuilt.interrupt.HumanInterrupt",
    "InjectedState": "reference/agents/#langgraph.prebuilt.tool_node.InjectedState",
    "InMemorySaver": "reference/checkpoints/#langgraph.checkpoint.memory.InMemorySaver",
    "interrupt": "reference/types/#langgraph.types.Interrupt",
    "CompiledStateGraph.invoke": "reference/graphs/#langgraph.graph.state.CompiledStateGraph.invoke",
    "JsonPlusSerializer": "reference/checkpoints/#langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer",
    "langgraph.json": "cloud/reference/cli/#configuration-file",
    "LastValue": "reference/channels/#langgraph.channels.LastValue",
    # "MemorySaver": "<insert-ref>",
    # "messagesStateReducer": "<insert-ref>",
    "PostgresSaver": "reference/checkpoints/#langgraph.checkpoint.postgres.PostgresSaver",
    "Pregel": "reference/pregel/",
    "Pregel.stream": "reference/pregel/#langgraph.pregel.Pregel.stream",
    "pre_model_hook": "reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent",
    "protocol": "reference/checkpoints/#langgraph.checkpoint.serde.base.SerializerProtocol",
    "Send": "reference/types/#langgraph.types.Send",
    "SerializerProtocol": "reference/checkpoints/#langgraph.checkpoint.serde.base.SerializerProtocol",
    "SqliteSaver": "reference/checkpoints/#langgraph.checkpoint.sqlite.SqliteSaver",
    "START": "reference/constants/#langgraph.constants.START",
    "CompiledStateGraph.stream": "reference/graphs/#langgraph.graph.state.CompiledStateGraph.stream",
    "task": "reference/func/#langgraph.func.task",
    "Topic": "reference/channels/#langgraph.channels.Topic",
    "update_state": "reference/graphs/#langgraph.graph.state.CompiledStateGraph.update_state",
}


# JavaScript-specific link mappings
JS_LINK_MAP = {
    "Auth": "reference/classes/sdk_auth.Auth.html",
    "StateGraph": "reference/classes/langgraph.StateGraph.html",
    "add_conditional_edges": "reference/functions/langgraph_StateGraph.addConditionalEdges.html",
    "add_edge": "reference/functions/langgraph_StateGraph.addEdge.html",
    "add_node": "reference/functions/langgraph_StateGraph.addNode.html",
    "add_messages": "reference/functions/langgraph_message.addMessages.html",
    "ToolNode": "reference/classes/langgraph_prebuilt.ToolNode.html",
    "CompiledStateGraph.astream()": "reference/functions/langgraph_CompiledStateGraph.astream.html",
    "Pregel.astream": "reference/functions/langgraph_Pregel.astream.html",
    "AsyncPostgresSaver": "reference/classes/langgraph_checkpoint_postgres_aio.AsyncPostgresSaver.html",
    "AsyncSqliteSaver": "reference/classes/langgraph_checkpoint_sqlite_aio.AsyncSqliteSaver.html",
    "BaseCheckpointSaver": "reference/classes/langgraph_checkpoint_base.BaseCheckpointSaver.html",
    "BaseStore": "reference/classes/langgraph_store_base.BaseStore.html",
    "BaseStore.put": "reference/functions/langgraph_store_base.BaseStore.put.html",
    "BinaryOperatorAggregate": "reference/classes/langgraph_channels.BinaryOperatorAggregate.html",
    "CipherProtocol": "reference/classes/langgraph_checkpoint_serde_base.CipherProtocol.html",
    "client.runs.stream": "reference/functions/langgraph_sdk_client.RunsClient.stream.html",
    "client.runs.wait": "reference/functions/langgraph_sdk_client.RunsClient.wait.html",
    "client.threads.get_history": "reference/functions/langgraph_sdk_client.ThreadsClient.getHistory.html",
    "client.threads.update_state": "reference/functions/langgraph_sdk_client.ThreadsClient.updateState.html",
    "Command": "reference/classes/langgraph.Command.html",
    "CompiledStateGraph": "reference/classes/langgraph.CompiledStateGraph.html",
    "create_react_agent": "reference/functions/langgraph_prebuilt.createReactAgent.html",
    "create_supervisor": "reference/functions/langgraph_supervisor.createSupervisor.html",
    "EncryptedSerializer": "reference/classes/langgraph_checkpoint_serde_encrypted.EncryptedSerializer.html",
    "entrypoint.final": "reference/functions/langgraph_func.entrypoint.final.html",
    "entrypoint": "reference/functions/langgraph_func.entrypoint.html",
    "from_pycryptodome_aes": "reference/functions/langgraph_checkpoint_serde_encrypted.EncryptedSerializer.fromPycryptodomeAes.html",
    "getContextVariable": "https://v03.api.js.langchain.com/functions/_langchain_core.context.getContextVariable.html",
    "get_state_history": "reference/functions/langgraph_CompiledStateGraph.getStateHistory.html",
    "get_stream_writer": "reference/functions/langgraph_config.getStreamWriter.html",
    "HumanInterrupt": "reference/classes/langgraph_prebuilt.HumanInterrupt.html",
    "InjectedState": "reference/classes/langgraph_prebuilt.InjectedState.html",
    "InMemorySaver": "reference/classes/langgraph_checkpoint_memory.InMemorySaver.html",
    "interrupt": "reference/functions/langgraph.interrupt-2.html",
    "CompiledStateGraph.invoke": "reference/functions/langgraph_CompiledStateGraph.invoke.html",
    "JsonPlusSerializer": "reference/classes/langgraph_checkpoint_serde_jsonplus.JsonPlusSerializer.html",
    "langgraph.json": "reference/configuration.html",
    "LastValue": "reference/classes/langgraph_channels.LastValue.html",
    "MemorySaver": "reference/classes/checkpoint.MemorySaver.html",
    "messagesStateReducer": "reference/functions/langgraph.messagesStateReducer.html",
    "PostgresSaver": "reference/classes/langgraph_checkpoint_postgres.PostgresSaver.html",
    "Pregel": "reference/classes/langgraph.Pregel.html",
    "Pregel.stream": "reference/functions/langgraph_Pregel.stream.html",
    "pre_model_hook": "reference/functions/langgraph_prebuilt.createReactAgent.html",
    "protocol": "reference/classes/langgraph_checkpoint_serde_base.SerializerProtocol.html",
    "Send": "reference/classes/langgraph.Send.html",
    "SerializerProtocol": "reference/classes/langgraph_checkpoint_serde_base.SerializerProtocol.html",
    "SqliteSaver": "reference/classes/langgraph_checkpoint_sqlite.SqliteSaver.html",
    "START": "reference/constants.html#START",
    "CompiledStateGraph.stream": "reference/functions/langgraph_CompiledStateGraph.stream.html",
    "task": "reference/functions/langgraph_func.task.html",
    "Topic": "reference/classes/langgraph_channels.Topic.html",
    "update_state": "reference/functions/langgraph_CompiledStateGraph.updateState.html",
}

# TODO: Allow updating these to localhost for local development
PY_REFERENCE_HOST = "https://langchain-ai.github.io/langgraph/"
JS_REFERENCE_HOST = "https://langchain-ai.github.io/langgraphjs/"

for key, value in PYTHON_LINK_MAP.items():
    # Ensure the link is absolute
    if not value.startswith("http"):
        PYTHON_LINK_MAP[key] = f"{PY_REFERENCE_HOST}{value}"

for key, value in JS_LINK_MAP.items():
    # Ensure the link is absolute
    if not value.startswith("http"):
        JS_LINK_MAP[key] = f"{JS_REFERENCE_HOST}{value}"

# Global scope is assembled from the Python and JS mappings
# Combined mapping by scope
SCOPE_LINK_MAPS = {
    "python": PYTHON_LINK_MAP,
    "js": JS_LINK_MAP,
}
