# Graph Definitions

::: langgraph.graph.state.StateGraph
    options:
      show_if_no_docstring: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - add_node
        - add_edge
        - add_conditional_edges
        - add_sequence
        - compile

::: langgraph.graph.state.CompiledStateGraph
    options:
      show_if_no_docstring: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - stream
        - astream
        - invoke
        - ainvoke
        - get_state
        - aget_state
        - get_state_history
        - aget_state_history
        - update_state
        - aupdate_state
        - bulk_update_state
        - abulk_update_state
        - get_graph
        - aget_graph
        - get_subgraphs
        - aget_subgraphs
        - with_config

::: langgraph.graph.message
    options:
      members:
        - add_messages