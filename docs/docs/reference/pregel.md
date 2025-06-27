# Pregel

::: langgraph.pregel.NodeBuilder
    options:
      show_if_no_docstring: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - subscribe_only
        - subscribe_to
        - read_from
        - do
        - write_to
        - meta
        - retry
        - cache
        - build

::: langgraph.pregel.Pregel
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