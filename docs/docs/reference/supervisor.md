# LangGraph Supervisor

::: langgraph_supervisor.supervisor
    options:
      members:
        - create_supervisor

::: langgraph_supervisor.handoff
    options:
      members:
        - create_handoff_tool
        - create_forward_message_tool