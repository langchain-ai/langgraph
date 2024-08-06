def test_quickstart(snapshot) -> None:
    from typing import Annotated, TypedDict

    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        # Messages have the type "list". The `add_messages` function
        # in the annotation defines how this state key should be updated
        # (in this case, it appends messages to the list, rather than overwriting them)
        messages: Annotated[list, add_messages]

    builder = StateGraph(State)

    from langchain_openai import ChatOpenAI

    model = ChatOpenAI()

    def chatbot(state: State):
        answer = model.invoke(state["messages"])
        return {"messages": [answer]}

    # The first argument is the unique node name
    # The second argument is the function or Runnable to run
    builder.add_node("chatbot", chatbot)

    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    graph = builder.compile()

    assert graph.get_graph().draw_mermaid() == snapshot
