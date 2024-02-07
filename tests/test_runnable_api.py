from langgraph.graph.message import MessageGraph


async def test_astream_events():
    graph_builder = MessageGraph()

    def foo(state: list) -> list:
        return []

    def bar(state: list) -> list:
        return []

    graph_builder.add_node("foo", foo)

    graph_builder.add_node("bar", bar)
    graph_builder.add_edge("foo", "bar")
    graph_builder.set_entry_point("foo")
    graph_builder.set_finish_point("bar")
    graph = graph_builder.compile()
    events = graph.astream_events(
        [],
        version="v1",
    )
    async for event in events:
        print(event)
