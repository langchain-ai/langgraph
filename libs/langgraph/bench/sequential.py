"""Create a sequential no-op graph consisting of a few hundred nodes."""

from langgraph.graph import MessagesState, StateGraph
from langgraph.utils.runnable import RunnableCallable


def create_sequential(number_nodes: int) -> StateGraph:
    """Create a sequential no-op graph consisting of a few hundred nodes."""
    builder = StateGraph(MessagesState)

    def noop(state: MessagesState) -> None:
        """No-op function."""
        pass

    async def anoop(state: MessagesState) -> None:
        """No-op function."""
        pass

    prev_node = "__start__"

    for i in range(number_nodes):
        name = f"node_{i}"
        builder.add_node(name, RunnableCallable(noop, anoop))
        builder.add_edge(prev_node, name)
        prev_node = name

    builder.add_edge(prev_node, "__end__")
    return builder


if __name__ == "__main__":
    import asyncio
    import sys
    import time

    def get_event_loop_policy():
        """Get the best available event loop policy for the platform."""
        if sys.platform in ('linux', 'darwin'):
            try:
                import uvloop
                return uvloop.EventLoopPolicy()
            except ImportError:
                return asyncio.DefaultEventLoopPolicy()
        else:
            return asyncio.DefaultEventLoopPolicy()

    graph = create_sequential(3000).compile()
    input = {"messages": []}  # Empty list of messages
    config = {"recursion_limit": 20000000000}

    async def run():
        len([c async for c in graph.astream(input, config=config)])

    asyncio.set_event_loop_policy(get_event_loop_policy())
    start = time.time()
    asyncio.run(run())
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
