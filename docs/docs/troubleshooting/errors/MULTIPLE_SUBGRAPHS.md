# MULTIPLE_SUBGRAPHS

You are calling subgraphs multiple times within a single LangGraph node with checkpointing enabled for each subgraph.

This is currently not allowed due to internal restrictions on how checkpoint namespacing for subgraphs works.

## Troubleshooting

The following may help resolve this error:

- If you don't need to interrupt/resume from a subgraph, pass `checkpointer=False` when compiling it like this: `.compile(checkpointer=False)`
- Don't imperatively call graphs multiple times in the same node, and instead use the [`Send`](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) API.
