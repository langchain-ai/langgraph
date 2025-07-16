# Observability

Use [LangSmith](https://smith.langchain.com/) to visualize the execution of a LangGraph application and do the following:

- [Enable tracing for your application](#enable-tracing-for-your-application).
- [Debug a locally running application](../cloud/how-tos/clone_traces_studio.md).
- [Evaluate the application performance](../agents/evals.md).
- [Monitor the application](https://docs.smith.langchain.com/observability/how_to_guides/dashboards).

To get started, sign up for a free account at [LangSmith](https://smith.langchain.com/).

## Enable tracing for your application

To use LangSmith with your LangGraph application, enable tracing:

```python
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
```

For more information, see [Trace with LangGraph](https://docs.smith.langchain.com/observability/how_to_guides/trace_with_langgraph).