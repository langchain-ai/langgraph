### Do I need to use LangChain in order to use LangGraph?

No! LangGraph is a general-purpose framework and can be used for defining and executing arbitrary computation flows. For LLM applications, you could directly use it with the provider client libraries.

### Does LangGraph work with any LangChain Model?

It depends on your use case. If you are using a tool calling agent like ReAct, you will need to make sure the model you are using supports tool calling (for example, `ChatOpenAI`, `ChatAnthropic`).