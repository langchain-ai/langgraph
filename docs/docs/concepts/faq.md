# FAQ

Common questions and their answers!

## Do I need to use LangChain in order to use LangGraph?

No! LangGraph is a general-purpose framework - the nodes and edges are nothing more than Python functions. You can use LangChain, raw HTTP requests, or even other frameworks inside these nodes and edges.

## Does LangGraph work with LLMs that don't support tool calling?

Yes! You can use LangGraph with any LLMs. The main reason we use LLMs that support tool calling is that this is often the most convenient way to have the LLM make its decision about what to do. If your LLM does not support tool calling, you can still use it - you just need to write a bit of logic to convert the raw LLM string response to a decision about what to do.

## Does LangGraph work with OSS LLMs?

Yes! LangGraph is totally ambivalent to what LLMs are used under the hood. The main reason we use closed LLMs in most of the tutorials is that they seamlessly support tool calling, while OSS LLMs often don't. But tool calling is not necessary (see [this section](#does-langgraph-work-with-llms-that-dont-support-tool-calling)) so you can totally use LangGraph with OSS LLMs.
