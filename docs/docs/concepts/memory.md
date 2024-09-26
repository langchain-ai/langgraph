# Memory

## What is Memory?

Memory refers to the processing of data from the past to make it more useful for an application. Examples include:

- Managing what messages (e.g., from a long message history) are sent to a chat model;
- Summarizing or distilling past conversations or other information to inform model responses;
- Fetching structured information gleaned from previous conversations to inform model responses.

These features are common used to accommodate restricted LLM context windows, manage latency and costs,
and to tune the quality of an application's output.


## Tiers of memory 

**LLM memory**
* Enabled by: context window 
* Timescale: lifetime of LLM invocation 

**Short-term application memory**
* Enabled by: Chat-messages 
* Timescale: lifetime of [graph execution](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9291220) 

**Long-term application memory**
* Enabled by: [Chat history management](https://python.langchain.com/docs/how_to/chatbots_memory/#chat-history), [Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/#persistence)
* Timescale: set by choice of persistence layer

## LLM memory

### Motivation

What is a model context window

### Approaches 

Prompt caching (e.g., Anthropic)

### Use-cases

RAG: using prompt catching to reduce token usage

## Short-term application memory

### Motivation

Why do we need to manage history

### Approaches    

Trimming and filtering messages 
https://github.com/langchain-ai/langchain-academy/blob/main/module-2/trim-filter-messages.ipynb
https://python.langchain.com/docs/how_to/chatbots_memory/#trimming-messages

Summarizing messages
https://github.com/langchain-ai/langchain-academy/blob/main/module-2/chatbot-summarization.ipynb
https://python.langchain.com/docs/how_to/chatbots_memory/#summary-memory

Storing and retrieving facts or observations from a database during a conversation
https://github.com/langchain-ai/langgraph/pull/1850

Chatbot memory 
https://python.langchain.com/docs/how_to/chatbots_memory/

### Use-cases

Long-running chatbots without token creep 

* Uses trimming / filtering or summarization to manage memory

## Long-term application memory

### Approaches

LangChain built-in message history classes
https://python.langchain.com/docs/how_to/chatbots_memory/#chat-history

Using threads with persistence layer 
https://langchain-ai.github.io/langgraph/concepts/persistence/#persistence

Using SharedValue with persistence layer 
https://github.com/langchain-ai/langgraph/pull/1783

### Use-cases

Multi-turn chat with interruptions

* Uses persistence layer to save chat history

Persisting information across executions

* Uses SharedValue to save specific keys 

