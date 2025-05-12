# llms.txt

Below you can find a list of documentation files in the [`llms.txt`](https://llmstxt.org/) format, specifically `llms.txt` and `llms-full.txt`. These files allow large language models (LLMs) and agents to access programming documentation and APIs, particularly useful within integrated development environments (IDEs).

| Language Version | llms.txt                                                                                                   | llms-full.txt                                                                                                        |
|------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| LangGraph Python | [https://langchain-ai.github.io/langgraph/llms.txt](https://langchain-ai.github.io/langgraph/llms.txt)     | [https://langchain-ai.github.io/langgraph/llms-full.txt](https://langchain-ai.github.io/langgraph/llms-full.txt)     |
| LangGraph JS     | [https://langchain-ai.github.io/langgraphjs/llms.txt](https://langchain-ai.github.io/langgraphjs/llms.txt) | [https://langchain-ai.github.io/langgraphjs/llms-full.txt](https://langchain-ai.github.io/langgraphjs/llms-full.txt) |
| LangChain Python | [https://python.langchain.com/llms.txt](https://python.langchain.com/llms.txt)                             | N/A                                                                                                                  |
| LangChain JS     | [https://js.langchain.com/llms.txt](https://js.langchain.com/llms.txt)                                     | N/A                                                                                                                  |

!!! info "Review the output"

    Even with access to up-to-date documentation, current state-of-the-art models may not always generate correct code. Treat the generated code as a starting point, and always review it before shipping
    code to production.

## Differences Between `llms.txt` and `llms-full.txt`

- **`llms.txt`** is an index file containing links with brief descriptions of the content. An LLM or agent must follow these links to access detailed information.

- **`llms-full.txt`** includes all the detailed content directly in a single file, eliminating the need for additional navigation.

A key consideration when using `llms-full.txt` is its size. For extensive documentation, this file may become too large to fit into an LLM's context window.

## Using `llms.txt` via an MCP Server

As of March 9, 2025, IDEs [do not yet have robust native support for `llms.txt`](https://x.com/jeremyphoward/status/1902109312216129905?t=1eHFv2vdNdAckajnug0_Vw&s=19). However, you can still use `llms.txt` effectively through an MCP server.

### 🚀 Use the `mcpdoc` Server

We provide an **MCP server** that was designed to serve documentation for LLMs and IDEs:

👉 **[langchain-ai/mcpdoc GitHub Repository](https://github.com/langchain-ai/mcpdoc)**

This MCP server allows integrating `llms.txt` into tools like **Cursor**, **Windsurf**, **Claude**, and **Claude Code**.

📘 **Setup instructions and usage examples** are available in the repository.

## Using `llms-full.txt`

The LangGraph `llms-full.txt` file typically contains several hundred thousand tokens, exceeding the context window limitations of most LLMs. To effectively use this file:

1. **With IDEs (e.g., Cursor, Windsurf)**:
    - Add the `llms-full.txt` as custom documentation. The IDE will automatically chunk and index the content, implementing Retrieval-Augmented Generation (RAG).

2. **Without IDE support**:
    - Use a chat model with a large context window.
    - Implement a RAG strategy to manage and query the documentation efficiently.

