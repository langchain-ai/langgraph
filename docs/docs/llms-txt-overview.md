# LLMs-txt for LangGraph

## Overview

LangGraph provides documentation files in the [`llms.txt`](https://llmstxt.org/) format, specifically `llms.txt` and `llms-full.txt`. These files allow large language models (LLMs) and agents to access programming documentation and APIs, particularly useful within integrated development environments (IDEs).

| Language Version | llms.txt                                                                                                   | llms-full.txt                                                                                                        |
|------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| LangGraph Python | [https://langchain-ai.github.io/langgraph/llms.txt](https://langchain-ai.github.io/langgraph/llms.txt)     | [https://langchain-ai.github.io/langgraph/llms-full.txt](https://langchain-ai.github.io/langgraph/llms-full.txt)     |
| LangGraph JS     | [https://langchain-ai.github.io/langgraphjs/llms.txt](https://langchain-ai.github.io/langgraphjs/llms.txt) | [https://langchain-ai.github.io/langgraphjs/llms-full.txt](https://langchain-ai.github.io/langgraphjs/llms-full.txt) |

## Differences Between `llms.txt` and `llms-full.txt`

- **`llms.txt`** is an index file containing links with brief descriptions of the content. An LLM or agent must follow these links to access detailed information.

- **`llms-full.txt`** includes all the detailed content directly in a single file, eliminating the need for additional navigation.

A key consideration when using `llms-full.txt` is its size. For extensive documentation, this file may become too large to fit into an LLM's context window.

## Using `llms.txt` via an MCP Server

As of March 9, 2025, IDEs [do not yet have robust native support for `llms.txt`](https://x.com/jeremyphoward/status/1902109312216129905?t=1eHFv2vdNdAckajnug0_Vw&s=19). However, you can utilize `llms.txt` effectively through an MCP server.

We provide an MCP server specifically designed to serve documentation, called [`mcpdoc`](https://github.com/langchain-ai/mcpdoc). This setup is compatible with IDEs and platforms such as Cursor, Windsurf, Claude, and Claude Code. Instructions for using `mcpdoc` with these tools are available in the repository.

## Using `llms-full.txt`

The LangGraph `llms-full.txt` file typically contains several hundred thousand tokens, exceeding the context window limitations of most LLMs. To effectively use this file:

1. **With IDEs (e.g., Cursor, Windsurf)**:
    - Add the `llms-full.txt` as custom documentation. The IDE will automatically chunk and index the content, implementing Retrieval-Augmented Generation (RAG).

2. **Without IDE support**:
    - Use a chat model with a large context window.
    - Implement a RAG strategy to manage and query the documentation efficiently.

