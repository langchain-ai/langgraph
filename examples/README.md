# LangGraph examples

This directory is retained purely for archival purposes and is no longer updated.

## 🤔 What is this?

The examples previously found here have been moved to the consolidated LangChain documentation. This directory remains available for historical reference, but new examples and usage guidance are published in the docs.

## 📖 Documentation

For up-to-date LangGraph examples, tutorials, and guides, see the [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/overview). Get started with the [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart).


## OpenAI-compatible multi-model gateways

Several examples use `langchain_openai.ChatOpenAI`. The same client works with any OpenAI-compatible endpoint via `base_url` (or `OPENAI_API_BASE` / `OPENAI_BASE_URL`), for example a multi-model gateway like [DaoXE](https://daoxe.com) at `https://api.daoxe.com/v1`:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.daoxe.com/v1",
    api_key="...",
)
```
