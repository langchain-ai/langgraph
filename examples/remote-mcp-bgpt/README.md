# Remote MCP: BGPT Scientific Evidence Agent

Example LangGraph ReAct agent that loads tools from the hosted [BGPT](https://bgpt.pro/mcp/) MCP server via [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters).

BGPT searches scientific papers and returns structured evidence fields: methods, sample sizes, limitations, conflicts of interest, data availability, and falsifiability prompts.

## Endpoints

| Type | URL |
|------|-----|
| MCP (streamable HTTP) | `https://bgpt.pro/mcp/stream` |
| REST search | `POST https://bgpt.pro/api/mcp-search` |
| REST DOI lookup | `POST https://bgpt.pro/api/mcp-doi-lookup` |

Free tier works without an API key.

## Setup

```bash
pip install langgraph langchain-mcp-adapters "langchain[openai]" mcp
```

## List tools

```bash
python bgpt_research_agent.py --list-tools
```

## Run a query

```bash
export OPENAI_API_KEY=sk-...
python bgpt_research_agent.py --query "semaglutide cardiovascular outcomes"
```

## Related

- langchain-mcp-adapters BGPT example: https://github.com/langchain-ai/langchain-mcp-adapters/tree/main/examples/clients/bgpt_streamable_http
- BGPT docs: https://bgpt.pro/mcp/
