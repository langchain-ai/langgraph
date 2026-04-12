# LangGraph + SupraWall: Multi-Agent Billing Desk

This example demonstrates how to build a production-grade billing support desk with LangGraph, secured by [SupraWall](https://supra-wall.com).

## Overview

In multi-agent systems, agents often have access to sensitive tools (Refunds, PII, Destructive Actions). supra-wall provides a **network-level security layer** that enforces permissions per agent identity, even if your LLM is tricked by a prompt injection attack.

### Agents
- `support-agent`: Can view customer history.
- `refund-agent`: Can process refunds under $500.
- `ops-agent`: Requires human approval for account deletions.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY=sk-...
   export SUPRAWALL_API_KEY=sw_...
   ```

## Running

```bash
python example.py
```

## LangGraph Studio Support

This example includes a `langgraph.json` configuration. You can open this directory directly in **LangGraph Studio** to visualize the graph and simulate policy violations in the UI.
