# Multi-Agent Billing Desk with SupraWall Security

This example demonstrates how to secure a [LangGraph](https://github.com/langchain-ai/langgraph) multi-agent workflow using the [SupraWall](https://supra-wall.com) security layer.

## The Story: AI Billing Desk

In this scenario, we have a billing support system with three specialized agents:

1.  **Support Agent (`support-agent`)**: Handles general inquiries and customer history.
2.  **Refund Agent (`refund-agent`)**: Authorized to process refunds.
3.  **Ops Agent (`ops-agent`)**: Authorized for destructive actions like deleting customer records.

### The Security Challenge
Without SupraWall, if the `Support Agent` is compromised via prompt injection, it might try to call the `delete_customer_record` tool even though it shouldn't have access.

### The SupraWall Solution
By using the `SupraWallCallbackHandler` with unique `agent_id` values for each node, SupraWall enforces **least-privilege policies at the network layer**. Even if an agent is tricked into calling a restricted tool, the SupraWall Gateway will block the request and log a policy violation in your dashboard.

## Setup

1.  **Install dependencies**:
    ```bash
    pip install langgraph langchain-openai langchain-suprawall
    ```

2.  **Configure Environment**:
    ```bash
    export OPENAI_API_KEY="sk-..."
    export SUPRAWALL_API_KEY="sw_..."
    ```

3.  **SupraWall Policies**:
    Ensure your SupraWall dashboard has policies defined for the following identities:
    *   `support-agent`: Allowed `get_customer_history`. Denied others.
    *   `refund-agent`: Allowed `process_refund` (with a limit, e.g., < $100).
    *   `ops-agent`: `delete_customer_record` set to `REQUIRE_APPROVAL`.

## Running the Example

```bash
python example.py
```

The script will demonstrate three scenarios:
1.  **Success**: Support agent querying customer history.
2.  **Limited Success**: Refund agent processing a small refund.
3.  **Blocked/Approval**: Ops agent trying to delete a record, triggering an approval requirement or block depending on your policy.

## Why this matters for Production
In production multi-agent systems, agents often share the same model and tools. SupraWall provides an external, verifiable audit log and enforcement layer that works independently of the LLM's "intent," providing a robust safety net for financial and destructive operations.
