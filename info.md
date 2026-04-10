# 🧠 LangGraph Overview (Quick Reference)

## 📌 What is LangGraph?
**LangGraph** is a framework for building **stateful, multi-step AI workflows** using graphs instead of linear chains.

It is part of the LangChain ecosystem and is especially useful for:
- Agent workflows
- Tool usage loops
- Complex decision trees
- Long-running AI processes

---

## 🧩 Core Concepts

### 🔄 Graph
A graph is a collection of **nodes** (steps) and **edges** (transitions).

```text
Start → Node A → Node B → End
           ↓
        Node C
