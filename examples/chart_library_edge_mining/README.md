# LangGraph + Chart Library — Edge-Mining Agent

Explicit StateGraph implementation of Chart Library's edge-mining loop
(cohort → explain → refine → report). Deterministic, checkpointable,
one LLM call per run (final write-up only; retrieval and analysis are
pure tool calls).

## Why LangGraph for this

The edge-mining loop is not a free-form tool-calling task. It's a
specific workflow:

1. Retrieve an initial cohort
2. Ask which filter moves the distribution most
3. Refine with that filter
4. Write up the comparison

A single-LLM-with-tools pattern works but wastes tokens on orchestration
the workflow already prescribes. LangGraph encodes the order explicitly,
so the model only handles the synthesis step, and every retrieval is
checkpointable via LangGraph's saver interface.

## Run

```bash
pip install langgraph langchain langchain-anthropic requests
export ANTHROPIC_API_KEY=sk-ant-...
export CHART_LIBRARY_KEY=cl_...     # free at chartlibrary.io/developers
python cohort_edge_mining_graph.py NVDA 2024-06-18
```

## Sample output

```
[initial cohort] cohort_id=coh_...  n=491/500  5d_p50=+0.48%  elapsed=1891ms
[explain] same_vix_bucket    n_after= 180  shift=-0.64pp
[explain] same_trend         n_after= 347  shift=+1.88pp <-- picked
[explain] recent_5y          n_after= 291  shift=+0.99pp
[refine] filter=same_trend  n=345/347  5d_p50=+0.62%  elapsed=12ms

─── REPORT ───
NVDA 2024-06-18 — baseline cohort of 491 historical analogs (54
delisted included) showed 54.0% above-entry at 5 days with p50
return +0.48%. Refining by same_trend narrowed to 345 analogs with
55.8% above-entry and p50 +0.62% (+1.88pp shift). The trend regime
is the most informative conditional for this setup...
```

## Extend

- Swap `node_report`'s synthesis LLM for any Anthropic/OpenAI/local model.
- Add branching: run `node_refine` twice with different winning filters
  and compare in the report.
- Plug in a LangGraph saver (memory / redis / postgres) for
  checkpointable runs.
- Add a `node_decide` that uses the distribution to set position size
  inside a live trading agent.

## Submit upstream

LangGraph examples live at `github.com/langchain-ai/langgraph` under
`examples/`. If you want to propose this as an upstream example, fork,
drop it in `examples/chart_library_edge_mining/`, and open a PR with
a short rationale pointing to the Chart Library docs + a live API key
demo in the notebook.
