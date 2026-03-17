# saf-python-sdk

Standalone Python SDK for the `advanced_graph` runtime backed by the Rust engine.

This package intentionally contains only:

- `saf_python_sdk.advanced_graph` (Python API)
- `langgraph_rust_core` (Rust execution engine via PyO3)

It does not package the original `langgraph` `stategraph` stack.

## Moved assets

- Original advanced graph design doc: `evolve-extend-langgraph.md`
- Original advanced graph tests: `tests/advanced-graph/`

