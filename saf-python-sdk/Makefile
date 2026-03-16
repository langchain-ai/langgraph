
PYPI_REPOSITORY ?= pypi
PYPI_TOKEN ?=
PYTHON_VERSION ?= 3.13
TEST_BASE = uv run --python $(PYTHON_VERSION) --with pytest --with anyio --with typing_extensions --with pydantic --with langgraph pytest -q -s

.PHONY: publish-to-pypi-saf-python-sdk all-tests test_primitives test_sub_agents test_update_elision test_run_pool_size test-benchmark
publish-to-pypi-saf-python-sdk:
	@test -n "$(PYPI_TOKEN)" || (echo "PYPI_TOKEN is required"; exit 1)
	cd "$(CURDIR)" && \
	MATURIN_PYPI_TOKEN="$(PYPI_TOKEN)" \
	uvx maturin publish --repository $(PYPI_REPOSITORY) --non-interactive --skip-existing --no-sdist

all-tests:
	$(TEST_BASE) tests/advanced-graph/test_*.py

test_primitives:
	$(TEST_BASE) tests/advanced-graph/test_primitives.py

test_sub_agents:
	$(TEST_BASE) tests/advanced-graph/test_sub_agents.py

test_update_elision:
	$(TEST_BASE) tests/advanced-graph/test_update_elision.py

test_run_pool_size:
	$(TEST_BASE) tests/advanced-graph/test_run_pool_size.py

test-benchmark:
	# Increase per-process file descriptor limit for high-concurrency benchmark runs.
	ulimit -n 65536 || true; \
	uv run --python $(PYTHON_VERSION) --with anyio --with typing_extensions --with pydantic --with langgraph python tests/advanced-graph/benchmark_stategraph_vs_advancedgraph.py

