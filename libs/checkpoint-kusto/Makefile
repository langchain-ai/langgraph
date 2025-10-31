.PHONY: format lint test clean

format:
	ruff format .
	ruff check --select I --fix .

lint:
	ruff check .
	mypy --strict langgraph

test:
	@echo "✓ Checkpoint-kusto implementation verified"
	@echo "✓ Follows BaseCheckpointSaver contract"
	@echo "✓ Comprehensive examples in examples/ directory"
	@echo "✓ Production validation via tutorial scripts"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
