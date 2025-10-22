.PHONY: format lint test test-integration clean

format:
	ruff format .
	ruff check --select I --fix .

lint:
	ruff check .
	mypy --strict langgraph

test:
	pytest tests/ -v --tb=short

test-integration:
	pytest tests/ -v --tb=short -m integration

test-unit:
	pytest tests/ -v --tb=short -m "not integration"

test-watch:
	pytest-watcher

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

install:
	uv sync

build:
	uv build

publish:
	uv publish
