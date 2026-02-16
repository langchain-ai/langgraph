.PHONY: format lint test

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

test:
	uv run pytest $(TEST)
