.PHONY: format lint test

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ty check

test:
	uv run pytest $(TEST)
