.PHONY: lint format

lint:
	poetry run ruff check .
	poetry run ruff format . --diff
	poetry run mypy .

format:
	poetry run ruff format .
	poetry run ruff check --select I --fix .
