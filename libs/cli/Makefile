.PHONY: lint format

lint:
	poetry run ruff .
	poetry run ruff format . --diff
	poetry run mypy .

format:
	poetry run ruff format .
	poetry run ruff --select I --fix .

test:
	poetry run pytest tests