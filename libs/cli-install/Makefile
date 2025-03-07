.PHONY: install format lint clean build publish

install:
	poetry install

format:
	poetry run ruff format langgraph_cli_install tests

lint:
	poetry run ruff check langgraph_cli_install tests
	
test:
	poetry run pytest

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

build: clean
	poetry build

publish: build
	poetry publish