.PHONY: build-docs serve-docs serve-clean-docs clean-docs

build-docs:
	poetry run python docs/_scripts/copy_notebooks.py
	poetry run python -m mkdocs build --clean -f docs/mkdocs.yml --strict

serve-clean-docs: clean-docs
	poetry run python docs/_scripts/copy_notebooks.py
	poetry run python -m mkdocs serve -c -f docs/mkdocs.yml --strict -w ./libs/langgraph

serve-docs:
	poetry run python docs/_scripts/copy_notebooks.py
	poetry run python -m mkdocs serve -f docs/mkdocs.yml -w ./libs/langgraph --dirty

clean-docs:
	find ./docs/docs -name "*.ipynb" -type f -delete
	rm -rf docs/site