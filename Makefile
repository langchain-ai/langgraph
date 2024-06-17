.PHONY: all clean format lint test tests test_watch integration_tests docker_tests help extended_tests coverage spell_check spell_fix build-docs serve-docs serve-clean-docs clean-docs

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

build-docs:
	poetry run python docs/_scripts/copy_notebooks.py
	poetry run mkdocs build --clean -f docs/mkdocs.yml --strict

serve-clean-docs: clean-docs
	poetry run python docs/_scripts/copy_notebooks.py
	poetry run python -m mkdocs serve -c -f docs/mkdocs.yml --strict -w ./langgraph

serve-docs:
	poetry run python docs/_scripts/copy_notebooks.py
	poetry run python -m mkdocs serve -f docs/mkdocs.yml -w ./langgraph --dirty

clean-docs:
	find ./docs/docs -name "*.ipynb" -type f -delete
	rm -rf docs/site