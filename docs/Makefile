.PHONY: lint-docs format-docs build-docs serve-docs serve-clean-docs clean-docs codespell llms-text build-prebuilt tests

build-prebuilt:
	# Use to create an update to date prebuilt page.
	# Looks up download stats for each of the prebuilt packages and
	# generates the final prebuilt page.
	@if [ "$(DOWNLOAD_STATS)" = "true" ]; then \
		set -x; \
		uv run python -m _scripts.third_party_page.get_download_stats stats.yml; \
		set +x; \
	else \
		set -x; \
		uv run python -m _scripts.third_party_page.get_download_stats --fake stats.yml; \
		set +x; \
	fi
	uv run python -m _scripts.third_party_page.create_third_party_page stats.yml docs/agents/prebuilt.md

build-docs: build-prebuilt
	TARGET_LANGUAGE=python uv run python -m mkdocs build --clean -f mkdocs.yml --strict

build-docs-js: build-prebuilt
	TARGET_LANGUAGE=js uv run python -m mkdocs build --clean -f mkdocs.yml --strict

llms-text:
	uv run python -m _scripts.generate_llms_text docs/llms-full.txt

install-vercel-deps:
	curl -sL "https://astral.sh/uv/install.sh" | bash -s
	export PATH="${HOME}/.cargo/bin:${PATH}"
	uv venv --python 3.11
	uv sync --all-groups

tests:
	# Run unit tests
	uv run pytest tests/unit_tests


vercel-build-docs: install-vercel-deps
	make build-docs


serve-clean-docs: clean-docs
	uv run python -m mkdocs serve -c -f mkdocs.yml --strict -w ../libs/langgraph

serve-docs:
	uv run python -m mkdocs serve -f mkdocs.yml -w ../libs/langgraph  -w ../libs/checkpoint -w ../libs/sdk-py --dirty

clean-docs:
	find ./docs -name "*.ipynb" -type f -delete
	rm -rf site

## Run format against the project documentation.
format-docs:
	uv run ruff format docs
	uv run ruff check --fix docs

# Check the docs for linting violations
lint-docs:
	uv run ruff format --check docs
	uv run ruff check docs

codespell:
	./codespell_notebooks.sh .

start-services:
	docker compose -f test-compose.yml up -V --force-recreate --wait --remove-orphans

stop-services:
	docker compose -f test-compose.yml down
