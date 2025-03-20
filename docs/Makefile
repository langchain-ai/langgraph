.PHONY: lint-docs format-docs build-docs serve-docs serve-clean-docs clean-docs codespell build-typedoc llms-text build-prebuilt tests

build-typedoc:
	cd ../libs/sdk-js && yarn install --include-dev && yarn typedoc
	cd ../libs/sdk-js && yarn --silent concat-md --decrease-title-levels --ignore=js_ts_sdk_ref.md --start-title-level-at 2 docs > ../../docs/docs/cloud/reference/sdk/js_ts_sdk_ref.md 2>/dev/null
	#  Add links to the monorepo
	sed -e '1,10s|@langchain/langgraph-sdk|[@langchain/langgraph-sdk](https://github.com/langchain-ai/langgraph/tree/main/libs/sdk-js)|g' docs/cloud/reference/sdk/js_ts_sdk_ref.md > temp_file && mv temp_file docs/cloud/reference/sdk/js_ts_sdk_ref.md

build-prebuilt:
	# Use to create an update to date prebuilt page.
	# Looks up download stats for each of the prebuilt packages and
	# generates the final prebuilt page.
	@if [ "$(DOWNLOAD_STATS)" = "true" ]; then \
		set -x; \
		poetry run python -m _scripts.third_party_page.get_download_stats stats.yml; \
		set +x; \
	else \
		set -x; \
		poetry run python -m _scripts.third_party_page.get_download_stats --fake stats.yml; \
		set +x; \
	fi
	poetry run python -m _scripts.third_party_page.create_third_party_page stats.yml docs/prebuilt.md --language python

build-docs: build-typedoc build-prebuilt
	poetry run python -m mkdocs build --clean -f mkdocs.yml --strict

llms-text:
	poetry run python -m _scripts.generate_llms_text docs/llms-full.txt

install-vercel-deps:
	dnf install -y python3.11
	curl -sSL https://install.python-poetry.org | python3 -
	poetry self update 1.8.5
	# don't use vercel's python - it wasn't compiled with sqlite support, and it fails when installing ipython's kernel
	poetry env use /usr/bin/python3.11
	poetry install --with docs --with test --no-root

tests:
	# Run unit tests
	poetry run pytest tests/unit_tests


vercel-build-docs: install-vercel-deps
	make build-docs


serve-clean-docs: clean-docs
	poetry run python -m mkdocs serve -c -f mkdocs.yml --strict -w ../libs/langgraph

serve-docs: build-typedoc
	poetry run python -m mkdocs serve -f mkdocs.yml -w ../libs/langgraph  -w ../libs/checkpoint -w ../libs/sdk-py --dirty

clean-docs:
	find ./docs -name "*.ipynb" -type f -delete
	rm -rf site

## Run format against the project documentation.
format-docs:
	poetry run ruff format docs
	poetry run ruff check --fix docs

# Check the docs for linting violations
lint-docs:
	poetry run ruff format --check docs
	poetry run ruff check docs

codespell:
	./codespell_notebooks.sh .

start-services:
	docker compose -f test-compose.yml up -V --force-recreate --wait --remove-orphans

stop-services:
	docker compose -f test-compose.yml down
