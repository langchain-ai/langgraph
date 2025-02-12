.PHONY: lint-docs format-docs build-docs serve-docs serve-clean-docs clean-docs codespell build-typedoc llms-text

build-typedoc:
	cd ../libs/sdk-js && yarn install --include-dev && yarn typedoc
	cd ../libs/sdk-js && yarn --silent concat-md --decrease-title-levels --ignore=js_ts_sdk_ref.md --start-title-level-at 2 docs > ../../docs/docs/cloud/reference/sdk/js_ts_sdk_ref.md 2>/dev/null
	#  Add links to the monorepo
	sed -e '1,10s|@langchain/langgraph-sdk|[@langchain/langgraph-sdk](https://github.com/langchain-ai/langgraph/tree/main/libs/sdk-js)|g' docs/cloud/reference/sdk/js_ts_sdk_ref.md > temp_file && mv temp_file docs/cloud/reference/sdk/js_ts_sdk_ref.md

build-docs: build-typedoc
	poetry run python -m mkdocs build --clean -f mkdocs.yml --strict

llms-text:
	poetry run python _scripts/generate_llms_text.py docs/llms-full.txt

install-vercel-deps:
	dnf install -y sqlite-devel python3.11
	rm /usr/bin/python3
	ln -s /usr/bin/python3.11 /usr/bin/python3

	echo "python3 path: $$(which python3)"
	echo "python path:' $$(which python)"
	
	curl -sSL https://install.python-poetry.org | python3 -
	poetry self update 1.8.5
	poetry env use /usr/bin/python3.11
	poetry run python3 --version
	poetry run bash -c 'echo "poetry python3.11 path: $$(which python3.11)"'
	poetry run bash -c 'echo "poetry python3 path: $$(which python3)"'
	poetry run bash -c 'echo "poetry python path: $$(which python)"'
	poetry run bash -c 'echo $$PATH'
	poetry run bash -c 'echo $$PYTHONPATH'

vercel-build-docs: install-vercel-deps
	poetry install
	poetry run pip install "git+https://github.com/benjamincburns/markdown-exec.git@10cdd338bfdb1f99705b3a4d06b244f7f185ecae"
	poetry run pip install jupyter ipykernel
	poetry run python3 -m ipykernel install --user --name=python3
	poetry run jupyter kernelspec list
	npm install -g tslab
	poetry run tslab install --python=python3
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
