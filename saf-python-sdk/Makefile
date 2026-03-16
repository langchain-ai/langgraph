PYPI_REPOSITORY ?= pypi
PYPI_TOKEN ?=

.PHONY: publish-to-pypi-saf-python-sdk
publish-to-pypi-saf-python-sdk:
	@test -n "$(PYPI_TOKEN)" || (echo "PYPI_TOKEN is required"; exit 1)
	cd "$(CURDIR)" && \
	MATURIN_PYPI_TOKEN="$(PYPI_TOKEN)" \
	uvx maturin publish --repository $(PYPI_REPOSITORY) --non-interactive --skip-existing --no-sdist

