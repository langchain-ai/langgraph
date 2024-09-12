# Setup

To setup requirements for building docs you can run:

```bash
poetry install --with test
```

## Serving documentation locally

To run the documentation server locally you can run:

```bash
make serve-docs
```

## Execute notebooks

If you would like to automatically execute all of the notebooks, to mimic the "Run notebooks" GHA, you can run:

```bash
python docs/_scripts/prepare_notebooks_for_ci.py
./execute_notebooks.sh
```

**Note**: if you want to run the notebooks without `%pip install` cells, you can run:

```bash
python docs/_scripts/prepare_notebooks_for_ci.py --comment-install-cells
./execute_notebooks.sh
```

`prepare_notebooks_for_ci.py` script will add VCR cassette context manager for each cell in the notebook, so that:
* when the notebook is run for the first time, cells with network requests will be recorded to a VCR cassette file
* when the notebook is run subsequently, the cells with network requests will be replayed from the cassettes

**Note**: this is currently limited only to the notebooks in `docs/docs/how-tos`

## Adding new notebooks

If you are adding a new notebook, please make sure to first run `prepare_notebooks_for_ci.py` script and

```bash
jupyter execute <path_to_notebook>
```

Once the notebook is executed, you should see the new VCR cassettes recorded in `docs/cassettes` directory.

## Updating existing notebooks

If you are updating an existing notebook, please make sure to remove any existing cassettes for the notebook and then run the steps from the "Adding new notebooks" section above.