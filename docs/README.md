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

This will start the documentation server on [http://127.0.0.1:8000/langgraph/](http://127.0.0.1:8000/langgraph/).

## Execute notebooks

If you would like to automatically execute all of the notebooks, to mimic the "Run notebooks" GHA, you can run:

```bash
python _scripts/prepare_notebooks_for_ci.py
./_scripts/execute_notebooks.sh
```

**Note**: if you want to run the notebooks without `%pip install` cells, you can run:

```bash
python _scripts/prepare_notebooks_for_ci.py --comment-install-cells
./_scripts/execute_notebooks.sh
```

`prepare_notebooks_for_ci.py` script will add VCR cassette context manager for each cell in the notebook, so that:
* when the notebook is run for the first time, cells with network requests will be recorded to a VCR cassette file
* when the notebook is run subsequently, the cells with network requests will be replayed from the cassettes

## Adding new notebooks

If you are adding a notebook with API requests, it's **recommended** to record network requests so that they can be subsequently replayed. If this is not done, the notebook runner will make API requests every time the notebook is run, which can be costly and slow.

To record network requests, please make sure to first run `prepare_notebooks_for_ci.py` script.

Then, run

```bash
jupyter execute <path_to_notebook>
```

Once the notebook is executed, you should see the new VCR cassettes recorded in `cassettes` directory and discard the updated notebook.

## Updating existing notebooks

If you are updating an existing notebook, please make sure to remove any existing cassettes for the notebook in `cassettes` directory (each cassette is prefixed with the notebook name), and then run the steps from the "Adding new notebooks" section above.

To delete cassettes for a notebook, you can run:

```bash
rm cassettes/<notebook_name>*
```