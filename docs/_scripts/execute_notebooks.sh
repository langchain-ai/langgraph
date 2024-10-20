#!/bin/bash

# Set the concurrency level (default: 4)
CONCURRENCY=${CONCURRENCY:-4}

# Read the list of notebooks to skip from the JSON file
SKIP_NOTEBOOKS=$(python -c "import json; print('\n'.join(json.load(open('docs/notebooks_no_execution.json'))))")

# Function to execute a single notebook
execute_notebook() {
    local file="$1"
    echo "Starting execution of $file"
    start_time=$(date +%s)

    if ! output=$(poetry run jupyter execute "$file" 2>&1); then
        end_time=$(date +%s)
        execution_time=$((end_time - start_time))
        echo "Error in $file. Execution time: $execution_time seconds"
        echo "Error details: $output" >&2
        return 1
    fi

    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    echo "Finished $file. Execution time: $execution_time seconds"
}

export -f execute_notebook

# Check if custom notebook paths are provided
if [ $# -gt 0 ]; then
    notebooks=$(echo "$@" | tr ' ' '\n' | grep -vFf <(echo "$SKIP_NOTEBOOKS"))
else
    # Find all notebooks and filter out those in the skip list
    notebooks=$(find docs/docs/tutorials docs/docs/how-tos -name "*.ipynb" | grep -v ".ipynb_checkpoints" | grep -vFf <(echo "$SKIP_NOTEBOOKS"))
fi

# Display the found notebooks in a more readable way
echo "====================================="
echo "Notebooks to be executed:"
echo "====================================="
for notebook in $notebooks; do
    echo "- $notebook"
done

echo
echo "================================================================="
echo "Starting notebook execution with concurrency level: $CONCURRENCY"
echo "================================================================="
echo

# Run notebooks in parallel with controllable concurrency
echo "$notebooks" | xargs -n 1 -P "$CONCURRENCY" bash -c 'execute_notebook "$@"' _

# Check exit status and handle any errors
if [ $? -ne 0 ]; then
    echo "====================================="
    echo "One or more notebooks failed to execute."
    echo "====================================="
    echo >&2
    exit 1
fi

echo
echo "====================================="
echo "All notebooks processed successfully."
echo "====================================="
