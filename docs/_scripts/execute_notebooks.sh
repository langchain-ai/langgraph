#!/bin/bash

# Read the list of notebooks to skip from the JSON file
SKIP_NOTEBOOKS=$(python -c "import json; print('\n'.join(json.load(open('notebooks_no_execution.json'))))")

# Function to execute a single notebook
execute_notebook() {
    file="$1"
    echo "Starting execution of $file"
    start_time=$(date +%s)
    if ! output=$(time uv run jupyter execute "$file" 2>&1); then
        end_time=$(date +%s)
        execution_time=$((end_time - start_time))
        echo "Error in $file. Execution time: $execution_time seconds"
        echo "Error details: $output"
        exit 1
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
    notebooks=$(find docs/tutorials docs/how-tos -name "*.ipynb" | grep -v ".ipynb_checkpoints" | grep -vFf <(echo "$SKIP_NOTEBOOKS"))
fi

# Execute notebooks sequentially
for file in $notebooks; do
    execute_notebook "$file"
done