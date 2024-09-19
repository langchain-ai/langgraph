#!/bin/bash

# Read the list of notebooks to skip from the JSON file
SKIP_NOTEBOOKS=$(python -c "import json; print(' '.join(json.load(open('docs/notebooks_no_execution.json'))))")

errors=()  # Initialize an array to collect errors

for file in $(find docs/docs/tutorials -name "*.ipynb" | grep -v ".ipynb_checkpoints")
do
  # Check if the file is in the list of notebooks to skip
  if [[ $SKIP_NOTEBOOKS == *"$file"* ]]; then
    echo "Skipping $file (in no-execution list)"
    continue
  fi

  echo "Executing $file"
  if ! output=$(poetry run jupyter execute --allow-errors "$file" 2>&1); then
    if grep -q '"tags": \["no_execution"\]' "$file"; then
      echo "Notebook $file has no_execution tag, skipping error"
    else
      errors+=("$file: $output")  # Add a tuple of the file and error message to the errors list
      printf '%s\n' "${errors[@]}"
      exit 1
    fi
  fi
done

# Optionally, print the errors
if [ ${#errors[@]} -ne 0 ]; then
  echo "Errors occurred in the following files:"
  printf '%s\n' "${errors[@]}"
  exit 1  # Exit with an error code if there are errors
fi