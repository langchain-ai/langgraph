errors=()  # Initialize an array to collect errors

for file in $(find docs/docs/how-tos docs/docs/tutorials -name "*.ipynb" | grep -v ".ipynb_checkpoints")
do
  echo "Executing $file"
  if ! output=$(poetry run jupyter execute "$file" 2>&1); then
    errors+=("$file: $output")  # Add a tuple of the file and error message to the errors list
  fi
done

# Optionally, print the errors
if [ ${#errors[@]} -ne 0 ]; then
  echo "Errors occurred in the following files:"
  printf '%s\n' "${errors[@]}"
  exit 1  # Exit with an error code if there are errors
fi