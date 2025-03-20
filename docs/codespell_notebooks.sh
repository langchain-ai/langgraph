ERROR_FOUND=0
for file in $(find $1 -name "*.ipynb" | grep -v ".ipynb_checkpoints"); do
    # Adding regexp to ignore base64 strings
    OUTPUT=$(cat "$file" | jupytext --from ipynb --to py:percent | codespell --ignore-regex='[A-Za-z0-9+/=]{25,}' -)
    if [ -n "$OUTPUT" ]; then
        echo "Errors found in $file"
        echo "$OUTPUT"
        ERROR_FOUND=1
    fi
done

for file in $(find $1 -name "*.md"); do
    # Adding regexp to ignore base64 strings
    OUTPUT=$(cat "$file" | codespell --ignore-regex='[A-Za-z0-9+/=]{25,}' -)
    if [ -n "$OUTPUT" ]; then
        echo "Errors found in $file"
        echo "$OUTPUT"
        ERROR_FOUND=1
    fi
done

if [ "$ERROR_FOUND" -ne 0 ]; then
    exit 1
fi
