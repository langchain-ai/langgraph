ERROR_FOUND=0
for file in $(find $1 -name "*.ipynb"); do
    OUTPUT=$(cat "$file" | jupytext --from ipynb --to py:percent | codespell -)
    if [ -n "$OUTPUT" ]; then
        echo "Errors found in $file"
        echo "$OUTPUT"
        ERROR_FOUND=1
    fi
done

if [ "$ERROR_FOUND" -ne 0 ]; then
    exit 1
fi