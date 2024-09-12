for file in $(find docs/docs/how-tos -name "*.ipynb" | grep -v ".ipynb_checkpoints")
do
  echo "Executing $file"
  jupyter execute "$file"
done