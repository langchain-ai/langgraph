for file in $(find docs/docs/how-tos docs/docs/tutorials -name "*.ipynb" | grep -v ".ipynb_checkpoints")
do
  echo "Executing $file"
  poetry run jupyter execute "$file"
done