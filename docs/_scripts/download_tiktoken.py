import tiktoken

# This will trigger the download and caching of the necessary files
for encoding in ("gpt2", "cl100k_base"):
    tiktoken.encoding_for_model(encoding)