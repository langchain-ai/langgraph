import tiktoken

# This will trigger the download and caching of the necessary files
for encoding in ("gpt2", "gpt-3.5"):
    tiktoken.encoding_for_model(encoding)