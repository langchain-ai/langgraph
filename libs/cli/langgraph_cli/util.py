def clean_empty_lines(input_str: str):
    return "\n".join(filter(None, input_str.splitlines()))
