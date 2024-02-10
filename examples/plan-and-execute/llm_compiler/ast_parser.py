import ast
import re
from typing import Dict, Union


def get_args(s: str) -> Dict[str, Union[str, bool, int, list, dict, None]]:
    # Extract the argument string
    args_str = re.search(r"\((.*?)\)", s).group(1)

    # Split the arguments on comma, considering nested structures
    args = re.split(r",(?![^[]*\]|[^(]*\))", args_str)

    # Create a dictionary from the split arguments
    args_dict = {}
    for arg in args:
        key, value = arg.split("=", 1)
        key = key.strip()
        value = ast.literal_eval(value.strip())
        args_dict[key] = value

    return args_dict


if __name__ == "__main__":
    # Should work on all these cases:
    signatures = [
        'func(a="foo", b=1, c=None, d=[1, 2, 3], e={"a": 1, "b": 2})',
        'another_func(idk={"nesting": {\'is\': ["fun", "right?"]}})',
        'once_more(a="How do you know that a = b?")',
    ]
    expected = [
        {"a": "foo", "b": 1, "c": None, "d": [1, 2, 3], "e": {"a": 1, "b": 2}},
        {"idk": {"nesting": {"is": ["fun", "right?"]}}},
        {"a": "How do you know that a = b?"},
    ]
    for i, s in enumerate(signatures):
        assert get_args(s) == expected[i]
