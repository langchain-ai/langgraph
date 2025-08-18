import ast
import os
from itertools import filterfalse
from typing import Dict, List, Tuple

ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
CLIENT_PATH = os.path.join(ROOT_PATH, "libs", "sdk-py", "langgraph_sdk", "client.py")
ASYNC_TO_SYNC_METHOD_MAP: Dict[str, str] = {
    "aclose": "close",
    "__aenter__": "__enter__",
    "__aexit__": "__exit__",
}


def get_class_methods(node: ast.ClassDef) -> List[str]:
    return [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def find_classes(tree: ast.AST) -> List[Tuple[str, List[str]]]:
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = get_class_methods(node)
            classes.append((node.name, methods))
    return classes


def compare_sync_async_methods(sync_methods: List[str], async_methods: List[str]) -> List[str]:
    sync_set = set(sync_methods)
    async_set = {ASYNC_TO_SYNC_METHOD_MAP.get(async_method, async_method) for async_method in async_methods}
    missing_in_sync = list(async_set - sync_set)
    missing_in_async = list(sync_set - async_set)
    return missing_in_sync + missing_in_async


def main():
    with open(CLIENT_PATH, "r") as file:
        tree = ast.parse(file.read())

    classes = find_classes(tree)
  
    def is_sync(class_spec: Tuple[str, List[str]]) -> bool:
        return class_spec[0].startswith("Sync")

    sync_class_name_to_methods = {class_name: class_methods for class_name, class_methods in filter(is_sync, classes)}
    async_class_name_to_methods = {class_name: class_methods for class_name, class_methods in filterfalse(is_sync, classes)}

    mismatches = []

    for async_class_name, async_class_methods in async_class_name_to_methods.items():
        sync_class_name = "Sync" + async_class_name
        sync_class_methods = sync_class_name_to_methods.get(sync_class_name, [])
        diff = compare_sync_async_methods(sync_class_methods, async_class_methods)
        if diff:
            mismatches.append((sync_class_name, async_class_name, diff))

    if mismatches:
        error_message = "Mismatches found between sync and async client methods:\n"
        for sync_class_name, async_class_name, diff in mismatches:
            error_message += f"{sync_class_name} vs {async_class_name}:\n"
            for method in diff:
                error_message += f"  - {method}\n"
        raise ValueError(error_message)

    print("All sync and async client methods match.")


if __name__ == "__main__":
    main()
