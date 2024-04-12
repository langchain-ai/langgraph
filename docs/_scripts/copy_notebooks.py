import os
import shutil
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]

examples_dir = root_dir / "examples"
docs_dir = root_dir / "docs/docs"
how_tos_dir = docs_dir / "how-tos"
tutorials_dir = docs_dir / "tutorials"

_HOW_TOS = {"agent_executor", "chat_agent_executor_with_function_calling", "docs"}
_MAP = {
    "persistence_postgres.ipynb": "tutorial",
}
_IGNORE = (".ipynb_checkpoints", ".venv", ".cache")


def clean_notebooks():
    roots = (how_tos_dir, tutorials_dir)
    for dir_ in roots:
        traversed = []
        for root, dirs, files in os.walk(dir_):
            for file in files:
                if file.endswith(".ipynb"):
                    os.remove(os.path.join(root, file))
            # Now delete the dir if it is empty now
            if root not in roots:
                traversed.append(root)

        for root in reversed(traversed):
            if not os.listdir(root):
                os.rmdir(root)


def copy_notebooks():
    # Nested ones are mostly tutorials rn
    for root, dirs, files in os.walk(examples_dir):
        # if root == str(examples_dir):
        #     continue
        if any(
            path.startswith(".") or path.startswith("__") for path in root.split(os.sep)
        ):
            continue

        if any(path in _HOW_TOS for path in root.split(os.sep)):
            dst_dir = how_tos_dir
        else:
            dst_dir = tutorials_dir
        for file in files:
            dst_dir_ = dst_dir
            if file.endswith((".ipynb", ".png")):
                if file in _MAP:
                    dst_dir = os.path.join(dst_dir, _MAP[file])
                src_path = os.path.join(root, file)
                dst_path = os.path.join(
                    dst_dir, os.path.relpath(src_path, examples_dir)
                )
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)
                dst_dir = dst_dir_
    # Top level notebooks are "how-to's"
    # for file in examples_dir.iterdir():
    #     if file.suffix.endswith(".ipynb") and not os.path.isdir(
    #         os.path.join(examples_dir, file)
    #     ):
    #         src_path = os.path.join(examples_dir, file)
    #         dst_path = os.path.join(docs_dir, "how-tos", file.name)
    #         shutil.copy(src_path, dst_path)


if __name__ == "__main__":
    clean_notebooks()
    copy_notebooks()
