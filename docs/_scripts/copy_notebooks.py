import os
import shutil
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]

examples_dir = root_dir / "examples"
docs_dir = root_dir / "docs/docs"
how_tos_dir = docs_dir / "how-tos"
tutorials_dir = docs_dir / "tutorials"

_MANUAL = {
    "how-tos": [
        "async.ipynb",
        "streaming-tokens.ipynb",
        "human-in-the-loop.ipynb",
        "persistence.ipynb",
        "time-travel.ipynb",
        "visualization.ipynb",
        "state-model.ipynb",
        "subgraph.ipynb",
        "force-calling-a-tool-first.ipynb",
        "pass-run-time-values-to-tools.ipynb",
        "dynamic-returning-direct.ipynb",
        "managing-agent-steps.ipynb",
        "respond-in-format.ipynb",
        "branching.ipynb",
        "dynamically-returning-directly.ipynb",
        "configuration.ipynb",
        "map-reduce.ipynb",
        "extraction/retries.ipynb",
    ],
    "tutorials": [
        "introduction.ipynb",
        "customer-support/customer-support.ipynb",
        "tutorials/tnt-llm/tnt-llm.ipynb",
    ],
}
_MANUAL_INVERSE = {v: docs_dir / k for k, vs in _MANUAL.items() for v in vs}
_HOW_TOS = {"agent_executor", "chat_agent_executor_with_function_calling", "docs"}
_MAP = {
    "persistence_postgres.ipynb": "tutorial",
}
_HIDE = set(
    str(examples_dir / f)
    for f in [
        "persistence_postgres.ipynb",
        "agent_executor/base.ipynb",
        "agent_executor/force-calling-a-tool-first.ipynb",
        "agent_executor/high-level.ipynb",
        "agent_executor/human-in-the-loop.ipynb",
        "agent_executor/managing-agent-steps.ipynb",
        "chat_agent_executor_with_function_calling/anthropic.ipynb",
        "chat_agent_executor_with_function_calling/base.ipynb",
        "chat_agent_executor_with_function_calling/dynamically-returning-directly.ipynb",
        "chat_agent_executor_with_function_calling/force-calling-a-tool-first.ipynb",
        "chat_agent_executor_with_function_calling/high-level-tools.ipynb",
        "chat_agent_executor_with_function_calling/high-level.ipynb",
        "chat_agent_executor_with_function_calling/human-in-the-loop.ipynb",
        "chat_agent_executor_with_function_calling/managing-agent-steps.ipynb",
        "chat_agent_executor_with_function_calling/prebuilt-tool-node.ipynb",
        "chat_agent_executor_with_function_calling/respond-in-format.ipynb",
        "chatbots/customer-support.ipynb",
        "rag/langgraph_rag_agent_llama3_local.ipynb",
        "rag/langgraph_self_rag_pinecone_movies.ipynb",
        "rag/langgraph_adaptive_rag_cohere.ipynb",
    ]
)


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
                if src_path in _HIDE:
                    print("Hiding:", src_path)
                    continue
                dst_path = os.path.join(
                    dst_dir, os.path.relpath(src_path, examples_dir)
                )
                for k in _MANUAL_INVERSE:
                    if src_path.endswith(k):
                        overridden_dir = _MANUAL_INVERSE[k]
                        dst_path = os.path.join(
                            overridden_dir, os.path.relpath(src_path, examples_dir)
                        )
                        dst_path = dst_path.replace(
                            "tutorials/tutorials", "tutorials"
                        ).replace("how-tos/how-tos", "how-tos")
                        print(f"Overriding: {src_path} to {dst_path}")
                        break

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                print(f"Copying: {src_path} to {dst_path}")
                shutil.copy(src_path, dst_path)
                # Convert all ./img/* to ../img/*
                if file.endswith(".ipynb"):
                    with open(dst_path, "r") as f:
                        content = f.read()
                    content = content.replace("(./img/", "(../img/")
                    content = content.replace('src=\\"./img/', 'src=\\"../img/')
                    with open(dst_path, "w") as f:
                        f.write(content)
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
