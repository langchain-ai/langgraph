from typing import Any, Optional

from langchain_core.callbacks import Callbacks
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import COPIABLE_KEYS, DEFAULT_RECURSION_LIMIT

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.constants import CONFIG_KEY_CHECKPOINT_MAP


def patch_configurable(
    config: Optional[RunnableConfig], patch: dict[str, Any]
) -> RunnableConfig:
    if config is None:
        return {"configurable": patch}
    else:
        return {**config, "configurable": {**config["configurable"], **patch}}


def patch_checkpoint_map(
    config: RunnableConfig, metadata: Optional[CheckpointMetadata]
) -> RunnableConfig:
    if parents := (metadata.get("parents") if metadata else None):
        return patch_configurable(
            config,
            {
                CONFIG_KEY_CHECKPOINT_MAP: {
                    **parents,
                    config["configurable"]["checkpoint_ns"]: config["configurable"][
                        "checkpoint_id"
                    ],
                },
            },
        )
    else:
        return config


def merge_configs(*configs: Optional[RunnableConfig]) -> RunnableConfig:
    """Merge multiple configs into one.

    Args:
        *configs (Optional[RunnableConfig]): The configs to merge.

    Returns:
        RunnableConfig: The merged config.
    """
    base: RunnableConfig = {}
    # Even though the keys aren't literals, this is correct
    # because both dicts are the same type
    for config in configs:
        if config is None:
            continue
        for key in config:
            if key == "metadata":
                base[key] = {  # type: ignore
                    **base.get(key, {}),  # type: ignore
                    **(config.get(key) or {}),  # type: ignore
                }
            elif key == "tags":
                base[key] = sorted(  # type: ignore
                    set(base.get(key, []) + (config.get(key) or [])),  # type: ignore
                )
            elif key == "configurable":
                base[key] = {  # type: ignore
                    **base.get(key, {}),  # type: ignore
                    **(config.get(key) or {}),  # type: ignore
                }
            elif key == "callbacks":
                base_callbacks = base.get("callbacks")
                these_callbacks = config["callbacks"]
                # callbacks can be either None, list[handler] or manager
                # so merging two callbacks values has 6 cases
                if isinstance(these_callbacks, list):
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        base["callbacks"] = base_callbacks + these_callbacks
                    else:
                        # base_callbacks is a manager
                        mngr = base_callbacks.copy()
                        for callback in these_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                elif these_callbacks is not None:
                    # these_callbacks is a manager
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        mngr = these_callbacks.copy()
                        for callback in base_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                    else:
                        # base_callbacks is also a manager
                        base["callbacks"] = base_callbacks.merge(these_callbacks)
            elif key == "recursion_limit":
                if config["recursion_limit"] != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = config["recursion_limit"]
            elif key in COPIABLE_KEYS and config[key] is not None:  # type: ignore[literal-required]
                base[key] = config[key].copy()  # type: ignore[literal-required]
            else:
                base[key] = config[key] or base.get(key)  # type: ignore
    return base


def patch_config(
    config: Optional[RunnableConfig],
    *,
    callbacks: Optional[Callbacks] = None,
    recursion_limit: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    run_name: Optional[str] = None,
    configurable: Optional[dict[str, Any]] = None,
) -> RunnableConfig:
    """Patch a config with new values.

    Args:
        config (Optional[RunnableConfig]): The config to patch.
        callbacks (Optional[BaseCallbackManager], optional): The callbacks to set.
          Defaults to None.
        recursion_limit (Optional[int], optional): The recursion limit to set.
          Defaults to None.
        max_concurrency (Optional[int], optional): The max concurrency to set.
          Defaults to None.
        run_name (Optional[str], optional): The run name to set. Defaults to None.
        configurable (Optional[Dict[str, Any]], optional): The configurable to set.
          Defaults to None.

    Returns:
        RunnableConfig: The patched config.
    """
    config = config or {}
    if callbacks is not None:
        # If we're replacing callbacks, we need to unset run_name
        # As that should apply only to the same run as the original callbacks
        config["callbacks"] = callbacks
        if "run_name" in config:
            del config["run_name"]
        if "run_id" in config:
            del config["run_id"]
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if run_name is not None:
        config["run_name"] = run_name
    if configurable is not None:
        config["configurable"] = {**config.get("configurable", {}), **configurable}
    return config
