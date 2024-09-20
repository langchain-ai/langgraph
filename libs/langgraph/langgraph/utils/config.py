from collections import ChainMap
from typing import Any, Optional, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManager,
    BaseCallbackManager,
    CallbackManager,
    Callbacks,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import (
    CONFIG_KEYS,
    COPIABLE_KEYS,
    DEFAULT_RECURSION_LIMIT,
    var_child_runnable_config,
)

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.constants import CONFIG_KEY_CHECKPOINT_MAP


def patch_configurable(
    config: Optional[RunnableConfig], patch: dict[str, Any]
) -> RunnableConfig:
    if config is None:
        return {"configurable": patch}
    elif "configurable" not in config:
        return {**config, "configurable": patch}
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
        for key, value in config.items():
            if not value:
                continue
            if key == "metadata":
                if base_value := base.get(key):
                    base[key] = {**base_value, **value}  # type: ignore
                else:
                    base[key] = value  # type: ignore[literal-required]
            elif key == "tags":
                if base_value := base.get(key):
                    base[key] = [*base_value, *value]  # type: ignore
                else:
                    base[key] = value  # type: ignore[literal-required]
            elif key == "configurable":
                if base_value := base.get(key):
                    base[key] = {**base_value, **value}  # type: ignore
                else:
                    base[key] = value  # type: ignore[literal-required]
            elif key == "callbacks":
                base_callbacks = base.get("callbacks")
                # callbacks can be either None, list[handler] or manager
                # so merging two callbacks values has 6 cases
                if isinstance(value, list):
                    if base_callbacks is None:
                        base["callbacks"] = value.copy()
                    elif isinstance(base_callbacks, list):
                        base["callbacks"] = base_callbacks + value
                    else:
                        # base_callbacks is a manager
                        mngr = base_callbacks.copy()
                        for callback in value:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                elif isinstance(value, BaseCallbackManager):
                    # value is a manager
                    if base_callbacks is None:
                        base["callbacks"] = value.copy()
                    elif isinstance(base_callbacks, list):
                        mngr = value.copy()
                        for callback in base_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                    else:
                        # base_callbacks is also a manager
                        base["callbacks"] = base_callbacks.merge(value)
                else:
                    raise NotImplementedError
            elif key == "recursion_limit":
                if config["recursion_limit"] != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = config["recursion_limit"]
            else:
                base[key] = config[key]  # type: ignore[literal-required]
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
    config = config.copy() if config is not None else {}
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


def get_callback_manager_for_config(
    config: RunnableConfig, tags: Optional[Sequence[str]] = None
) -> CallbackManager:
    """Get a callback manager for a config.

    Args:
        config (RunnableConfig): The config.

    Returns:
        CallbackManager: The callback manager.
    """
    from langchain_core.callbacks.manager import CallbackManager

    # merge tags
    all_tags = config.get("tags")
    if all_tags is not None and tags is not None:
        all_tags = [*all_tags, *tags]
    elif tags is not None:
        all_tags = list(tags)
    # use existing callbacks if they exist
    if (callbacks := config.get("callbacks")) and isinstance(
        callbacks, CallbackManager
    ):
        if all_tags:
            callbacks.add_tags(all_tags)
        if metadata := config.get("metadata"):
            callbacks.add_metadata(metadata)
        return callbacks
    else:
        # otherwise create a new manager
        return CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=all_tags,
            inheritable_metadata=config.get("metadata"),
        )


def get_async_callback_manager_for_config(
    config: RunnableConfig,
    tags: Optional[Sequence[str]] = None,
) -> AsyncCallbackManager:
    """Get an async callback manager for a config.

    Args:
        config (RunnableConfig): The config.

    Returns:
        AsyncCallbackManager: The async callback manager.
    """
    from langchain_core.callbacks.manager import AsyncCallbackManager

    # merge tags
    all_tags = config.get("tags")
    if all_tags is not None and tags is not None:
        all_tags = [*all_tags, *tags]
    elif tags is not None:
        all_tags = list(tags)
    # use existing callbacks if they exist
    if (callbacks := config.get("callbacks")) and isinstance(
        callbacks, AsyncCallbackManager
    ):
        if all_tags:
            callbacks.add_tags(all_tags)
        if metadata := config.get("metadata"):
            callbacks.add_metadata(metadata)
        return callbacks
    else:
        # otherwise create a new manager
        return AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )


def ensure_config(*configs: Optional[RunnableConfig]) -> RunnableConfig:
    """Ensure that a config is a dict with all keys present.

    Args:
        config (Optional[RunnableConfig], optional): The config to ensure.
          Defaults to None.

    Returns:
        RunnableConfig: The ensured config.
    """
    empty = RunnableConfig(
        tags=[],
        metadata=ChainMap(),
        callbacks=None,
        recursion_limit=DEFAULT_RECURSION_LIMIT,
        configurable={},
    )
    if var_config := var_child_runnable_config.get():
        empty.update(
            {
                k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                for k, v in var_config.items()
                if v is not None
            },
        )
    for config in configs:
        if config is None:
            continue
        for k, v in config.items():
            if v is not None and k in CONFIG_KEYS:
                empty[k] = v  # type: ignore[literal-required]
        for k, v in config.items():
            if v is not None and k not in CONFIG_KEYS:
                empty["configurable"][k] = v
    for key, value in empty["configurable"].items():
        if (
            not key.startswith("__")
            and isinstance(value, (str, int, float, bool))
            and key not in empty["metadata"]
        ):
            empty["metadata"][key] = value
    return empty
