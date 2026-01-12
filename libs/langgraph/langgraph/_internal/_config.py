from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping, Sequence
from os import getenv
from typing import Any, cast

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
    var_child_runnable_config,
)
from langgraph.checkpoint.base import CheckpointMetadata

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_NS,
    NS_END,
    NS_SEP,
)

DEFAULT_RECURSION_LIMIT = int(getenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "10000"))


def recast_checkpoint_ns(ns: str) -> str:
    """Remove task IDs from checkpoint namespace.

    Args:
        ns: The checkpoint namespace with task IDs.

    Returns:
        str: The checkpoint namespace without task IDs.
    """
    return NS_SEP.join(
        part.split(NS_END)[0] for part in ns.split(NS_SEP) if not part.isdigit()
    )


def patch_configurable(
    config: RunnableConfig | None, patch: dict[str, Any]
) -> RunnableConfig:
    if config is None:
        return {CONF: patch}
    elif CONF not in config:
        return {**config, CONF: patch}
    else:
        return {**config, CONF: {**config[CONF], **patch}}


def patch_checkpoint_map(
    config: RunnableConfig | None, metadata: CheckpointMetadata | None
) -> RunnableConfig:
    if config is None:
        return config
    elif parents := (metadata.get("parents") if metadata else None):
        conf = config[CONF]
        return patch_configurable(
            config,
            {
                CONFIG_KEY_CHECKPOINT_MAP: {
                    **parents,
                    conf[CONFIG_KEY_CHECKPOINT_NS]: conf[CONFIG_KEY_CHECKPOINT_ID],
                },
            },
        )
    else:
        return config


def merge_configs(*configs: RunnableConfig | None) -> RunnableConfig:
    """Merge multiple configs into one.

    Args:
        *configs: The configs to merge.

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
            elif key == CONF:
                if base_value := base.get(key):
                    base[key] = {**base_value, **value}  # type: ignore[dict-item]
                else:
                    base[key] = value
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
    if CONF not in base:
        base[CONF] = {}
    return base


def patch_config(
    config: RunnableConfig | None,
    *,
    callbacks: Callbacks = None,
    recursion_limit: int | None = None,
    max_concurrency: int | None = None,
    run_name: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> RunnableConfig:
    """Patch a config with new values.

    Args:
        config: The config to patch.
        callbacks: The callbacks to set.
        recursion_limit: The recursion limit to set.
        max_concurrency: The max number of concurrent steps to run, which also applies to parallelized steps.
        run_name: The run name to set.
        configurable: The configurable to set.

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
        config[CONF] = {**config.get(CONF, {}), **configurable}
    return config


def get_callback_manager_for_config(
    config: RunnableConfig, tags: Sequence[str] | None = None
) -> CallbackManager:
    """Get a callback manager for a config.

    Args:
        config: The config.

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
    tags: Sequence[str] | None = None,
) -> AsyncCallbackManager:
    """Get an async callback manager for a config.

    Args:
        config: The config.

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
            inheritable_tags=all_tags,
            inheritable_metadata=config.get("metadata"),
        )


def _is_not_empty(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    else:
        return value is not None


def ensure_config(*configs: RunnableConfig | None) -> RunnableConfig:
    """Return a config with all keys, merging any provided configs.

    Args:
        *configs: Configs to merge before ensuring defaults.

    Returns:
        RunnableConfig: The merged and ensured config.
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
                if _is_not_empty(v)
            },
        )
    for config in configs:
        if config is None:
            continue
        for k, v in config.items():
            if _is_not_empty(v) and k in CONFIG_KEYS:
                if k == CONF:
                    empty[k] = cast(dict, v).copy()
                else:
                    empty[k] = v  # type: ignore[literal-required]
        for k, v in config.items():
            if _is_not_empty(v) and k not in CONFIG_KEYS:
                empty[CONF][k] = v
    _empty_metadata = empty["metadata"]
    for key, value in empty[CONF].items():
        if _exclude_as_metadata(key, value, _empty_metadata):
            continue
        _empty_metadata[key] = value
    return empty


_OMIT = ("key", "token", "secret", "password", "auth")


def _exclude_as_metadata(key: str, value: Any, metadata: Mapping[str, Any]) -> bool:
    key_lower = key.casefold()
    return (
        key.startswith("__")
        or not isinstance(value, (str, int, float, bool))
        or key in metadata
        or any(substr in key_lower for substr in _OMIT)
    )
