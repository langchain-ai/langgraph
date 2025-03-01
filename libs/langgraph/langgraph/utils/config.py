import uuid
from collections import ChainMap
from contextvars import ContextVar
from typing import Any, Optional, Sequence, Union, cast

from langsmith.run_trees import RunTree, get_cached_client
from typing_extensions import TypedDict

from langgraph.checkpoint.base import CheckpointConfig, CheckpointMetadata


class RunnableConfig(TypedDict, total=False):
    """Configuration for a Runnable."""

    tags: list[str]
    """
    Tags for this call and any sub-calls (eg. a Chain calling an LLM).
    You can use these to filter calls.
    """

    metadata: dict[str, Any]
    """
    Metadata for this call and any sub-calls (eg. a Chain calling an LLM).
    Keys should be strings, values should be JSON-serializable.
    """

    run_name: str
    """
    Name for the tracer run for this call. Defaults to the name of the class.
    """

    max_concurrency: Optional[int]
    """
    Maximum number of parallel calls to make. If not provided, defaults to
    ThreadPoolExecutor's default.
    """

    recursion_limit: int
    """
    Maximum number of times a call can recurse. If not provided, defaults to 25.
    """

    configurable: dict[str, Any]
    """
    Runtime values for attributes previously made configurable on this Runnable,
    or sub-Runnables, through .configurable_fields() or .configurable_alternatives().
    Check .output_schema() for a description of the attributes that have been made
    configurable.
    """

    run_id: Optional[uuid.UUID]
    """
    Unique identifier for the tracer run for this call. If not provided, a new UUID
        will be generated.
    """

    run_tree: RunTree
    """
    The trace tree of the caller.
    """


AnyConfig = Union[RunnableConfig, CheckpointConfig]

CONFIG_KEYS = [
    "tags",
    "metadata",
    "run_tree",
    "run_name",
    "max_concurrency",
    "recursion_limit",
    "configurable",
    "run_id",
]

COPIABLE_KEYS = [
    "tags",
    "metadata",
    "run_tree",
    "configurable",
]

DEFAULT_RECURSION_LIMIT = 25

var_child_runnable_config = ContextVar(
    "child_runnable_config", default=RunnableConfig()
)


def set_config_in_context(context: AnyConfig) -> None:
    """Set the context for the current thread."""
    var_child_runnable_config.set(context)


def recast_checkpoint_ns(ns: str) -> str:
    """Remove task IDs from checkpoint namespace.

    Args:
        ns (str): The checkpoint namespace with task IDs.

    Returns:
        str: The checkpoint namespace without task IDs.
    """
    from langgraph.constants import (
        NS_END,
        NS_SEP,
    )

    return NS_SEP.join(
        part.split(NS_END)[0] for part in ns.split(NS_SEP) if not part.isdigit()
    )


def patch_configurable(
    config: Optional[AnyConfig], patch: dict[str, Any]
) -> RunnableConfig:
    from langgraph.constants import (
        CONF,
    )

    if config is None:
        return {CONF: patch}
    elif CONF not in config:
        return {**config, CONF: patch}
    else:
        return {**config, CONF: {**config[CONF], **patch}}


def patch_checkpoint_map(
    config: Optional[AnyConfig],
    metadata: Optional[CheckpointMetadata],
) -> RunnableConfig:
    from langgraph.constants import (
        CONF,
        CONFIG_KEY_CHECKPOINT_ID,
        CONFIG_KEY_CHECKPOINT_MAP,
        CONFIG_KEY_CHECKPOINT_NS,
    )

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


def merge_configs(*configs: Optional[AnyConfig]) -> RunnableConfig:
    """Merge multiple configs into one.

    Args:
        *configs (Optional[RunnableConfig]): The configs to merge.

    Returns:
        RunnableConfig: The merged config.
    """
    from langgraph.constants import (
        CONF,
    )

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
            elif key == "recursion_limit":
                if config["recursion_limit"] != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = config["recursion_limit"]
            else:
                base[key] = config[key]  # type: ignore[literal-required]
    if CONF not in base:
        base[CONF] = {}
    return base


def patch_config(
    config: Optional[AnyConfig],
    *,
    runtree: Optional[RunTree] = None,
    recursion_limit: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    run_name: Optional[str] = None,
    configurable: Optional[dict[str, Any]] = None,
) -> RunnableConfig:
    """Patch a config with new values.

    Args:
        config (Optional[RunnableConfig]): The config to patch.
        runtree (Optional[RunTree], optional): The runtree to set.
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
    from langgraph.constants import (
        CONF,
    )

    config = config.copy() if config is not None else {}
    if runtree is not None:
        # If we're replacing callbacks, we need to unset run_name
        # As that should apply only to the same run as the original callbacks
        config["run_tree"] = runtree
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


def get_runtree_for_config(
    config: AnyConfig,
    inputs: Any,
    *,
    name: str,
    tags: Optional[Sequence[str]] = None,
) -> RunTree:
    """Get a runtree for a config.

    Args:
        config (RunnableConfig): The config.

    Returns:
        RunTree: The runtree.
    """

    # merge tags
    all_tags = config.get("tags")
    if all_tags is not None and tags is not None:
        all_tags = [*all_tags, *tags]
    elif tags is not None:
        all_tags = list(tags)
    # use existing callbacks if they exist
    if (runtree := config.get("run_tree")) and isinstance(runtree, RunTree):
        # TODO why is this needed?
        if not hasattr(runtree, "ls_client"):
            runtree.ls_client = get_cached_client()
        return runtree.create_child(
            inputs=inputs if isinstance(inputs, dict) else {"input": inputs},
            tags=all_tags,
            extra={"metadata": config.get("metadata")},
            name=name,
            run_id=config.get("run_id", uuid.uuid4()),
        )
    else:
        # otherwise create a new manager
        return RunTree(
            id=config.get("run_id", uuid.uuid4()),
            name=name,
            extra={"metadata": config.get("metadata")},
            tags=all_tags,
            inputs=inputs if isinstance(inputs, dict) else {"input": inputs},
            ls_client=get_cached_client(),
        )


def _is_not_empty(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    else:
        return value is not None


def ensure_config(*configs: Optional[AnyConfig]) -> RunnableConfig:
    """Ensure that a config is a dict with all keys present.

    Args:
        config (Optional[RunnableConfig], optional): The config to ensure.
          Defaults to None.

    Returns:
        RunnableConfig: The ensured config.
    """
    from langgraph.constants import (
        CONF,
    )

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
    for key, value in empty[CONF].items():
        if (
            not key.startswith("__")
            and isinstance(value, (str, int, float, bool))
            and key not in empty["metadata"]
        ):
            empty["metadata"][key] = value
    return empty
