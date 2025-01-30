import asyncio
import sys

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config

from langgraph.constants import CONF, CONFIG_KEY_STORE, CONFIG_KEY_STREAM_WRITER
from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter


def get_config() -> RunnableConfig:
    if sys.version_info < (3, 11):
        try:
            if asyncio.current_task():
                raise RuntimeError(
                    "Python 3.11 or later required to use this in an async context"
                )
        except RuntimeError:
            pass
    if var_config := var_child_runnable_config.get():
        return var_config
    else:
        raise RuntimeError("Called get_config outside of a runnable context")


def get_store() -> BaseStore:
    config = get_config()
    return config[CONF][CONFIG_KEY_STORE]


def get_stream_writer() -> StreamWriter:
    config = get_config()
    return config[CONF][CONFIG_KEY_STREAM_WRITER]
