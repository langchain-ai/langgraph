import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from typing import Union

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import Checkpoint
from langgraph.constants import MISSING
from langgraph.managed.base import (
    ConfiguredManagedValue,
    ManagedValueMapping,
    ManagedValueSpec,
)
from langgraph.managed.context import Context
from langgraph.types import LoopProtocol


@contextmanager
def ChannelsManager(
    specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]],
    checkpoint: Checkpoint,
    loop: LoopProtocol,
    *,
    skip_context: bool = False,
) -> Iterator[tuple[Mapping[str, BaseChannel], ManagedValueMapping]]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    channel_specs: dict[str, BaseChannel] = {}
    managed_specs: dict[str, ManagedValueSpec] = {}
    for k, v in specs.items():
        if isinstance(v, BaseChannel):
            channel_specs[k] = v
        elif (
            skip_context and isinstance(v, ConfiguredManagedValue) and v.cls is Context
        ):
            managed_specs[k] = Context.of(noop_context)
        else:
            managed_specs[k] = v
    with ExitStack() as stack:
        yield (
            {
                k: v.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
                for k, v in channel_specs.items()
            },
            ManagedValueMapping(
                {
                    key: stack.enter_context(
                        value.cls.enter(loop, **value.kwargs)
                        if isinstance(value, ConfiguredManagedValue)
                        else value.enter(loop)
                    )
                    for key, value in managed_specs.items()
                }
            ),
        )


@asynccontextmanager
async def AsyncChannelsManager(
    specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]],
    checkpoint: Checkpoint,
    loop: LoopProtocol,
    *,
    skip_context: bool = False,
) -> AsyncIterator[tuple[Mapping[str, BaseChannel], ManagedValueMapping]]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    channel_specs: dict[str, BaseChannel] = {}
    managed_specs: dict[str, ManagedValueSpec] = {}
    for k, v in specs.items():
        if isinstance(v, BaseChannel):
            channel_specs[k] = v
        elif (
            skip_context and isinstance(v, ConfiguredManagedValue) and v.cls is Context
        ):
            managed_specs[k] = Context.of(noop_context)
        else:
            managed_specs[k] = v
    async with AsyncExitStack() as stack:
        # managed: create enter tasks with reference to spec, await them
        if tasks := {
            asyncio.create_task(
                stack.enter_async_context(
                    value.cls.aenter(loop, **value.kwargs)
                    if isinstance(value, ConfiguredManagedValue)
                    else value.aenter(loop)
                )
            ): key
            for key, value in managed_specs.items()
        }:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        else:
            done = set()
        yield (
            # channels: enter each channel with checkpoint
            {
                k: v.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
                for k, v in channel_specs.items()
            },
            # managed: build mapping from spec to result
            ManagedValueMapping({tasks[task]: task.result() for task in done}),
        )


@contextmanager
def noop_context() -> Iterator[None]:
    yield None
