"""Timeout handling for Pregel execution."""
from typing import TypeVar, Optional, Any, Coroutine
import asyncio
from langgraph.errors import ParentCommand
from langgraph.types import Command

T = TypeVar("T")

async def with_timeout(
    timeout: Optional[float],
    coro: Coroutine[Any, Any, T],
    *,
    handle_parent_command: bool = True
) -> T:
    """Execute a coroutine with a timeout, properly handling parent commands.
    
    Args:
        timeout: The timeout in seconds, or None for no timeout
        coro: The coroutine to execute
        handle_parent_command: Whether to handle parent commands specially
        
    Returns:
        The result of the coroutine
        
    Raises:
        asyncio.TimeoutError: If the timeout is reached
        ParentCommand: If the coroutine raises a ParentCommand and handle_parent_command is True
    """
    if timeout is None:
        return await coro
        
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if handle_parent_command:
            # Check if there was a parent command being issued
            try:
                result = await coro
                if isinstance(result, Command):
                    raise ParentCommand(result)
            except Exception:
                pass
        raise 