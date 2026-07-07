import asyncio
import sys

import pytest

from langgraph.config import get_config


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="Python < 3.11 guard only applies on older interpreters",
)
def test_get_config_raises_in_async_context_on_python_lt_311() -> None:
    """Issue #8203: the async guard must not be swallowed on Python < 3.11."""

    async def main() -> None:
        with pytest.raises(
            RuntimeError,
            match="Python 3.11 or later required to use this in an async context",
        ):
            get_config()

    asyncio.run(main())
