from _typeshed import Incomplete
from langgraph.types import PregelExecutableTask as PregelExecutableTask, RetryPolicy as RetryPolicy
from typing import Any

logger: Incomplete
SUPPORTS_EXC_NOTES: Incomplete

def run_with_retry(task: PregelExecutableTask, retry_policy: RetryPolicy | None, configurable: dict[str, Any] | None = None) -> None: ...
async def arun_with_retry(task: PregelExecutableTask, retry_policy: RetryPolicy | None, stream: bool = False, configurable: dict[str, Any] | None = None) -> None: ...
