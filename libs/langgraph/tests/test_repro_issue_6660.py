import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt

pytestmark = pytest.mark.anyio

async def test_async_functional_api_interrupt_consumption(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """
    Regression test for issue #6660.
    
    Ensures that async functional API tasks do not double-consume interrupt resume values.
    Each iteration in the loop should trigger exactly one interrupt and require one resume.
    """
    
    @task
    async def dummy(x: int) -> int:
        interrupt("hello")
        return x

    @entrypoint(checkpointer=async_checkpointer)
    async def workflow(count: int) -> list[int]:
        results = []
        for i in range(1, count + 1):
            val = await dummy(i)
            results.append(val)
        return results

    config = {"configurable": {"thread_id": "1"}}
    
    # Run slightly more than the minimal case to be sure (5 iterations)
    target_count = 5
    
    # 1. Initial call - should run until first interrupt (i=1)
    # The result when interrupted is a dict containing "__interrupt__"
    result = await workflow.ainvoke(target_count, config)
    
    resumes_count = 0
    max_resumes = 10  # Safety break
    
    # Loop while we are in an interrupted state
    while isinstance(result, dict) and "__interrupt__" in result:
        resumes_count += 1
        if resumes_count > max_resumes:
            pytest.fail(f"Exceeded max resumes ({max_resumes}). Infinite loop?")
            
        # Resume execution
        # We expect this resume to be consumed by EXACTLY one task instance
        result = await workflow.ainvoke(Command(resume="ignored"), config)
    
    # Verification:
    # 1. We should have resumed exactly 'target_count' times (once for each number 1..5)
    #    Buggy behavior was 3 resumes (1, 3, 5) or infinite loops depending on the bug variants.
    assert resumes_count == target_count, f"Expected {target_count} resumes, but performed {resumes_count}"
    
    # 2. Final result should be the list of all numbers
    assert result == [1, 2, 3, 4, 5]
