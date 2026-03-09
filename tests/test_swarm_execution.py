import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from qwen_mcp.orchestrator import SwarmOrchestrator, SubTask

@pytest.mark.asyncio
async def test_execute_subtasks_parallel_execution():
    # Mock completion_handler
    mock_handler = AsyncMock()
    
    async def mock_gen(messages, **kwargs):
        prompt = messages[0]["content"]
        await asyncio.sleep(0.5)  # Simulate network/inference delay
        return f"Result for: {prompt[:10]}"

    mock_handler.generate_completion.side_effect = mock_gen
    
    orchestrator = SwarmOrchestrator(completion_handler=mock_handler, max_concurrent_tasks=5)
    
    subtasks = [
        SubTask(id="1", task="Task 1 description"),
        SubTask(id="2", task="Task 2 description"),
        SubTask(id="3", task="Task 3 description")
    ]
    
    start_time = time.time()
    results = await orchestrator.execute_subtasks(subtasks)
    duration = time.time() - start_time
    
    assert len(results) == 3
    assert results["1"] == "Result for: Task 1 des"
    assert results["2"] == "Result for: Task 2 des"
    assert results["3"] == "Result for: Task 3 des"
    
    # Parallel check: 3 tasks * 0.5s = 1.5s if sequential, should be ~0.5s if parallel
    assert duration < 0.8  # Allowing some overhead

@pytest.mark.asyncio
async def test_execute_subtasks_semaphore_limit():
    mock_handler = AsyncMock()
    
    active_count = 0
    max_active = 0
    
    async def mock_gen(messages, **kwargs):
        nonlocal active_count, max_active
        active_count += 1
        max_active = max(max_active, active_count)
        await asyncio.sleep(0.1)
        active_count -= 1
        return "done"

    mock_handler.generate_completion.side_effect = mock_gen
    
    # Limit to 2 concurrent tasks
    orchestrator = SwarmOrchestrator(completion_handler=mock_handler, max_concurrent_tasks=2)
    
    subtasks = [SubTask(id=str(i), task=f"Task {i}") for i in range(5)]
    
    await orchestrator.execute_subtasks(subtasks)
    
    assert max_active == 2  # Semaphore should have limited parallelism to 2
