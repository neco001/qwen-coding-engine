import asyncio
import json
import logging
import os
import re
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
from qwen_mcp.prompts.swarm import DECOMPOSE_SYSTEM_PROMPT, SYNTHESIZE_SYSTEM_PROMPT
from qwen_mcp.io_utils import resolve_context_keys

logger = logging.getLogger("qwen_mcp.swarm")


class SubTask(BaseModel):
    id: str
    task: str
    priority: int = 1
    context_keys: List[str] = Field(default_factory=list)

    @field_validator('priority', mode='after')
    @classmethod
    def validate_priority(cls, v):
        if v < 0:
            raise ValueError('Priority must be greater than or equal to 0')
        return v


class SwarmResult(BaseModel):
    intent_validation: bool
    sub_tasks: List[SubTask]
    estimated_total_tokens: Optional[int] = None


class SwarmOrchestrator:
    def __init__(self, completion_handler, max_concurrent_tasks: int = 5):
        self.completion_handler = completion_handler
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def execute_subtasks(self, sub_tasks: List[SubTask]) -> Dict[str, str]:
        """
        Executes multiple sub-tasks in parallel, respecting the concurrency limit.
        Handles individual task failures by returning the error message instead of crashing.
        """
        if not sub_tasks:
            return {}

        tasks = [self._execute_single_task(st) for st in sub_tasks]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for st, res in zip(sub_tasks, responses):
            if isinstance(res, Exception):
                results[st.id] = f"ERROR: {str(res)}"
            else:
                results[st.id] = res
                
        return results

    async def _execute_single_task(self, sub_task: SubTask) -> str:
        """
        Executes a single sub-task with semaphore protection.
        
        Now resolves context_keys to actual file contents before execution.
        This fixes the root cause of template code generation - agents now
        receive actual file content instead of just task descriptions.
        """
        async with self._semaphore:
            # Build the prompt
            prompt_content = sub_task.task
            
            # RESOLVE CONTEXT KEYS - THIS IS THE FIX!
            if sub_task.context_keys:
                logger.info(f"Resolving context_keys for task {sub_task.id}: {sub_task.context_keys}")
                
                context_contents = await resolve_context_keys(
                    sub_task.context_keys,
                    project_root=os.getcwd()
                )
                
                # Inject file contents into prompt
                if context_contents:
                    prompt_content += "\n\n### CONTEXT FILES:\n\n"
                    for file_path, content in context_contents.items():
                        prompt_content += f"--- {file_path} ---\n"
                        prompt_content += f"```\n{content}\n```\n\n"
            
            messages = [{"role": "user", "content": prompt_content}]
            result = await self.completion_handler.generate_completion(messages)
            return result

    async def decompose(self, prompt: str) -> SwarmResult:
        """
        Decomposes a complex user prompt into atomic sub-tasks using the model.
        """
        messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        response_text = await self.completion_handler.generate_completion(messages)
        
        # Robust JSON extraction using regex
        clean_json = response_text.strip()
        
        # FIX: Check for empty/whitespace response before parsing
        if not clean_json:
            logger.error("Swarm decomposition returned empty response")
            # Fallback: return single task with original prompt
            return SwarmResult(
                intent_validation=False,
                sub_tasks=[SubTask(id="1", task=prompt, priority=5)],
                estimated_total_tokens=100
            )
        
        json_match = re.search(r'```json\s*(.*?)\s*```', clean_json, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1).strip()
        else:
            code_block_match = re.search(r'```(.*?)```', clean_json, re.DOTALL)
            if code_block_match:
                clean_json = code_block_match.group(1).strip()

        # FIX: Check again after extraction
        if not clean_json:
            logger.error("Swarm decomposition: JSON extraction yielded empty content")
            return SwarmResult(
                intent_validation=False,
                sub_tasks=[SubTask(id="1", task=prompt, priority=5)],
                estimated_total_tokens=100
            )

        try:
            parsed_data = json.loads(clean_json)
            return SwarmResult(**parsed_data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Swarm decomposition JSON parse error: {e}. Raw response: {response_text[:200]}")
            # FIX: Graceful fallback instead of crash
            return SwarmResult(
                intent_validation=False,
                sub_tasks=[SubTask(id="1", task=prompt, priority=5)],
                estimated_total_tokens=100
            )

    async def synthesize(self, original_prompt: str, subtask_results: Dict[str, str]) -> str:
        """
        Synthesizes the final response by combining the original prompt and subtask results.
        """
        if not subtask_results:
            logger.warning("Synthesis called with empty subtask results")
            return "No subtask results were generated to synthesize."

        # Format the synthesis input
        synthesis_input = f"ORIGINAL USER REQUEST: {original_prompt}\n\n"
        synthesis_input += "COLLECTED SUB-TASK RESULTS:\n"
        
        valid_results = 0
        for task_id, result in subtask_results.items():
            if not result or result.strip().startswith("ERROR:"):
                continue
            synthesis_input += f"--- Result for task {task_id} ---\n{result}\n\n"
            valid_results += 1
            
        if valid_results == 0:
            return "Aggregated results were empty or contained only errors."

        logger.info(f"Synthesizing results from {valid_results} valid sub-tasks")
        
        messages = [
            {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_input}
        ]
        final_response = await self.completion_handler.generate_completion(messages)
        
        return final_response

    async def run_swarm(self, prompt: str, task_type: str = "general") -> str:
        """
        High-level entry point: Decompose -> Parallel Execute -> Synthesize.
        """
        logger.info(f"Starting Swarm for prompt: {prompt[:50]}... (Task Type: {task_type})")
        
        # 1. Decompose
        swarm_plan = await self.decompose(prompt)
        
        # 🧪 Force Decomposition for Coding if needed
        is_coding = task_type and task_type.startswith("coding")
        if is_coding and (not swarm_plan.intent_validation or not swarm_plan.sub_tasks):
            logger.info(f"Forcing {task_type} decomposition into phases...")
            from .orchestrator import SubTask
            swarm_plan.intent_validation = True
            swarm_plan.sub_tasks = [
                SubTask(id="T1", task=f"Analyze current state for: {prompt}"),
                SubTask(id="T2", task=f"Plan changes for: {prompt}"),
                SubTask(id="T3", task=f"Implement changes for: {prompt}")
            ]

        if not swarm_plan.intent_validation or not swarm_plan.sub_tasks:
            logger.info("Swarm decomposition skipped: intent invalid or no sub-tasks.")
            # Fallback to normal completion if no decomposition
            # Wrap prompt as message for consistency
            return await self.completion_handler.generate_completion([{"role": "user", "content": prompt}], task_type=task_type)

        # 2. Execute Parallel
        logger.info(f"Executing swarm with {len(swarm_plan.sub_tasks)} agents")
        results = await self.execute_subtasks(swarm_plan.sub_tasks)
        
        # 3. Synthesize
        final_answer = await self.synthesize(prompt, results)
        return final_answer
