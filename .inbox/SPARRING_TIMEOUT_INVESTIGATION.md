# Sparring Timeout Investigation - Session Context

## Problem Statement

`qwen_sparring(mode="flash")` consistently times out with MCP error -32001 after 60 seconds, while `qwen_coder` works fine with the same infrastructure.

## Key Observations

1. **qwen_coder works**: Single API call completes successfully in ~30 seconds
2. **qwen_sparring flash fails**: Makes 2 sequential API calls, times out at 60s
3. **Direct Python test also times out**: Running sparring directly via `mcp--py-executor--run_python` with 120s timeout still fails at 60s
4. **Server doesn't crash**: Error is -32001 (timeout), not -32000 (connection closed)

## Architecture

### Sparring Flash Mode Flow
```
_execute_flash() → 
  Step 1: Analyst (task_type="audit" → glm-5) → 
  Step 2: Drafter (task_type="strategist" → qwen3.5-plus)
```

### Key Files Modified

1. **`src/qwen_mcp/completions.py`**:
   - Added `progress_callback` parameter to `_stream_completion()`
   - Added progress reporting every 10 chunks to keep MCP connection alive
   - Removed `asyncio.timeout()` wrapper (caused crash)

2. **`src/qwen_mcp/engines/sparring_v2/engine.py`**:
   - Added `progress_callback=ctx.report_progress if ctx else None` to both API calls
   - Changed complexity from "high"/"critical" to "medium"
   - Changed `include_reasoning=False` for drafter

## Root Cause Hypotheses (Unresolved)

### Hypothesis 1: MCP Tool Timeout vs API Timeout
- MCP has a **hard 60-second tool timeout** that cannot be extended
- The OpenAI client timeout (20s per call) only applies to individual requests
- Two sequential 20s calls + overhead = potentially >60s total

### Hypothesis 2: Streaming Not Progressing
- The `progress_callback` is supposed to keep MCP connection alive
- But it's only called every 10 chunks during streaming
- If streaming doesn't start or is very slow, no progress is reported

### Hypothesis 3: Model Routing Issue
- `task_type="audit"` routes to `glm-5` (deep thinking model)
- GLM-5 has thinking mode enabled by default
- Thinking mode can take significant time before producing output

## Tests Performed

```python
# Test 1: qwen_coder - WORKS
client.generate_completion(task_type="coder", complexity="low")  # ~30s, success

# Test 2: sparring flash via MCP tool - FAILS
qwen_sparring(mode="flash", topic="...")  # 60s timeout

# Test 3: sparring flash via direct Python - FAILS
SparringEngineV2().execute(mode="flash")  # 60s timeout (even with 120s limit)
```

## Critical Question

Why does the direct Python test (via `mcp--py-executor--run_python`) also timeout at 60s when we set `timeout=120`? This suggests the timeout is NOT coming from the MCP tool layer, but from somewhere else in the stack.

## Next Steps to Investigate

1. **Check if API calls are actually starting**: Add logging at the start of each `generate_completion` call
2. **Check model routing**: Verify what model `task_type="audit"` actually routes to
3. **Test single analyst call**: Isolate just the first API call to see if it completes
4. **Check if streaming is working**: Verify that chunks are being received
5. **Check OpenAI client timeout behavior**: The `timeout` parameter might not work as expected with streaming

## Code Locations

- Sparring engine: [`src/qwen_mcp/engines/sparring_v2/engine.py`](src/qwen_mcp/engines/sparring_v2/engine.py:126) (`_execute_flash`)
- Streaming completion: [`src/qwen_mcp/completions.py`](src/qwen_mcp/completions.py:155) (`_stream_completion`)
- Model routing: [`src/qwen_mcp/registry.py`](src/qwen_mcp/registry.py:473) (`audit` → `analyst` → `glm-5`)
- Timeout config: [`src/qwen_mcp/engines/sparring_v2/config.py`](src/qwen_mcp/engines/sparring_v2/config.py:11) (`TIMEOUTS`)

## Related Fixes (Previous Session)

1. Fixed `thinking_buffer` initialization bug (int → string)
2. Fixed `broadcast_state()` to sync all project states
3. Fixed potential deadlock in `broadcast_state()` with lock
4. Added fire-and-forget pattern in `update_stream()`