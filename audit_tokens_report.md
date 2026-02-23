Oczekuję na Audyt od Qwena...

--- AUDIT QWEN ---

# 🔍 Senior DevOps & SRE Audit Report
**Target:** `client.py` (Token Tracking) & `tools.py` (Cost Estimation Logic)
**Context:** DashScope (Qwen) API via OpenAI Compatible Client
**Severity:** ⚠️ **Medium-High** (Financial Risk & Data Integrity)

---

## 1. Root Cause Analysis: Mathematical & Logic Flaws

### A. The "Context Window" Pricing Fallacy
**Location:** `tools.py` line ~235
```python
cost = (p_tok / 1000000.0) * (7.0 if p_tok > 240000 else 0.8) + ...
```
**Critical Issue:** You are applying the **higher tier price to the entire input volume** once it crosses the threshold, rather than just the overflow.
*   **Current Logic:** If `p_tok` = 240,001, the cost is calculated as `240,001 * $7.0`.
*   **Reality (DashScope/Qwen):** Pricing is usually tiered or based on the specific model variant selected for that context size. Even if it were a hard cutoff, most APIs charge the base rate for the first 240k and the premium rate only for the excess.
*   **Impact:** **Massive Overestimation.** A request of 241k tokens would be reported as costing ~$1.68, whereas the real cost is likely closer to ~$0.20 (base) + negligible overflow. This destroys trust in your cost reporting tool.

### B. Model-Specific Pricing Hardcoding
**Location:** `tools.py`
**Issue:** The logic assumes a single pricing model (`$0.8/$7.0` input, `$2.0` output).
*   Your `ModelRegistry` dynamically switches between `qwen-turbo`, `qwen-plus`, `qwen-max`, and `qwen-coder`.
*   **Fact:** `qwen-turbo` is significantly cheaper than `qwen-plus`. `qwen-max` is more expensive.
*   **Result:** The cost calculation is **incorrect for 75% of your requests** because it doesn't know which model actually processed the tokens.

### C. Token Accumulation Race Condition
**Location:** `client.py` `generate_completion` (Streaming mode)
```python
if hasattr(chunk, "usage") and chunk.usage:
    self.session_prompt_tokens += getattr(chunk.usage, "prompt_tokens", 0)
    # ...
```
**Issue:** In the OpenAI compatible stream protocol (and DashScope's implementation), the `usage` object typically appears **only in the final chunk**.
*   If the connection drops before the final chunk (timeout, network blip), `session_prompt_tokens` remains **0** or partially updated, while the user was charged by Alibaba.
*   **Edge Case:** If `generate_completion` is called concurrently (which `AsyncOpenAI` supports), `self.session_prompt_tokens` is **not thread-safe**. Two simultaneous requests will corrupt the global counter.

---

## 2. Architecture & Design Patterns

### A. Global Mutable State (Anti-Pattern)
**Location:** `DashScopeClient` class attributes `session_prompt_tokens`
**Critique:** Storing session state on the *client instance* which appears to be instantiated per tool call (in `tools.py`) is fragile.
*   In `tools.py`, you do `client = DashScopeClient()`. This creates a *new* instance every time.
*   **Result:** `session_prompt_tokens` resets to 0 on every tool invocation. The accumulation logic inside `generate_completion` only works for *single* long-running streams within that specific function call. It fails to track costs across a multi-step "Lachman Protocol" session unless the `client` object is persisted globally (which it isn't in the provided snippet).

### B. Tight Coupling of Business Logic
**Location:** `tools.py` inside `generate_lp_blueprint`
**Critique:** The cost calculation logic is duplicated and hardcoded inside the blueprint generator.
*   **Recommendation:** Move pricing logic to a dedicated `PricingEngine` class that accepts `model_name` and `token_counts` as arguments. This adheres to the Single Responsibility Principle.

### C. Retry Logic vs. Token Counting
**Location:** `client.py` `@retry` decorator
**Issue:** The `tenacity` retry decorator retries the *entire* function.
*   If a request fails after streaming 50% of the content, the retry starts from scratch.
*   However, if the failure happens *after* the API has processed the tokens but before the client receives the response (rare but possible with gateways), you might be charged for the failed attempt, but your local counter won't reflect it because the `usage` block never arrived.

---

## 3. Performance & Optimization

### A. Streaming Overhead
**Observation:** You use `stream_options={"include_usage": True}`.
**Analysis:** This is correct for getting usage data in streaming mode. However, be aware that some providers delay the final chunk slightly to aggregate usage stats.
**Optimization:** Ensure your `progress_callback` handles the final empty chunk gracefully. Currently, your loop adds `delta` only if present, which is good. But ensure `full_response` is returned even if the last chunk only contains `usage`.

### B. Regex Compilation
**Location:** `SecuritySanitizer`
**Optimization:** You compile regex patterns on every call implicitly via `re.finditer`.
**Fix:** Pre-compile patterns in `__init_subclass__` or as module-level constants using `re.compile()`. This saves CPU cycles on every message sanitization.

---

## 4. Security Vulnerabilities

### A. Path Traversal (High Severity)
**Location:** `tools.py` `read_repo_file`
```python
path = Path(file_path)
if not path.exists(): ...
content = path.read_text(...)
```
**Vulnerability:** There is **no validation** that `file_path` stays within the project root.
**Exploit:** An attacker (or hallucinating LLM) could request `../../etc/passwd` or `../../.env`.
**Fix:** Resolve the absolute path and ensure it starts with the allowed workspace directory.
```python
import os
ALLOWED_ROOT = os.getcwd() # Or specific config
abs_path = Path(file_path).resolve()
if not str(abs_path).startswith(ALLOWED_ROOT):
    raise ValueError("Path traversal detected")
```

### B. Secret Leakage via Logs
**Location:** `client.py` `logger.info`
**Risk:** Although `SecuritySanitizer` redacts content *before* sending to the LLM, the logs inside `DashScopeClient` (e.g., `logger.info(f"Requesting completion from {target_model}...")`) are safe, BUT if you ever log the `messages` list for debugging *before* sanitization, secrets leak.
**Current Status:** You sanitize `msg["content"]` in place. This is good, but ensure no other logging statements capture the raw `messages` variable.

### C. Environment Variable Exposure
**Location:** `client.py`
**Risk:** `OLLAMA_BASE_URL` logic falls back to `api_key = "ollama"`. If this client is accidentally exposed to the internet without auth, and Ollama is running locally with no auth, it might be accessible. (Low risk in local dev, high risk if containerized incorrectly).

---

## 5. Edge Cases & Potential Bugs

### A. The "Zero Usage" Bug
**Scenario:** Network timeout occurs exactly after headers are received but before the body/stream starts.
**Result:** `response.usage` is never accessed. `session_prompt_tokens` remains 0. The error handler returns a string, but the financial cost is lost.
**Fix:** Wrap the token extraction in a `finally` block or ensure the error handler attempts to parse any partial usage data if available (though often impossible in streaming).

### B. JSON Parsing Fragility
**Location:** `tools.py` `extract_json_from_text`
**Risk:** LLMs sometimes output markdown with extra spaces or comments inside JSON (invalid JSON). Your regex `r"(\{.*\})"` is greedy and might fail if the LLM outputs multiple JSON blocks or explanatory text *inside* the braces (unlikely but possible with complex reasoning).
**Improvement:** Use a library like `json_repair` which can fix minor syntax errors in LLM output, rather than raw `json.loads`.

### C. Currency Hardcoding
**Risk:** The cost string says `USD`. If your Alibaba account is billed in CNY or another currency, the estimation is misleading.
**Fix:** Make currency configurable via Env Var.

---

## 🛠 Actionable Remediation Plan

### Step 1: Fix Cost Calculation Logic (Immediate)
Refactor `tools.py` to use a mapping of model prices and correct tier logic.

```python
# tools.py - New Helper Class
class PricingEngine:
    # Prices per 1M tokens (Example values - VERIFY against current DashScope docs)
    PRICES = {
        "qwen-turbo": {"input": 0.2, "output": 0.6},
        "qwen-plus": {"input": 0.8, "output": 2.0},
        "qwen-max": {"input": 2.5, "output": 7.0}, # Example
        "default": {"input": 1.0, "output": 3.0}
    }

    @staticmethod
    def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Normalize model name (handle versions like qwen-plus-0912)
        base_model = "default"
        for key in PricingEngine.PRICES:
            if key in model_name:
                base_model = key
                break
        
        rates = PricingEngine.PRICES.get(base_model, PricingEngine.PRICES["default"])
        
        # Correct Math: No sudden jumps, linear scaling based on model
        # If tiered pricing exists, implement logic here. 
        # Assuming flat rate per model for now as per standard API practices.
        input_cost = (prompt_tokens / 1_000_000) * rates["input"]
        output_cost = (completion_tokens / 1_000_000) * rates["output"]
        
        return input_cost + output_cost
```

### Step 2: Secure File Reading
Update `read_repo_file` in `tools.py`:

```python
async def read_repo_file(file_path: str) -> str:
    import os
    from pathlib import Path
    
    # Define safe root (e.g., current working dir or specific repo path)
    safe_root = Path(os.getenv("WORKSPACE_ROOT", ".")).resolve()
    target_path = (safe_root / file_path).resolve()
    
    # Security Check
    try:
        target_path.relative_to(safe_root)
    except ValueError:
        return "Error: Access denied. Path outside workspace."

    if not target_path.exists():
        return f"Error: File '{file_path}' not found."
    
    # ... rest of logic
```

### Step 3: Thread-Safe Token Tracking
If you intend to track costs across multiple tool calls in a session, you cannot rely on a local `client` instance variable.
*   **Option A:** Pass a shared `SessionContext` object containing the token counters to every tool.
*   **Option B:** Use a global singleton `TokenBucket` with a lock (`asyncio.Lock`) if running in a single-process environment.

### Step 4: Robust Error Handling for Usage
In `client.py`, ensure that if an exception occurs, you log that usage data might be missing:

```python
except Exception as e:
    logger.warning(f"Request failed. Token usage for this request may be unrecorded due to early termination: {e}")
    return await self._handle_error(e, request_timeout)
```

### Final Verdict
The current implementation is a **functional prototype** but **unsafe for production financial tracking**. The pricing math is fundamentally flawed (tier jump), and the security posture on file reading is critical. Implement the `PricingEngine` and Path Traversal fixes immediately before exposing this to any untrusted inputs or relying on the cost estimates for budget decisions.
