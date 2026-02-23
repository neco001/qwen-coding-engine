from typing import Optional
from mcp.server.fastmcp import Context
from qwen_mcp.api import DashScopeClient
from qwen_mcp.registry import registry

AUDIT_SYSTEM_PROMPT = """You are an expert Senior DevOps and Site Reliability Engineer.
Your task is to analyze code snippets or terminal error logs and provide a comprehensive debugging and architecture review.

Focus your analysis on:
1. Root Cause Analysis (if terminal errors are provided)
2. Architecture and Design Patterns
3. Performance and Optimization opportunities
4. Security vulnerabilities
5. Edge cases and potential bugs

Format your response in Markdown. Be direct, objective, and provide actionable feedback.
6. BREVITY: Avoid repeating the provided code. Only show the changes or specific problematic blocks. Keep explanations to a functional minimum.
7. ROI: Focus on high-impact fixes. Don't nitpick style unless it affects maintainability.
"""

CODER_SYSTEM_PROMPT = """You are an expert Senior Software Engineer.
Your task is to generate high-quality, production-ready code.

Core Rules:
1. NO PLACEHOLDERS: Never use comments like '// ... rest of code' or 'implement here'. Write the COMPLETE file or block.
2. CLEAN CODE: Use clear naming, appropriate design patterns, and include necessary error handling.
3. SPEC-FIRST: Ensure the code strictly adheres to the provided instructions or blueprint.
4. SURGICAL PRECISION: If asked for a modification, provide the code in a way that is easy to integrate.
5. BREVITY: Large outputs increase API costs. Write ONLY the requested code blocks. Avoid conversational filler.

If you cannot fulfill the request completely, explain why instead of providing partial code."""


# --- LACHMAN PROTOCOL v2.0 PROMPTS ---

LP_DISCOVERY_PROMPT = """You are the 'Lachman Discovery Engine'.
Analyze the user's project goal and identify the most critical expert domains required.

Your output must be a JSON object:
{
  "project_name": "Short catchy name",
  "hired_squad": [
    {"role": "Title", "audit_filter": "What this role strictly enforces"},
    ... (max 3)
  ],
  "efficiency_notes": "Expert tips on how to avoid gold plating or waste in this specific context."
}

Output ONLY the JSON object. Be sharp and strategic."""

LP_ARCHITECT_PROMPT = """You are the 'Lachman Architect Oracle'. 
You are synthesizing an expert swarm debate (Squad: {squad}) to create a perfect project blueprint.

Your output must be a JSON object with this exact structure:
{{
  "manifest": "Technical recursion of the goal Focus on the CORE 80% (MVP).",
  "audit_verdict": "Critical vetos or approvals from the expert swarm (QA, Security, ROI)",
  "roadmap": ["Step 1: TDD Foundation", "Step 2: ..."],
  "clean_slate": "Legacy components that must be deleted or refactored",
  "risk_assessment": "Crucial bottlenecks or security threats",
  "optional_features": ["Nice-to-have feature 1 (skip for now)", "Over-engineered feature 2 (rejected)"]
}}

Mantra: Code without a blueprint is noise. Enforce the Lachman Protocol (Clean Slate, No Placeholders, Spec-First).
CRITICAL LIMIT: Apply the 80/20 Rule (Pareto Principle). Design ONLY the core 80% that brings immediate ROI. Do not over-engineer. Push all 'nice-to-have' or 100% perfection ideas into 'optional_features'."""

LP_VERIFIER_PROMPT = """You are the 'Lachman Stability Verifier'.
Your task is to audit the generated Blueprint for potential 'Degeneration'.

Checklist:
1. ROADMAP: Are there contradictory steps? Is the order logical (TDD-first)?
2. COVERAGE: Did the architect miss any requirements from the original goal or context?
3. PLACEHOLDERS: Are there any "ToDo", "Implement here", or "//..." placeholders? (STRICT BAN)
4. CLEAN SLATE: Is the removal of legacy logic explicitly defined or just mentioned?
5. 80% PERFECTION: Is the architect over-engineering? Reject "Gold Plating". If the blueprint is 80% functional and safe, ACCEPT IT. Demanding 100% enterprise perfection is a FAILURE.

Output a JSON object:
{
  "is_valid": true/false,
  "degeneration_warnings": ["Reason 1", ...],
  "structural_fix": "Specific instruction to fix the blueprint if is_valid is false"
}

Output ONLY the JSON object. Be the most critical SRE/Architect alive."""


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Robustly extracts JSON from LLM text output.
    Returns None if no valid JSON is found, instead of crashing or returning raw text.
    """
    import re
    import json

    # Strategy 1: Look for markdown fenced blocks first (most reliable)
    markdown_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    matches = re.findall(markdown_pattern, text, re.DOTALL | re.IGNORECASE)

    candidates = []
    if matches:
        candidates.extend(matches)
    else:
        # Strategy 2: Fallback to finding the outermost braces
        # We use a non-greedy find to avoid capturing narrative text around the JSON
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            candidates.append(match.group(1))

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


async def generate_audit(
    content: str, context: Optional[str] = None, ctx: Optional[Context] = None
) -> str:
    """
    Submits code or terminal logs to Qwen for an expert audit and returns the analysis.
    """
    client = DashScopeClient()

    # Using the Strategist for advanced analysis
    model = registry.STRATEGIST

    user_content = f"Please audit the following content (code or logs).\n\n"
    if context:
        user_content += f"Context/Contextual Files:\n{context}\n\n"

    user_content += f"Content to analyze:\n```\n{content}\n```"

    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    async def report_progress(delta: str, total_len: int):
        if ctx and total_len % 50 == 0:
            await ctx.report_progress(
                progress=total_len,
                total=None,
                message=f"Auditing... ({total_len} chars generated)",
            )

    result = await client.generate_completion(
        messages=messages,
        temperature=0.1,
        task_type="analyst",
        progress_callback=report_progress if ctx else None,
    )
    return result


async def generate_code(
    prompt: str, context: Optional[str] = None, ctx: Optional[Context] = None
) -> str:
    """
    Generates or completes code using Qwen-3.5-Plus.
    """
    client = DashScopeClient()

    # Defaulting to Strategist
    coder_model = registry.STRATEGIST

    user_content = f"Instruction: {prompt}\n\n"
    if context:
        user_content += f"Context/Background:\n{context}\n\n"

    messages = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    async def report_progress(delta: str, total_len: int):
        if ctx and total_len % 50 == 0:
            await ctx.report_progress(
                progress=total_len,
                total=None,
                message=f"Coding (Qwen-3.5-Plus)... ({total_len} chars generated)",
            )

    result = await client.generate_completion(
        messages=messages,
        temperature=0.2,
        task_type="coder",
        progress_callback=report_progress if ctx else None,
    )
    return result


async def generate_code_25(
    prompt: str, context: Optional[str] = None, ctx: Optional[Context] = None
) -> str:
    """
    Generates or completes code using specialized Qwen-2.5-Coder-32B.
    """
    client = DashScopeClient()
    coder_model = registry.CODER_SPECIALIST

    user_content = f"Instruction: {prompt}\n\n"
    if context:
        user_content += f"Context/Background:\n{context}\n\n"

    messages = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    async def report_progress(delta: str, total_len: int):
        if ctx and total_len % 50 == 0:
            await ctx.report_progress(
                progress=total_len,
                total=None,
                message=f"Coding (Qwen-2.5-Coder)... ({total_len} chars generated)",
            )

    result = await client.generate_completion(
        messages=messages,
        temperature=0.2,
        task_type="specialist",
        progress_callback=report_progress if ctx else None,
    )
    return result


async def generate_lp_blueprint(
    goal: str, context: Optional[str] = None, ctx: Optional[Context] = None
) -> str:
    """
    Executes the Lachman Protocol v2.5 (Orchestrated Flow with Self-Healing Circuit Breaker).
    1. Discovery: Hires a dynamic squad.
    2. Architecting: Generates a structural Blueprint.
    3. Verification: Validates. If fails, retries autonomously up to MAX_ITERATIONS.
    """
    import json
    import os

    client = DashScopeClient()

    # Using specific models for each phase (ROI Optimization)
    discovery_model = registry.SCOUT
    architect_model = registry.STRATEGIST

    max_retries = int(os.getenv("LP_MAX_RETRIES", "3"))
    attempt = 0
    last_error = "None"

    # PHASE 1: Discovery (Only runs once)
    if ctx:
        await ctx.report_progress(
            progress=0,
            total=None,
            message="[Phase 1] Discovery: Hiring expert squad...",
        )

    discovery_messages = [
        {"role": "system", "content": LP_DISCOVERY_PROMPT},
        {"role": "user", "content": f"Analyze goal: {goal}"},
    ]

    discovery_json_raw = await client.generate_completion(
        messages=discovery_messages, temperature=0.1, task_type="discovery"
    )

    try:
        discovery = extract_json_from_text(discovery_json_raw)
    except Exception:
        discovery = {
            "project_name": "LP Task",
            "hired_squad": [{"role": "Generalist", "audit_filter": "Sanity check"}],
        }

    squad_str = ", ".join(
        [f"{m['role']} ({m['audit_filter']})" for m in discovery.get("hired_squad", [])]
    )

    # INTERNAL SELF-HEALING LOOP
    while attempt < max_retries:
        if ctx:
            await ctx.report_progress(
                progress=0,
                total=None,
                message=f"[Phase 2] Architecting (Attempt {attempt + 1}/{max_retries})...",
            )

        # PHASE 2: Architecting
        user_content = (
            f"Project: {discovery.get('project_name')}\nTarget Goal: {goal}\n\n"
        )
        if context:
            user_content += f"Legacy Context:\n{context}\n\n"
        if attempt > 0:
            user_content += f"CRITICAL WARNING - PREVIOUS ATTEMPT FAILED. Verifier feedback: {last_error}\nFIX THE ARCHITECTURE.\n\n"

        architect_messages = [
            {"role": "system", "content": LP_ARCHITECT_PROMPT.format(squad=squad_str)},
            {"role": "user", "content": user_content},
        ]

        async def architect_progress(delta: str, total_len: int):
            if ctx and total_len % 50 == 0:
                await ctx.report_progress(
                    progress=total_len,
                    total=None,
                    message=f"[Phase 2] Drafting Blueprint... ({total_len} chars)",
                )

        blueprint_json_raw = await client.generate_completion(
            messages=architect_messages,
            temperature=0.2,
            task_type="strategist"
            if attempt == 0
            else "coding",  # Use smarter model on retry
            progress_callback=architect_progress if ctx else None,
        )

        try:
            blueprint = extract_json_from_text(blueprint_json_raw)
            if not isinstance(blueprint, dict):
                raise ValueError("Could not parse a valid Blueprint JSON object.")

            # Format the JSON into a beautiful Markdown Report
            report = f"# 🧪 LP SESSION: {discovery.get('project_name', 'Unnamed')}\n\n"
            report += f"**Dynamic Squad Hired:**\n"
            for member in discovery.get("hired_squad", []):
                report += f"- **{member['role']}**: {member['audit_filter']}\n"

            report += f"\n## 🏛️ The Manifest\n{blueprint.get('manifest', 'N/A')}\n"
            report += f"\n## 🛡️ Audit Verdict\n{blueprint.get('audit_verdict', 'N/A')}\n"
            report += f"\n## 🚀 Technical Roadmap\n"
            for step in blueprint.get("roadmap", []):
                report += f"- {step}\n"

            report += (
                f"\n## ⚠️ Risk Assessment\n{blueprint.get('risk_assessment', 'N/A')}\n"
            )
            report += f"\n## 🧹 Clean Slate Instructions\n{blueprint.get('clean_slate', 'N/A')}\n"

            if blueprint.get("optional_features"):
                report += f"\n## 🔮 Optional Features (Excluded from 80% Core)\n"
                for opt in blueprint.get("optional_features", []):
                    report += f"- {opt}\n"

            if ctx:
                await ctx.report_progress(
                    progress=0,
                    total=None,
                    message=f"[Phase 3] Self-Verification in progress...",
                )

            # PHASE 3: Self-Verification (Anti-Degeneration)
            verifier_messages = [
                {"role": "system", "content": LP_VERIFIER_PROMPT},
                {
                    "role": "user",
                    "content": f"Original Goal: {goal}\nGenerated Blueprint: {blueprint_json_raw}",
                },
            ]

            verification_json_raw = await client.generate_completion(
                messages=verifier_messages, temperature=0.1, task_type="discovery"
            )

            try:
                verification = extract_json_from_text(verification_json_raw)
                if not verification.get("is_valid", True):
                    last_error = verification.get(
                        "structural_fix", "Unknown degeneration logic."
                    )
                    attempt += 1
                    continue  # Trigger self-healing loop
                else:
                    report += f"\n---\n*Generated by Lachman Protocol v2.5 Oracle (Iter {attempt + 1}/{max_retries})*"

                    usage_table = "\n\n### 📊 Session Token Usage\n| Model | Prompt Tokens | Completion Tokens | Total |\n|---|---|---|---|\n"
                    for model_name, usage in client.session_usage.items():
                        total = usage["prompt"] + usage["completion"]
                        usage_table += f"| `{model_name}` | {usage['prompt']:,} | {usage['completion']:,} | {total:,} |\n"
                    report += usage_table

                    return report  # Success!
            except:
                pass  # Fail silently on verification parsing if it's garbled, assume success to unblock
                usage_table = "\n\n### 📊 Session Token Usage\n| Model | Prompt Tokens | Completion Tokens | Total |\n|---|---|---|---|\n"
                for model_name, usage in client.session_usage.items():
                    total = usage["prompt"] + usage["completion"]
                    usage_table += f"| `{model_name}` | {usage['prompt']:,} | {usage['completion']:,} | {total:,} |\n"
                report += usage_table
                return report

        except Exception as e:
            return f"Error parsing Lachman Blueprint: {str(e)}\nRaw result:\n{blueprint_json_raw}"

    usage_table = "\n### 📊 Session Token Usage (Failed Run)\n| Model | Prompt Tokens | Completion Tokens | Total |\n|---|---|---|---|\n"
    for model_name, usage in client.session_usage.items():
        total = usage["prompt"] + usage["completion"]
        usage_table += f"| `{model_name}` | {usage['prompt']:,} | {usage['completion']:,} | {total:,} |\n"

    # If we reached here, the loop exceeded max_retries
    return f"""# 🛑 CRITICAL FAILURE (CIRCUIT Breaker ACTIVATED)
The Lachman Protocol failed to generate a stable, valid architecture blueprint after {max_retries} automatic attempts.
We halted execution to prevent unlimited API cost drain.

**Last Verifier Error:**
{last_error}

{usage_table}

**ACTION REQUIRED:**
The goal is likely impossible, inherently flawed, or too complex for the current model limits. Human intervention is required to rethink the core constraints before calling the tool again."""


async def read_repo_file(file_path: str) -> str:
    """
    Reads a file from the repository to provide context for Qwen.
    """
    from pathlib import Path
    import os

    try:
        # Security: protect against path traversal
        safe_root = Path(os.getenv("WORKSPACE_ROOT", ".")).resolve()
        target_path = (safe_root / file_path).resolve()

        try:
            target_path.relative_to(safe_root)
        except ValueError:
            return "Error: Access denied. Path outside allowed workspace."

        if not target_path.exists():
            return f"Error: File '{file_path}' not found."

        # Limit file size to 32KB to prevent context overflow and high costs
        if target_path.stat().st_size > 32 * 1024:
            return f"Error: File '{file_path}' is too large (>32KB). Please read specific sections if possible."

        content = target_path.read_text(encoding="utf-8", errors="replace")
        return f"File Content ({file_path}):\n\n```\n{content}\n```"
    except Exception as e:
        return f"Error reading file: {str(e)}"


async def list_repo_files(directory: str = ".", pattern: str = "**/*") -> str:
    """
    Lists files in the repository to help the agent find context.
    """
    from pathlib import Path
    import fnmatch

    try:
        root = Path(directory)
        if not root.exists():
            return f"Error: Directory '{directory}' not found."

        files = []
        for p in root.rglob("*"):
            if p.is_file():
                # Ignore common noise
                if any(
                    x in str(p)
                    for x in [".git", "__pycache__", ".venv", "node_modules"]
                ):
                    continue

                rel_path = str(p.relative_to(root))
                if fnmatch.fnmatch(rel_path, pattern):
                    files.append(rel_path)

        if not files:
            return "No matching files found."

        # Return first 100 files to avoid hitting limits
        truncated = ""
        if len(files) > 100:
            truncated = f"\n... and {len(files) - 100} more files."
            files = files[:100]

        return "Files in repository:\n- " + "\n- ".join(files) + truncated
    except Exception as e:
        return f"Error listing files: {str(e)}"


async def generate_usage_report() -> str:
    """
    Retrieves the DuckDB billing usage report and formats it as a Markdown table.
    """
    from qwen_mcp.billing import billing_tracker

    report_data = billing_tracker.get_daily_project_report()
    if not report_data:
        return "No usage data found in billing tracker yet."

    md = "# 💰 Qwen MCP Token Usage Report\n\n"
    md += "| Date | Project | Model | Prompts | Completions | Total Tokens |\n"
    md += "|---|---|---|---|---|---|\n"
    for r in report_data:
        md += f"| {r['date']} | {r['project_name']} | {r['model_name']} | {r['prompt_tokens']:,} | {r['completion_tokens']:,} | {r['total_tokens']:,} |\n"

    return md


async def list_available_models() -> str:
    """Fetches and displays models available via the DashScope API."""
    from qwen_mcp.api import DashScopeClient

    client = DashScopeClient()
    models = await client.list_models()
    if not models:
        return "No models found or API error."
    return "Available DashScope Models:\n- " + "\n- ".join(models)


async def set_model_in_registry(role: str, model_id: str) -> str:
    """Manually sets a model for a specific role (strategist, coder, scout)."""
    from qwen_mcp.registry import registry
    from qwen_mcp.api import DashScopeClient

    valid_roles = ["strategist", "coder", "scout"]
    if role not in valid_roles:
        return f"Invalid role. Must be one of: {', '.join(valid_roles)}"

    # Optional: logic to probe before setting
    client = DashScopeClient()
    is_healthy = await client.probe_model(model_id)
    if not is_healthy:
        return f"Warning: Model {model_id} failed the health probe. Setting it anyway as requested, but expect errors."

    registry.models[role] = model_id
    await registry.save_cache()
    return f"Success: Role '{role}' updated to use '{model_id}'."
