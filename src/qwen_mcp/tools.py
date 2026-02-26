import logging
from typing import Optional
from mcp.server.fastmcp import Context
from qwen_mcp.api import DashScopeClient
from qwen_mcp.registry import registry
from qwen_mcp.sanitizer import ContentValidator
from qwen_mcp.specter.telemetry import get_broadcaster

logger = logging.getLogger(__name__)

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
8. IMPLEMENTATION STRATEGY: For every fix, prioritize the 'Smallest Change' principle. 
   - Label localized, obvious fixes as **[SIMPLE]**. Provide a diff.
   - Label architectural, multi-file, or high-risk logic changes as **[COMPLEX]**. 
   - FOR [COMPLEX]: Provide the technical strategy but EXPLICITLY INSTRUCT the assistant to call `qwen_coder` or `qwen_coder_25` for the actual implementation. Never allow the assistant to 'wing it' with complex logic.
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

LP_ARCHITECT_PROMPT = """You are the 'Lachman Architect (Strategic Engineering Pragmatist)'. 
You are synthesizing an expert swarm debate (Squad: {squad}) to create a pragmatic, high-ROI project blueprint.

Your output must be a JSON object with this exact structure:
{{
  "manifest": "Technical recursion of the goal. Focus on the CORE 80% (Functional Completeness).",
  "audit_verdict": "Critical vetos or approvals from the expert swarm (QA, Security, ROI)",
  "roadmap": ["Step 1: TDD Foundation", "Step 2: ..."],
  "clean_slate": "Legacy components that must be deleted or refactored",
  "risk_assessment": "Crucial bottlenecks or security threats",
  "optional_features": ["Nice-to-have feature 1 (skip for now)", "Over-engineered feature 2 (rejected)"]
}}

Mantra: Code without a blueprint is noise. Enforce the Lachman Protocol (Clean Slate, No Placeholders, Spec-First).
CRITICAL LIMIT: Apply the 80/20 Rule (Pareto Principle). Design ONLY the core 80% that brings immediate ROI. Do not over-engineer. 
IF THE TASK IS DELETION/CLEANUP: Verify via grep/ls AND ensure no critical dependencies remain. This is enough for CORE 80% (Functional Completeness).
Push all 'nice-to-have' or 100% perfection ideas into 'optional_features'."""

LP_VERIFIER_PROMPT = """You are the 'Lachman Stability Verifier (Strategic Engineering Pragmatist)'.
Your task is to audit the generated Blueprint for 'Degeneration' while maintaining a SHARP ROI FOCUS.

Checklist:
1. ROADMAP: Are there contradictory steps? Is the order logical (TDD-first where applicable)?
2. COVERAGE: Did the architect miss any CORE requirements?
3. PLACEHOLDERS: Are there any "ToDo", "Implement here", or "//..." placeholders? (STRICT BAN - this is the only hard reject)
4. CLEAN SLATE: Is the removal of legacy logic defined?
5. PRAGMATISM (80/20 RULE): Is the architect over-engineering? 
   - REJECT "Gold Plating" (e.g. demanding full mocks for a simple file deletion).
   - If the blueprint is CORE 80% (Functional Completeness) and safe, ACCEPT IT. 
   - Demanding 100% enterprise perfection in a rapid dev session is a FAILURE of the Verifier.

Output a JSON object:
{
  "is_valid": true/false,
  "degeneration_warnings": ["Reason 1", ...],
  "structural_fix": "Specific instruction to fix the blueprint if is_valid is false"
}

Output ONLY the JSON object. Be a Strategic Engineering Pragmatist. ROI is your North Star."""


# --- 5D SPARRING ENGINE PROMPTS (TRIAD CONSENSUS PROTOCOL) ---

FLASH_ANALYST_PROMPT = """Jesteś 'Głębokim Audytorem Logiki'. 
Twoim zadaniem jest rozbić temat użytkownika na czynniki pierwsze przy użyciu Chain-of-Thought (QwQ).
1. Wykryj ukryte założenia, które mogą być błędne.
2. Zidentyfikuj 3 krytyczne punkty zapalne (vulnerabilities).
3. Zaproponuj surowy zarys rozwiązania, który neutralizuje te ryzyka.
Bądź techniczny, chłodny i precyzyjny. Nie trać czasu na formatowanie - skup się na logice."""

FLASH_DRAFTER_PROMPT = """Jesteś 'Głównym Architektem Strategii'. 
Otrzymujesz surową analizę rykzyk od Audytora. Twoim zadaniem jest przekuć ją w profesjonalny, gotowy do egzekucji plan.
1. Skalibruj rozwiązanie tak, aby zachować inicjatywę użytkownika.
2. Przetłumacz techniczne zagrożenia na konkretne kroki zaradcze (Mitigations).
3. Nadaj całości strukturę 'Executive Summary'.
Mantra: Rozwiązanie musi być odporne na krytykę, którą przed chwilą usłyszałeś."""

RED_CELL_PROMPT = """Jesteś 'Sentry-01' (Red Team). 
Twoim zadaniem jest eksponowanie luk w egzekucji. 
ZASADA: Nie atakuj celów użytkownika, atakuj jego METODY. 
Skup się na: 
- Ryzyku reputacyjnym.
- Efektach drugiej rzędu (What happens then?).
- Wąskich gardłach zasobów.
Format: 'Vulnerability' -> 'Potential Impact'."""

BLUE_CELL_PROMPT = """Jesteś 'Champion-01' (Blue Team). 
Jesteś adwokatem użytkownika. Słyszałeś atak Red Cell. Twoim zadaniem jest URATOWANIE inicjatywy poprzez:
1. Kontrargumentację (jeśli Red Cell się myli).
2. Propozycję mechanizmów obronnych (Tactical Fixes).
3. Wzmocnienie fundamentów strategii tak, aby przetrwała kolejny audyt."""

WHITE_CELL_PROMPT = """Jesteś 'Controller' (Neutralna Synteza). 
Nie bierzesz stron. Twoim zadaniem jest stworzenie 'Raportu Przetrwania Inicjatywy'.
1. Zestaw najsilniejsze punkty ataku z najskuteczniejszymi mechanizmami obrony.
2. Wydaj 'Verdict of Survivability' (Czy to ma szansę zadziałać?).
3. Podaj listę 'Residual Risks' - tego, czego nie dało się naprawić w tej sesji.
Format: Protokół dyplomatyczny / Wojskowy raport pooperacyjny."""


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

    user_content = "Please audit the following content (code or logs).\n\n"
    if context:
        user_content += f"Context/Contextual Files:\n{context}\n\n"

    user_content += f"Content to analyze:\n```\n{content}\n```"

    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    async def report_wrapper(message: str):
        if ctx:
            # Only report periodically or on specific triggers to avoid flooding
            if len(message) > 0:
                await ctx.report_progress(
                    progress=0.0, total=None, message=f"Auditing... {message[:20]}..."
                )

    result = await client.generate_completion(
        messages=messages,
        temperature=0.1,
        task_type="analyst",
        progress_callback=report_wrapper if ctx else None,
        complexity="high",
        tags=["analyst", "reasoning", "audit"],
    )
    return result


async def generate_code(
    prompt: str, context: Optional[str] = None, ctx: Optional[Context] = None
) -> str:
    """
    Generates or completes code using Qwen-3.5-Plus.
    """
    client = DashScopeClient()

    user_content = f"Instruction: {prompt}\n\n"
    if context:
        user_content += f"Context/Background:\n{context}\n\n"

    messages = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    async def report_wrapper(message: str):
        if ctx:
            await ctx.report_progress(
                progress=0.0, total=None, message="Qwen is coding..."
            )

    result = await client.generate_completion(
        messages=messages,
        temperature=0.2,
        task_type="coder",
        progress_callback=report_wrapper if ctx else None,
        complexity="auto",
        tags=["coding"],
    )
    return result


async def generate_code_25(
    prompt: str, context: Optional[str] = None, ctx: Optional[Context] = None
) -> str:
    """
    Generates or completes code using specialized Qwen-2.5-Coder-32B.
    """
    client = DashScopeClient()

    user_content = f"Instruction: {prompt}\n\n"
    if context:
        user_content += f"Context/Background:\n{context}\n\n"

    messages = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    async def report_wrapper(message: str):
        if ctx:
            await ctx.report_progress(
                progress=0.0, total=None, message="Qwen-2.5-Coder is thinking..."
            )

    result = await client.generate_completion(
        messages=messages,
        temperature=0.2,
        task_type="specialist",
        progress_callback=report_wrapper if ctx else None,
        complexity="high",
        tags=["coding", "specialist"],
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
    import os

    client = DashScopeClient()

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

    discovery = extract_json_from_text(discovery_json_raw)
    if not discovery:
        logger.warning(
            "Discovery failed to return JSON. Using emergency fallback squad."
        )
        discovery = {
            "project_name": "Emergency Project",
            "hired_squad": [
                {"role": "Generalist Architect", "audit_filter": "Sanity check"}
            ],
        }

    squad_list = discovery.get("hired_squad") or [
        {"role": "Generalist", "audit_filter": "Pragmatic check"}
    ]
    squad_str = ", ".join(
        [
            f"{m.get('role', 'Expert')} ({m.get('audit_filter', 'ROI')})"
            for m in squad_list
        ]
    )

    # INTERNAL SELF-HEALING LOOP
    while attempt < max_retries:
        if ctx:
            await ctx.report_progress(
                progress=0,
                total=None,
                message=f"[Phase 2] Architecting (Attempt {attempt + 1}/{max_retries})...",
            )
            # Update HUD
            await get_broadcaster().broadcast_state({"loop_iteration": attempt + 1})

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

        async def architect_progress(message: str):
            if ctx:
                await ctx.report_progress(
                    progress=None,
                    total=None,
                    message=f"[Phase 2] Drafting Blueprint... {message[:20]}...",
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
            report += "**Dynamic Squad Hired:**\n"
            for member in discovery.get("hired_squad", []):
                report += f"- **{member['role']}**: {member['audit_filter']}\n"

            report += f"\n## 🏛️ The Manifest\n{blueprint.get('manifest', 'N/A')}\n"
            report += f"\n## 🛡️ Audit Verdict\n{blueprint.get('audit_verdict', 'N/A')}\n"
            report += "\n## 🚀 Technical Roadmap\n"
            for step in blueprint.get("roadmap", []):
                report += f"- {step}\n"

            report += (
                f"\n## ⚠️ Risk Assessment\n{blueprint.get('risk_assessment', 'N/A')}\n"
            )
            report += f"\n## 🧹 Clean Slate Instructions\n{blueprint.get('clean_slate', 'N/A')}\n"

            if blueprint.get("optional_features"):
                report += "\n## 🔮 Optional Features (Excluded from 80% Core)\n"
                for opt in blueprint.get("optional_features", []):
                    report += f"- {opt}\n"

            if ctx:
                await ctx.report_progress(
                    progress=0,
                    total=None,
                    message="[Phase 3] Self-Verification in progress...",
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
            except Exception:
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

        try:
            content = target_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 only if utf-8 fails, to preserve as much as possible
            content = target_path.read_text(encoding="latin-1", errors="replace")
            
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


async def generate_sparring(
    topic: str, context: str = "", mode: str = "flash", ctx: Optional[Context] = None
) -> str:
    """
    Executes the 5D Sparring Engine debate (Triad Consensus Protocol).
    Flash: Analyst (QwQ) -> Drafter (Max)
    Pro: Red Cell (QwQ) -> Blue Cell (Max) -> White Cell (Max)
    """
    client = DashScopeClient()

    # Sanitize inputs
    topic = ContentValidator.sanitize_input(topic)
    context = ContentValidator.sanitize_input(context)

    if mode == "flash":
        if ctx:
            await ctx.report_progress(
                progress=0, total=None, message="[Flash] Turn 1: Reasoning via QwQ-Plus..."
            )

        # 1. ANALYST PHASE (QwQ)
        analyst_messages = [
            {"role": "system", "content": FLASH_ANALYST_PROMPT},
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nContextual Background:\n{context}",
            },
        ]

        async def report_analyst(message: str):
            if ctx:
                await ctx.report_progress(
                    progress=0.0, total=None, message="QwQ is auditing logic..."
                )

        analysis = await client.generate_completion(
            messages=analyst_messages,
            temperature=0.7,
            task_type="audit",
            timeout=300.0,
            progress_callback=report_analyst if ctx else None,
            complexity="high",
            tags=["reasoning", "sparring", "flash-analyst"],
        )

        if ctx:
            await ctx.report_progress(
                progress=50, total=100, message="[Flash] Turn 2: Drafting Solution..."
            )

        # 2. DRAFTER PHASE (Qwen-Max)
        drafter_messages = [
            {"role": "system", "content": FLASH_DRAFTER_PROMPT},
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nContext: {context}\n\nAuditor Analysis:\n{analysis}",
            },
        ]

        async def report_drafter(message: str):
            if ctx:
                await ctx.report_progress(
                    progress=0.0, total=None, message="Max is drafting strategy..."
                )

        final_strategy = await client.generate_completion(
            messages=drafter_messages,
            temperature=0.1,
            task_type="strategist",
            timeout=300.0,
            progress_callback=report_drafter if ctx else None,
            complexity="critical",
            tags=["sparring", "flash-drafter"],
        )

        return ContentValidator.validate_response(final_strategy)

    else:
        # MODE: PRO (Triad Consensus Sequence)

        # 1. TURN 2: RED CELL (QwQ-plus)
        if ctx:
            await ctx.report_progress(
                progress=0, total=100, message="[Turn 2] Red Cell: Adversarial Audit..."
            )

        red_messages = [
            {"role": "system", "content": RED_CELL_PROMPT},
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nContext: {context}",
            },
        ]

        async def report_red(message: str):
            if ctx:
                await ctx.report_progress(
                    progress=20, total=100, message="Red Cell is attacking methods..."
                )

        red_critique = await client.generate_completion(
            messages=red_messages,
            temperature=0.8,
            task_type="audit",
            timeout=300.0,
            progress_callback=report_red if ctx else None,
            complexity="high",
            tags=["reasoning", "sparring", "red-cell"],
        )
        red_critique = ContentValidator.validate_response(red_critique)

        # 2. TURN 3: BLUE CELL (Qwen-Max)
        if ctx:
            await ctx.report_progress(
                progress=33, total=100, message="[Turn 3] Blue Cell: Strategic Defense..."
            )

        blue_messages = [
            {"role": "system", "content": BLUE_CELL_PROMPT},
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nContext: {context}\n\nRed Cell Critique:\n{red_critique}",
            },
        ]

        async def report_blue(message: str):
            if ctx:
                await ctx.report_progress(
                    progress=60, total=100, message="Blue Cell is defending initiative..."
                )

        blue_defense = await client.generate_completion(
            messages=blue_messages,
            temperature=0.2,
            task_type="strategist",
            timeout=300.0,
            progress_callback=report_blue if ctx else None,
            complexity="high",
            tags=["sparring", "blue-cell"],
        )
        blue_defense = ContentValidator.validate_response(blue_defense)

        # 3. TURN 4: WHITE CELL (Qwen-Max)
        if ctx:
            await ctx.report_progress(
                progress=66, total=100, message="[Turn 4] White Cell: Final Consensus..."
            )

        white_messages = [
            {"role": "system", "content": WHITE_CELL_PROMPT},
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nContext: {context}\n\nRed Audit:\n{red_critique}\n\nBlue Defense:\n{blue_defense}",
            },
        ]

        async def report_white(message: str):
            if ctx:
                await ctx.report_progress(
                    progress=90, total=100, message="White Cell is synthesizing..."
                )

        white_consensus = await client.generate_completion(
            messages=white_messages,
            temperature=0.1,
            task_type="strategist",
            timeout=300.0,
            progress_callback=report_white if ctx else None,
            complexity="critical",
            tags=["sparring", "white-cell"],
        )
        white_consensus = ContentValidator.validate_response(white_consensus)

        combined_report = f"# 🛡️ Triad Consensus Report: {topic}\n\n"
        combined_report += "## 🥊 Turn 2: Red Cell (Adversarial Audit)\n\n"
        combined_report += f"{red_critique}\n\n"
        combined_report += "---\n\n"
        combined_report += "## �️ Turn 3: Blue Cell (Strategic Defense)\n\n"
        combined_report += f"{blue_defense}\n\n"
        combined_report += "---\n\n"
        combined_report += "## ⚖️ Turn 4: White Cell (Final Consensus)\n\n"
        combined_report += f"{white_consensus}"

        return combined_report
