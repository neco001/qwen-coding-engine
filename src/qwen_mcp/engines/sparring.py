from typing import Optional, Dict
from mcp.server.fastmcp import Context
from qwen_mcp.api import DashScopeClient
from qwen_mcp.sanitizer import ContentValidator
from qwen_mcp.prompts.sparring import (
    SPARRING_DISCOVERY_PROMPT,
    FLASH_ANALYST_PROMPT,
    FLASH_DRAFTER_PROMPT,
    RED_CELL_PROMPT,
    BLUE_CELL_PROMPT,
    WHITE_CELL_PROMPT
)
from qwen_mcp.tools import extract_json_from_text

async def report_safe_progress(ctx: Optional[Context], progress: float, message: str):
    """Unified telemetry helper to avoid Pydantic float/None errors."""
    if ctx:
        try:
            # Explicitly force float to satisfy Pydantic
            await ctx.report_progress(progress=float(progress), message=message)
        except Exception:
            pass

class SparringEngine:
    def __init__(self, client: DashScopeClient):
        self.client = client

    async def run_flash(self, topic: str, context: str, ctx: Optional[Context]):
        await report_safe_progress(ctx, 0.0, "[Flash] Turn 1: Reasoning via QwQ-Plus...")
        
        analyst_messages = [
            {"role": "system", "content": FLASH_ANALYST_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nContext:\n{context}"},
        ]

        async def report_analyst(chunk: str):
            await report_safe_progress(ctx, 10.0, f"QwQ: {chunk[:40]}...")

        analysis = await self.client.generate_completion(
            messages=analyst_messages,
            temperature=0.7,
            task_type="audit",
            timeout=300.0,
            progress_callback=report_analyst if ctx else None,
            complexity="high",
            tags=["sparring", "flash-analyst"]
        )

        await report_safe_progress(ctx, 50.0, "[Flash] Turn 2: Drafting Strategy...")

        drafter_messages = [
            {"role": "system", "content": FLASH_DRAFTER_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nContext: {context}\n\nAnalysis:\n{analysis}"},
        ]

        async def report_drafter(chunk: str):
            await report_safe_progress(ctx, 60.0, f"Max: {chunk[:40]}...")

        final_strategy = await self.client.generate_completion(
            messages=drafter_messages,
            temperature=0.1,
            task_type="strategist",
            timeout=300.0,
            progress_callback=report_drafter if ctx else None,
            complexity="critical",
            tags=["sparring", "flash-drafter"],
            include_reasoning=True
        )

        return self.format_output(final_strategy, "Flash Analysis")

    async def run_pro(self, topic: str, context: str, ctx: Optional[Context]):
        await report_safe_progress(ctx, 0.0, "[Discovery] Assembling Expert Bench...")
        
        discovery_messages = [
            {"role": "system", "content": SPARRING_DISCOVERY_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nContext: {context}"},
        ]

        discovery_raw = await self.client.generate_completion(
            messages=discovery_messages,
            temperature=0.0,
            task_type="strategist",
            timeout=60.0,
            complexity="low",
            tags=["sparring", "discovery"]
        )
        
        try:
            roles = extract_json_from_text(discovery_raw)
            required_keys = ["red_role", "red_profile", "blue_role", "blue_profile", "white_role", "white_profile"]
            if not roles or not all(k in roles for k in required_keys):
                raise ValueError("Incomplete role discovery")
        except Exception:
            roles = {
                "red_role": "Red Cell (Adversarial Audit)",
                "red_profile": "Cyniczny audytor metod i niuansów persony",
                "blue_role": "Blue Cell (Strategic Defense)",
                "blue_profile": "Adwokat użytkownika i autentyczności tonu",
                "white_role": "White Cell (Final Consensus)",
                "white_profile": "Chief of Staff dbający o logiczną spójność i ROI"
            }

        # 1. TURN 2: RED CELL
        await report_safe_progress(ctx, 20.0, f"[Turn 2] {roles['red_role']} auditing...")

        red_messages = [
            {"role": "system", "content": f"Jesteś {roles['red_role']}. Profil: {roles['red_profile']}\n\nZADANIE:\n{RED_CELL_PROMPT}"},
            {"role": "user", "content": f"Topic: {topic}\n\nContext: {context}"},
        ]

        async def report_red(chunk: str):
            await report_safe_progress(ctx, 25.0, f"Red: {chunk[:40]}...")

        red_critique = await self.client.generate_completion(
            messages=red_messages,
            temperature=0.8,
            task_type="audit",
            timeout=300.0,
            progress_callback=report_red if ctx else None,
            complexity="high",
            tags=["reasoning", "sparring", "red-cell"],
            include_reasoning=True
        )
        red_critique = ContentValidator.validate_response(red_critique)

        # 2 & 3. DYNAMIC REGENERATION LOOP
        loop_count, max_loops = 0, 2
        white_consensus, blue_defense = "", ""

        while loop_count < max_loops:
            loop_count += 1
            
            # 2. TURN 3: BLUE CELL
            await report_safe_progress(ctx, float(40 + (loop_count * 10)), f"[Turn 3] {roles['blue_role']} defending...")

            blue_system = f"Jesteś {roles['blue_role']}. Profil: {roles['blue_profile']}\n\nZADANIE:\n{BLUE_CELL_PROMPT}"
            if "[REGENERATE" in white_consensus:
                blue_system += f"\n\nUWAGA: Poprzednia próba odrzucona: {white_consensus.split(']', 1)[0].replace('[REGENERATE:', '').strip()}."

            blue_messages = [
                {"role": "system", "content": blue_system},
                {"role": "user", "content": f"Topic: {topic}\n\nContext: {context}\n\nRed Critique:\n{red_critique}"},
            ]

            async def report_blue(chunk: str):
                await report_safe_progress(ctx, float(40 + (loop_count * 10)), f"Blue: {chunk[:40]}...")

            blue_defense = await self.client.generate_completion(
                messages=blue_messages,
                temperature=0.5,
                task_type="strategist",
                timeout=300.0,
                progress_callback=report_blue if ctx else None,
                complexity="high",
                tags=["sparring", "blue-cell", f"loop-{loop_count}"],
                include_reasoning=True
            )
            blue_defense = ContentValidator.validate_response(blue_defense)

            # 3. TURN 4: WHITE CELL
            await report_safe_progress(ctx, float(75 + (loop_count * 5)), f"[Turn 4] {roles['white_role']} synthesizing...")

            white_messages = [
                {"role": "system", "content": f"Jesteś {roles['white_role']}. Profil: {roles['white_profile']}\n\nZADANIE:\n{WHITE_CELL_PROMPT}"},
                {"role": "user", "content": f"Topic: {topic}\n\nContext: {context}\n\nRed Audit:\n{red_critique}\n\nBlue Defense:\n{blue_defense}"},
            ]

            async def report_white(chunk: str):
                await report_safe_progress(ctx, float(75 + (loop_count * 5)), f"White: {chunk[:40]}...")

            white_consensus = await self.client.generate_completion(
                messages=white_messages,
                temperature=0.1,
                task_type="strategist",
                timeout=300.0,
                progress_callback=report_white if ctx else None,
                complexity="critical",
                tags=["sparring", "white-cell", f"loop-{loop_count}"],
                include_reasoning=True
            )
            white_consensus = ContentValidator.validate_response(white_consensus)

            if "[REGENERATE" not in white_consensus or loop_count >= max_loops:
                if "[REGENERATE" in white_consensus:
                    white_consensus = white_consensus.split("]", 1)[-1].strip()
                break

        return self.assemble_report(topic, roles, red_critique, blue_defense, white_consensus, loop_count)

    def format_output(self, raw, label):
        if "<thought>" in raw:
            t, c = raw.split("</thought>")
            return f"<details>\n<summary>🧠 Proces Myślowy ({label})</summary>\n\n{t.replace('<thought>', '').strip()}\n</details>\n\n{c.strip()}"
        return ContentValidator.validate_response(raw)

    def assemble_report(self, topic, roles, red, blue, white, loops):
        report = f"# 🛡️ War Game Report: {topic}\n\n"
        report += f"> **CONFIDENTIAL: STRATEGIC DRAFT ONLY. NOT FOR EXTERNAL DISTRIBUTION.**\n\n"
        report += f"> **Selected Roles:** {roles['red_role']}, {roles['blue_role']}, {roles['white_role']}\n\n"
        report += f"## 🥊 Turn 2: {roles['red_role']}\n\n{self.format_output(red, roles['red_role'])}\n\n---\n\n"
        report += f"## 🛡️ Turn 3: {roles['blue_role']}\n\n{self.format_output(blue, roles['blue_role'])}\n\n---\n\n"
        report += f"## ⚖️ Turn 4: {roles['white_role']}\n\n{self.format_output(white, roles['white_role'])}\n\n"
        if loops > 1:
            report += f"\n\n*(Note: This report underwent {loops} optimization cycles)*"
        return report
