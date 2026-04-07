"""
Sparring Engine v2 - Output Formatters

This module contains formatting and report assembly logic for sparring sessions.
"""

import logging
from typing import Optional, Dict, Any

from qwen_mcp.engines.sparring_v2.interfaces import ReportFormatter
from qwen_mcp.engines.session_store import SessionCheckpoint
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.sanitizer import ContentValidator

logger = logging.getLogger(__name__)


class OutputFormatter(ReportFormatter):
    """Concrete implementation of ReportFormatter for formatting sparring session output."""
    
    def format(self, session: SessionCheckpoint, results: Dict[str, Any]) -> str:
        """
        Format a sparring session report.
        
        Args:
            session: SessionCheckpoint with session data
            results: Dictionary of step results
            
        Returns:
            Formatted report string
        """
        return self._assemble_report(
            session,
            results.get("red", ""),
            results.get("blue", ""),
            results.get("white", ""),
            results.get("loops", 1),
        )
    
    def _format_output(self, raw: str, label: str) -> str:
        """
        Format output with reasoning hidden in details.
        
        Args:
            raw: Raw output string from LLM
            label: Label for the output section
            
        Returns:
            Formatted string with collapsible reasoning
        """
        # Ensure raw is a string (handle None, int, or other types gracefully)
        if not isinstance(raw, str):
            raw = str(raw) if raw is not None else ""
        
        if "<thought>" in raw:
            parts = raw.split("</thought>")
            thought = parts[0].replace("<thought>", "").strip()
            content = parts[1].strip() if len(parts) > 1 else ""
            return f"<details>\n<summary>🧠 Proces Myślowy ({label})</summary>\n\n{thought}\n</details>\n\n{content}"
        return ContentValidator.validate_response(raw)
    
    def _assemble_report(
        self,
        session: SessionCheckpoint,
        red: str,
        blue: str,
        white: str,
        loops: int,
    ) -> str:
        """
        Assemble final war game report.
        
        Args:
            session: Session checkpoint with topic and roles
            red: Red Cell critique content
            blue: Blue Cell defense content
            white: White Cell consensus content
            loops: Number of optimization loops
            
        Returns:
            Formatted report string
        """
        report = f"# 🛡️ War Game Report: {session.topic}\n\n"
        report += f"> **CONFIDENTIAL: STRATEGIC DRAFT ONLY. NOT FOR EXTERNAL DISTRIBUTION.**\n\n"
        report += f"> **Session ID:** `{session.session_id}`\n\n"
        report += f"> **Selected Roles:** {session.roles.get('red_role', 'Red')}, {session.roles.get('blue_role', 'Blue')}, {session.roles.get('white_role', 'White')}\n\n"
        report += f"## 🥊 Turn 2: {session.roles.get('red_role', 'Red')}\n\n{self._format_output(red, session.roles.get('red_role', 'Red'))}\n\n---\n\n"
        report += f"## 🛡️ Turn 3: {session.roles.get('blue_role', 'Blue')}\n\n{self._format_output(blue, session.roles.get('blue_role', 'Blue'))}\n\n---\n\n"
        report += f"## ⚖️ Turn 4: {session.roles.get('white_role', 'White')}\n\n{self._format_output(white, session.roles.get('white_role', 'White'))}\n\n"
        # Ensure loops is an int (handle JSON deserialization edge cases)
        loops_int = int(loops) if not isinstance(loops, int) else loops
        if loops_int > 1:
            report += f"\n\n*(Note: This report underwent {loops_int} optimization cycles)*"
        return report


class ReportAssembler:
    """Utility class for assembling sparring session reports."""
    
    @staticmethod
    def assemble_full_report(
        session_store,
        discovery: SparringResponse,
        red: SparringResponse,
        blue: SparringResponse,
        white: SparringResponse,
    ) -> str:
        """
        Assemble full session report from all 4 steps.
        
        Args:
            session_store: SessionStore instance to load session data
            discovery: Discovery step response
            red: Red Cell step response
            blue: Blue Cell step response
            white: White Cell step response
            
        Returns:
            Formatted full report string
        """
        session_id = discovery.session_id
        
        # Load session to get topic and roles
        session = session_store.load(session_id)
        if not session:
            return "# Error: Session not found"
        
        # Extract results from each step
        red_content = red.result.get('critique', '') if red.result else ''
        blue_content = blue.result.get('defense', '') if blue.result else ''
        white_content = white.result.get('consensus', '') if white.result else ''
        
        roles = session.roles if session.roles else {}
        formatter = OutputFormatter()
        
        report = f"# 🛡️ War Game Report: {session.topic}\n\n"
        report += f"> **Session ID:** `{session_id}`\n\n"
        report += f"> **Selected Roles:** {roles.get('red_role', 'Red')}, {roles.get('blue_role', 'Blue')}, {roles.get('white_role', 'White')}\n\n"
        
        # Format each section - handle empty content gracefully
        if red_content:
            report += f"## 🥊 Turn 2: {roles.get('red_role', 'Red')}\n\n{formatter._format_output(red_content, roles.get('red_role', 'Red'))}\n\n---\n\n"
        else:
            report += f"## 🥊 Turn 2: {roles.get('red_role', 'Red')}\n\n*No content*\n\n---\n\n"
        
        if blue_content:
            report += f"## 🛡️ Turn 3: {roles.get('blue_role', 'Blue')}\n\n{formatter._format_output(blue_content, roles.get('blue_role', 'Blue'))}\n\n---\n\n"
        else:
            report += f"## 🛡️ Turn 3: {roles.get('blue_role', 'Blue')}\n\n*No content*\n\n---\n\n"
        
        if white_content:
            report += f"## ⚖️ Turn 4: {roles.get('white_role', 'White')}\n\n{formatter._format_output(white_content, roles.get('white_role', 'White'))}\n\n"
        else:
            report += f"## ⚖️ Turn 4: {roles.get('white_role', 'White')}\n\n*No content*\n\n"
        
        return report
    
    @staticmethod
    def assemble_step_report(
        session: SessionCheckpoint,
        red: str,
        blue: str,
        white: str,
        loops: int,
    ) -> str:
        """
        Assemble report from step results.
        
        Args:
            session: Session checkpoint with topic and roles
            red: Red Cell critique content
            blue: Blue Cell defense content
            white: White Cell consensus content
            loops: Number of optimization loops
            
        Returns:
            Formatted step report string
        """
        formatter = OutputFormatter()
        return formatter._assemble_report(session, red, blue, white, loops)
