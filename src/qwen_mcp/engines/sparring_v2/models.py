"""
Sparring Engine v2 - Response Models

This module provides the SparringResponse dataclass with factory methods
for structured responses in the guided sparring UX.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SparringResponse:
    """Structured response for guided sparring UX."""
    success: bool = False
    message: str = ""
    session_id: Optional[str] = None
    step_completed: Optional[str] = None
    next_step: Optional[str] = None
    next_command: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    # Multi-turn tracking fields
    messages_appended: int = 0  # Number of messages added in this turn
    context_truncated: bool = False  # Indicates if context truncation occurred
    
    @classmethod
    def error(cls, message: str, error: str, session_id: Optional[str] = None,
              step: Optional[str] = None) -> "SparringResponse":
        """Factory for error responses."""
        return cls(
            success=False, message=message, session_id=session_id,
            step_completed=step, next_step=None, next_command=None,
            result=None, error=error,
            messages_appended=0, context_truncated=False
        )
    
    @classmethod
    def success(cls, session_id: str, step: str, next_step: Optional[str],
                next_command: Optional[str], result: Any, message: str,
                messages_appended: int = 0, context_truncated: bool = False) -> "SparringResponse":
        """Factory for success responses."""
        return cls(
            success=True, message=message, session_id=session_id,
            step_completed=step, next_step=next_step, next_command=next_command,
            result=result, error=None,
            messages_appended=messages_appended, context_truncated=context_truncated
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "session_id": self.session_id,
            "step_completed": self.step_completed,
            "next_step": self.next_step,
            "next_command": self.next_command,
            "result": self.result,
            "message": self.message,
            "error": self.error,
            "messages_appended": self.messages_appended,
            "context_truncated": self.context_truncated
        }
    
    def to_markdown(self) -> str:
        """Convert to human-readable markdown for MCP output."""
        if not self.success:
            return f"❌ **Error:** {self.error}\n\n{self.message}"
        
        lines = []
        lines.append(f"✅ **{self.step_completed.title()} completed!**")
        lines.append("")
        
        if self.session_id:
            lines.append(f"📋 **Session ID:** `{self.session_id}`")
            lines.append("")
        
        # Multi-turn tracking info
        if self.messages_appended > 0:
            lines.append(f"💬 **Messages appended:** {self.messages_appended}")
            if self.context_truncated:
                lines.append("⚠️ **Context truncated:** Yes (rolling summary applied)")
            lines.append("")
        
        # Show result based on step
        if self.result:
            if isinstance(self.result, dict):
                if "roles" in self.result:
                    lines.append("🎭 **Wybrane role:**")
                    roles = self.result["roles"]
                    if "red_role" in roles:
                        lines.append(f"   • Red:  \"{roles['red_role']}\"")
                    if "blue_role" in roles:
                        lines.append(f"   • Blue: \"{roles['blue_role']}\"")
                    if "white_role" in roles:
                        lines.append(f"   • White: \"{roles['white_role']}\"")
                    lines.append("")
                elif "critique" in self.result:
                    lines.append("📝 **Red Critique:**")
                    lines.append(f"{self.result['critique'][:500]}...")
                    lines.append("")
                elif "defense" in self.result:
                    lines.append("🛡️ **Blue Defense:**")
                    lines.append(f"{self.result['defense'][:500]}...")
                    lines.append("")
                elif "consensus" in self.result:
                    lines.append("⚖️ **White Consensus:**")
                    lines.append(f"{self.result['consensus'][:500]}...")
                    lines.append("")
                elif "strategy" in self.result:
                    lines.append("💡 **Flash Strategy:**")
                    lines.append(f"{self.result['strategy'][:500]}...")
                    lines.append("")
        
        if self.next_step:
            lines.append(f"➡️ **Next step:** `{self.next_step}`")
            lines.append("")
            lines.append(f"💡 **Tip:** Run `sparring(session_id='{self.session_id}', mode='{self.next_step}')` to continue")
            lines.append("")
        
        if self.next_command:
            lines.append(f"📋 **Command:** `{self.next_command}`")
            lines.append("")
        
        return "\n".join(lines)