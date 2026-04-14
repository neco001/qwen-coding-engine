# src/qwen_mcp/specter/enforcement.py
"""Enforcement utility functions for MCP tool validation."""
from datetime import datetime, timezone
from qwen_mcp.specter.telemetry import get_broadcaster
from qwen_mcp.billing import billing_tracker
from typing import Optional  # Use generic Optional instead of missing Context type


async def qwen_init_request(ctx: Optional[object] = None) -> str:
    """Reset telemetry counters and broadcast to HUD (Layer 2 enforcement pre-flight check).

    Required for all MCP tool calls to prevent protocol bypass.
    Backward-compatible with existing workflows via default parameters.
    """
    await get_broadcaster().broadcast_state(
        {
            "operation": "request_start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_live": True,
            "ctx_type": type(ctx).__name__ if ctx else None  # SECURITY: Prevent PII/secret leaks
        },
        project_id="default"
    )
    billing_tracker.reset_request_counter()
    return "✅ Request initialized"