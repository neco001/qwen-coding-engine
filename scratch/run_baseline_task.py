"""Execute qwen_create_baseline_tool task directly."""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
from qwen_mcp.diff_audit import qwen_create_baseline


async def main():
    """Execute the baseline creation task."""
    workspace_root = str(Path(__file__).parent.parent.resolve())
    
    print(f"Creating baseline snapshot...")
    print(f"Workspace: {workspace_root}")
    
    result = await qwen_create_baseline(name="manual_baseline", workspace_root=workspace_root)
    
    print(f"Result: {result}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())