import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path.cwd() / "src"))

from qwen_mcp.config.sos_paths import DEFAULT_SOS_PATHS
from qwen_mcp.engines.decision_log_sync import DecisionLogSyncEngine

async def test_fix():
    print(f"Current Directory: {os.getcwd()}")
    
    # Simulate being in a different directory
    # Create a dummy folder and markers
    project_root = Path.cwd()
    print(f"Project Root: {project_root}")
    
    # Resolve from CWD
    resolved = DEFAULT_SOS_PATHS.resolve_workspace_root(".")
    print(f"Resolved Root (.): {resolved}")
    
    dl_path = DEFAULT_SOS_PATHS.get_decision_log_path(".")
    print(f"Decision Log Path: {dl_path}")
    
    engine = DecisionLogSyncEngine(dl_path)
    print(f"Engine Lock Path: {engine.lock_path}")
    
    print("Attempting to acquire lock...")
    try:
        engine._acquire_lock()
        print("Success: Lock acquired")
        engine._release_lock()
        print("Success: Lock released")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fix())
