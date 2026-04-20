
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from qwen_mcp.tools import generate_lp_blueprint
from qwen_mcp.config.sos_paths import DEFAULT_SOS_PATHS

async def run_isolation_debug():
    target_workspace = "c:/SecureProject"
    goal = "add workspace isolation to tests"
    
    print(f"DEBUG_SCRIPT: Starting, target_workspace={target_workspace}")
    
    with patch("qwen_mcp.api.DashScopeClient") as mock_client_cls, \
         patch("qwen_mcp.engines.scout.ScoutEngine") as mock_scout_cls, \
         patch("qwen_mcp.engines.decision_log_sync.DecisionLogSyncEngine") as mock_sync_cls, \
         patch("qwen_mcp.tools.get_broadcaster") as mock_broadcaster_cls:
        
        # Setup mocks
        mock_client = mock_client_cls.return_value
        mock_client.generate_completion = AsyncMock(return_value='{"swarm_tasks": [{"id": "1", "task": "test"}]}')
        
        mock_scout = mock_scout_cls.return_value
        mock_scout.analyze_task = AsyncMock(return_value={"complexity": "low", "is_brownfield": True})
        
        mock_sync = mock_sync_cls.return_value
        mock_sync.add_tasks = AsyncMock()
        
        mock_broadcaster = mock_broadcaster_cls.return_value
        mock_broadcaster.broadcast_state = AsyncMock()
        
        print("DEBUG_SCRIPT: Calling generate_lp_blueprint...")
        try:
            # We need to see what's happening inside LP
            await generate_lp_blueprint(
                goal=goal,
                auto_add_tasks=True,
                workspace_root=target_workspace
            )
        except Exception as e:
            print(f"DEBUG_SCRIPT: Caught exception: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"DEBUG_SCRIPT: mock_sync_cls.called = {mock_sync_cls.called}")
        if mock_sync_cls.called:
            print(f"DEBUG_SCRIPT: call args = {mock_sync_cls.call_args}")

if __name__ == "__main__":
    asyncio.run(run_isolation_debug())
