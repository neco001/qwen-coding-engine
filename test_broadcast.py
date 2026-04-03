import sys
import asyncio
sys.path.append('src')

from qwen_mcp.specter.telemetry import get_broadcaster

async def test():
    broadcaster = get_broadcaster()
    
    # Test 1: broadcast_state with no clients
    print("Test 1: broadcast_state with no clients...")
    try:
        await broadcaster.broadcast_state({
            "active_model": "test",
            "is_live": True
        }, project_id="test_project")
        print("Test 1: PASSED - broadcast_state returned immediately")
    except Exception as e:
        print(f"Test 1: FAILED - {e}")
    
    # Test 2: get_state
    print("\nTest 2: get_state...")
    try:
        state = await broadcaster.get_state("test_project")
        print(f"Test 2: PASSED - got state with keys: {list(state.keys())[:5]}...")
    except Exception as e:
        print(f"Test 2: FAILED - {e}")
    
    # Test 3: start_request
    print("\nTest 3: start_request...")
    try:
        await broadcaster.start_request("test_project")
        print("Test 3: PASSED - start_request completed")
    except Exception as e:
        print(f"Test 3: FAILED - {e}")

if __name__ == '__main__':
    asyncio.run(test())
