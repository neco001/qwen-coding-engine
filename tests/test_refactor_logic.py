
import pytest
from qwen_mcp.registry import registry
from qwen_mcp.tools import generate_code_pro, generate_code

def test_registry_roles():
    # Role 'coder' should map to 32B Coder (lite/smart)
    assert registry.models["coder"] == "qwen2.5-coder-32b-instruct"
    # Role 'coder_pro' should map to 72B Instruct (heavy/pro)
    assert registry.models["coder_pro"] == "qwen2.5-72b-instruct"

@pytest.mark.asyncio
async def test_tools_no_hardcode():
    # This should exist and not have model_override in the implementation
    # (Testing existence first)
    assert generate_code_pro is not None
    assert generate_code is not None
