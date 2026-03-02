from qwen_mcp.registry import registry
import json

def test_registry():
    print(f"Registry cache file: {registry.cache_file}")
    print("Attempting to load registry...")
    try:
        registry.load_cache()
        print("Registry models:")
        print(json.dumps(registry.models, indent=2))
        print("Registry successfuly loaded.")
    except Exception as e:
        print(f"Error loading registry: {e}")

if __name__ == "__main__":
    test_registry()
