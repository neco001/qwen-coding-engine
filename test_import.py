print("Importing server...")
try:
    import qwen_mcp.server
    print("Import successful.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Import failed: {e}")
