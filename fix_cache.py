import json
import os
from pathlib import Path
from platformdirs import user_cache_dir

def fix_cache():
    cache_dir = Path(user_cache_dir("qwen-coding", "Qwen"))
    # Check if we should use Qwen/qwen-coding or just qwen-coding
    # The registry uses user_cache_dir("qwen-coding", "Qwen")
    # which on Windows typically is AppData/Local/Qwen/qwen-coding-local/Cache or similar
    # based on my previous ls, it was C:\Users\pawel\AppData\Local\Qwen\qwen-coding\Cache
    
    # Let's try to be precise or search for it
    paths_to_try = [
        cache_dir / "models_cache.json",
        cache_dir / "Cache" / "models_cache.json",
        Path(os.environ["LOCALAPPDATA"]) / "Qwen" / "qwen-coding" / "Cache" / "models_cache.json"
    ]
    
    data = {
        "schema_version": 2,
        "updated_at": "2026-03-01T20:30:00.000000",
        "models": {
            "strategist": "qwen3.5-plus",
            "coder": "qwen-plus",
            "specialist": "qwen2.5-coder-32b-instruct",
            "analyst": "qwq-plus",
            "scout": "qwen-turbo",
            "artist": "qwen-image-plus"
        },
        "metadata": {}
    }
    
    for p in paths_to_try:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"SUCCESS: Written to {p}")
        except Exception as e:
            print(f"FAILED: {p} - {e}")

if __name__ == "__main__":
    fix_cache()
