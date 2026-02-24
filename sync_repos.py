import shutil
import os

src_dir = r"c:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local"
dst_dir = r"c:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-engine"

# List of things to sync
items_to_sync = ["README.md", "docs", "src", "pyproject.toml", "uv.lock", "Dockerfile"]


def sync():
    for item in items_to_sync:
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)

        if os.path.exists(s):
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
                print(f"Synced directory: {item}")
            else:
                shutil.copy2(s, d)
                print(f"Synced file: {item}")

    # Remove the old LP_SYSTEM_PROMPT.md from root if it exists in dst
    old_file = os.path.join(dst_dir, "LP_SYSTEM_PROMPT.md")
    if os.path.exists(old_file):
        os.remove(old_file)
        print("Removed legacy LP_SYSTEM_PROMPT.md from root")


if __name__ == "__main__":
    sync()
