import os
import hashlib

def get_current_project_id() -> str:
    """
    Identifies the unique project ID based on the environment or CWD.
    Prioritizes QWEN_PROJECT_NAME, then truncated hash of CWD.
    """
    project_name = os.getenv("QWEN_PROJECT_NAME")
    if project_name:
        return project_name
    
    cwd = os.getcwd()
    # Normalize path for multi-window consistency
    normalized_cwd = os.path.normpath(cwd).lower()
    project_hash = hashlib.sha256(normalized_cwd.encode()).hexdigest()[:8]
    return f"project_{project_hash}"
