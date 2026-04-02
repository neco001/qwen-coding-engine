"""Initialize .qwen directory structure for Decision Log."""

import os
from pathlib import Path
import pyarrow.parquet as pq

from .decision_schema import DecisionSchema


def init_qwen_directory(project_root: Path) -> Path:
    """Initialize .qwen directory in project root.
    
    Creates:
    - .qwen/decision_log.parquet (initialized with schema)
    - .qwen/.gitignore (ignores temp files)
    - .qwen/README.md (usage instructions)
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        Path to created .qwen directory
    """
    qwen_dir = project_root / ".qwen"
    
    # Create directory
    qwen_dir.mkdir(parents=True, exist_ok=True)
    
    # Create decision_log.parquet with empty schema
    parquet_path = qwen_dir / "decision_log.parquet"
    if not parquet_path.exists():
        empty_table = DecisionSchema.create_empty_table()
        pq.write_table(empty_table, str(parquet_path))
    
    # Create .gitignore
    gitignore_path = qwen_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(
            "# Ignore temporary files\n"
            "*.tmp\n"
            "*.lock\n"
            "*.bak\n"
            "\n"
            "# Keep decision_log.parquet\n"
            "!decision_log.parquet\n"
        )
    
    # Create README.md
    readme_path = qwen_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# .qwen Directory\n\n"
            "This directory contains AI-Driven Testing System data.\n\n"
            "## Files\n\n"
            "- `decision_log.parquet`: Decision log with all AI-driven changes\n"
            "- `.gitignore`: Excludes temporary files\n\n"
            "## Decision Log Schema\n\n"
            "| Field | Type | Description |\n"
            "|-------|------|-------------|\n"
            "| timestamp | timestamp | When decision was made |\n"
            "| session_id | string | Sparring session ID |\n"
            "| session_type | string | coder/test/validator |\n"
            "| change_hash | string | Hash of proposed changes |\n"
            "| files_modified | list<string> | Affected files |\n"
            "| lines_changed | int64 | Number of lines changed |\n"
            "| dependency_graph_hash | string | Hash of dependency graph |\n"
            "| verdict | string | approved/rejected/pending |\n"
            "| risk_score | float32 | Risk score (0.0-1.0) |\n"
            "| validator_triggers | list<string> | What triggered validation |\n"
            "| user_approval | bool | User approved change |\n"
            "| rationale | string | Decision rationale |\n\n"
            "## Usage\n\n"
            "```python\n"
            "from src.logging import DecisionLogWriter\n\n"
            "writer = DecisionLogWriter(\n"
            "    log_path=Path('.qwen/decision_log.parquet'),\n"
            "    backup_dir=Path.home() / 'AppData' / 'Roaming' / 'qwen-mcp' / 'decision_log_backup'\n"
            ")\n\n"
            "await writer.write_decision(decision_dict)\n"
            "```\n"
        )
    
    return qwen_dir


def get_qwen_dir(project_root: Path) -> Path:
    """Get .qwen directory path, creating if needed.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        Path to .qwen directory
    """
    qwen_dir = project_root / ".qwen"
    if not qwen_dir.exists():
        return init_qwen_directory(project_root)
    return qwen_dir