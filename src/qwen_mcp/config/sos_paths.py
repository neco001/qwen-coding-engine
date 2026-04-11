"""
SOS Sync Paths Configuration

Centralized configuration for all SOS Sync related paths:
- .PLAN/BACKLOG.md
- .PLAN/CHANGELOG.md
- .decision_log/decision_log.parquet
"""

from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel, Field


class SOSPathsConfig(BaseModel):
    """Configuration for SOS Sync file paths."""
    
    plan_dir: str = Field(
        default=".PLAN",
        description="Hidden directory for planning files (BACKLOG.md, CHANGELOG.md)"
    )
    
    decision_log_filename: str = Field(
        default="decision_log.parquet",
        description="Decision log parquet file name"
    )
    
    decision_log_subdir: str = Field(
        default=".decision_log",
        description="Subdirectory for decision log storage"
    )
    
    backlog_filename: str = Field(
        default="BACKLOG.md",
        description="Backlog file name"
    )
    
    changelog_filename: str = Field(
        default="CHANGELOG.md",
        description="Changelog file name"
    )
    
    @staticmethod
    def resolve_workspace_root(workspace_root: Optional[Union[str, Path]] = None) -> Path:
        """
        Resolve workspace_root to an absolute path.
        
        If workspace_root is None or ".", it searches upward for project markers 
        (.git, pyproject.toml) starting from CWD.
        """
        if not workspace_root or str(workspace_root) == ".":
            current = Path.cwd()
            # Search upward for project markers
            for parent in [current] + list(current.parents):
                if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                    return parent.resolve()
            # Fallback to absolute CWD
            return current.resolve()
        
        return Path(workspace_root).resolve()

    def get_decision_log_path(self, workspace_root: Optional[Union[str, Path]] = None) -> Path:
        """Get full path to decision log parquet file."""
        base = self.resolve_workspace_root(workspace_root)
        return base / self.decision_log_subdir / self.decision_log_filename
    
    def get_backlog_path(self, workspace_root: Optional[Union[str, Path]] = None) -> Path:
        """Get full path to BACKLOG.md file."""
        base = self.resolve_workspace_root(workspace_root)
        return base / self.plan_dir / self.backlog_filename
    
    def get_changelog_path(self, workspace_root: Optional[Union[str, Path]] = None) -> Path:
        """Get full path to CHANGELOG.md file."""
        base = self.resolve_workspace_root(workspace_root)
        return base / self.plan_dir / self.changelog_filename
    
    def ensure_directories_exist(self, workspace_root: Optional[Union[str, Path]] = None):
        """Create all required directories if they don't exist."""
        base = self.resolve_workspace_root(workspace_root)
        (base / self.plan_dir).mkdir(parents=True, exist_ok=True)
        (base / self.decision_log_subdir).mkdir(parents=True, exist_ok=True)


# Global default configuration instance
DEFAULT_SOS_PATHS = SOSPathsConfig()
