"""
SOS Sync Paths Configuration

Centralized configuration for all SOS Sync related paths:
- .PLAN/BACKLOG.md
- .PLAN/CHANGELOG.md
- .decision_log/decision_log.parquet
"""

from pathlib import Path
from typing import Optional
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
    
    def get_decision_log_path(self, workspace_root: Optional[Path] = None) -> Path:
        """Get full path to decision log parquet file."""
        base = Path(workspace_root) if workspace_root else Path.cwd()
        return base / self.decision_log_subdir / self.decision_log_filename
    
    def get_backlog_path(self, workspace_root: Optional[Path] = None) -> Path:
        """Get full path to BACKLOG.md file."""
        base = Path(workspace_root) if workspace_root else Path.cwd()
        return base / self.plan_dir / self.backlog_filename
    
    def get_changelog_path(self, workspace_root: Optional[Path] = None) -> Path:
        """Get full path to CHANGELOG.md file."""
        base = Path(workspace_root) if workspace_root else Path.cwd()
        return base / self.plan_dir / self.changelog_filename
    
    def ensure_directories_exist(self, workspace_root: Optional[Path] = None):
        """Create all required directories if they don't exist."""
        base = Path(workspace_root) if workspace_root else Path.cwd()
        (base / self.plan_dir).mkdir(parents=True, exist_ok=True)
        (base / self.decision_log_subdir).mkdir(parents=True, exist_ok=True)


# Global default configuration instance
DEFAULT_SOS_PATHS = SOSPathsConfig()
