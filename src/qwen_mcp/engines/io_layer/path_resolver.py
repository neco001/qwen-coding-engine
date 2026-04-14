"""
PathResolver: Centralised path configuration for SOS Sync system.

Rules:
- BACKLOG.md -> workspace_root/.PLAN/BACKLOG.md
- CHANGELOG.md -> workspace_root/CHANGELOG.md  (ROOT, not .PLAN/)
- decision_log.parquet -> workspace_root/.decision_log/decision_log.parquet
"""
from pathlib import Path
from typing import Optional, Union


class PathResolver:
    """Centralise all SOS Sync file path resolution."""

    PLAN_DIR = ".PLAN"
    DECISION_LOG_DIR = ".decision_log"

    def __init__(self, workspace_root: Optional[Union[str, Path]] = None):
        self.workspace_root = self._resolve_root(workspace_root)

    @staticmethod
    def _resolve_root(workspace_root: Optional[Union[str, Path]]) -> Path:
        if not workspace_root or str(workspace_root) == ".":
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                    return parent.resolve()
            return current.resolve()
        return Path(workspace_root).resolve()

    @property
    def backlog_path(self) -> Path:
        """BACKLOG.md is always in .PLAN/ subfolder."""
        return self.workspace_root / self.PLAN_DIR / "BACKLOG.md"

    @property
    def changelog_path(self) -> Path:
        """CHANGELOG.md is always in the project root, NOT in .PLAN/."""
        return self.workspace_root / "CHANGELOG.md"

    @property
    def decision_log_path(self) -> Path:
        """decision_log.parquet location."""
        return self.workspace_root / self.DECISION_LOG_DIR / "decision_log.parquet"

    def ensure_dirs(self) -> None:
        """Create required directories if missing."""
        (self.workspace_root / self.PLAN_DIR).mkdir(parents=True, exist_ok=True)
        (self.workspace_root / self.DECISION_LOG_DIR).mkdir(parents=True, exist_ok=True)
