"""
Context Builder Engine - Generates project context files using parallel worker pool.
"""

import os
import logging
import tempfile
import asyncio
import fnmatch
import shutil
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Set

from qwen_mcp.orchestrator import SwarmOrchestrator
from qwen_mcp.prompts.context import (
    PROJECT_CONTEXT_SYSTEM_PROMPT,
    DATA_CONTEXT_SYSTEM_PROMPT,
    SESSION_SUPPLEMENT_SYSTEM_PROMPT,
)
from qwen_mcp.engines.token_validator import estimate_tokens_heuristic

logger = logging.getLogger(__name__)

# Token threshold for chunking large files
CHUNK_TOKEN_THRESHOLD = 50000
# Target chunk size (40K tokens per chunk)
TARGET_CHUNK_SIZE = 40000

CONTEXT_DIR_NAME = ".context"

# Key files to scan
PROJECT_SCAN_FILES = [
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "setup.py",
    "Cargo.toml",
    "go.mod",
    "README.md",
    "AGENTS.md",
    "BUILD_GUIDE.md",
]

PROJECT_SCAN_DIRS = [
    "src",
    "lib",
    "app",
    "tests",
    "docs",
]

DATA_SCAN_FILES = [
    "*.duckdb",
    "*.db",
    "*.sqlite",
    "*.parquet",
    "*.csv",
]

# Max file size to read (in bytes)
MAX_FILE_SIZE = 50000
# Max lines per file to include in prompt
MAX_FILE_LINES = 100

def _load_gitignore_patterns(workspace_root: str) -> Set[str]:
    gitignore_path = Path(workspace_root) / ".gitignore"
    patterns: Set[str] = set()
    if not gitignore_path.exists():
        return patterns
    try:
        content = gitignore_path.read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pattern = line.strip("/")
            patterns.add(pattern)
    except Exception as e:
        logger.warning(f"Failed to load .gitignore: {e}")
    return patterns

def _should_exclude(filepath: Path, gitignore_patterns: Set[str]) -> bool:
    path_str = str(filepath)
    path_parts = filepath.parts
    for pattern in gitignore_patterns:
        clean_pattern = pattern.rstrip("/")
        if clean_pattern in path_parts:
            return True
        if fnmatch.fnmatch(path_str, clean_pattern):
            return True
        if fnmatch.fnmatch(filepath.name, clean_pattern):
            return True
    return False

class ContextBuilderEngine:
    """
    Engine for generating and updating context files using Swarm analysis with SOS patching.
    """
    
    def __init__(self, client: Any = None, context_dir: Optional[Path] = None):
        self.client = client
        self.context_dir = context_dir or Path(CONTEXT_DIR_NAME)
        self.orchestrator = SwarmOrchestrator(completion_handler=client) if client else None
    
    async def generate_project_context(
        self,
        workspace_root: str = "."
    ) -> Tuple[str, str]:
        """
        Generate _PROJECT_CONTEXT.md and _DATA_CONTEXT.md.
        
        OPTIMIZED: Uses existing _PROJECT_CONTEXT.md if available, otherwise generates
        using static analysis only (NO LLM calls per chunk - too slow for large projects).
        
        Returns:
            Tuple of (project_context_content, data_context_content)
        """
        # Resolve absolute context directory
        base_path = Path(workspace_root)
        resolved_context_dir = base_path / self.context_dir
        resolved_context_dir.mkdir(exist_ok=True, parents=True)
        
        # OPTIMIZATION: Check if _PROJECT_CONTEXT.md exists - use it directly
        existing_project_context = resolved_context_dir / "_PROJECT_CONTEXT.md"
        if existing_project_context.exists():
            logger.info(f"Using existing _PROJECT_CONTEXT.md from {existing_project_context}")
            project_context = existing_project_context.read_text(encoding="utf-8")
        else:
            project_context = await self._generate_static_project_context(workspace_root)
        data_context = self._generate_static_data_context(workspace_root)
        return project_context, data_context
    
    async def _generate_static_project_context(self, workspace_root: str = ".") -> str:
        scanned_files = self._scan_project_files(workspace_root)
        lines = ["# Project Context\n"]
        lines.append(f"Generated: {datetime.now().isoformat()}\n")
        lines.append(f"Files analyzed: {len(scanned_files)}\n")
        lines.append("---\n\n")
        for filename, content in scanned_files.items():
            if filename == "_directory_structure":
                lines.append(f"## Directory Structure\n```\n{content}\n```\n")
            else:
                lines.append(f"## {filename}\n```\n{content[:2000]}...\n```\n")
        return "\n".join(lines)
    
    def _generate_static_data_context(self, workspace_root: str = ".") -> str:
        data_scanned = self._scan_data_files(workspace_root)
        if not data_scanned:
            return "# Data Context\n\nNo data files detected in workspace."
        lines = ["# Data Context\n"]
        lines.append(f"Generated: {datetime.now().isoformat()}\n")
        lines.append("---\n\n")
        for filename, content in data_scanned.items():
            lines.append(f"## {filename}\n{content}\n")
        return "\n".join(lines)

    def _scan_project_files(self, workspace_root: str) -> Dict[str, str]:
        root = Path(workspace_root)
        scanned = {}
        gitignore_patterns = _load_gitignore_patterns(workspace_root)
        for filename in PROJECT_SCAN_FILES:
            filepath = root / filename
            if filepath.exists() and filepath.is_file():
                content = self._read_file_safe(filepath)
                if content: scanned[filename] = content
        for dirname in PROJECT_SCAN_DIRS:
            dirpath = root / dirname
            if dirpath.exists() and dirpath.is_dir():
                for pattern in ["*.py", "*.ts", "*.js", "*.go", "*.rs"]:
                    for filepath in dirpath.glob(pattern):
                        rel_path = filepath.relative_to(root)
                        if _should_exclude(rel_path, gitignore_patterns): continue
                        if any(kw in filepath.name.lower() for kw in ["main", "server", "app", "init", "index", "cli", "api", "tools", "registry", "base", "engine"]):
                            content = self._read_file_safe(filepath)
                            if content: scanned[str(rel_path)] = content
        scanned["_directory_structure"] = self._get_directory_structure(root)
        return scanned

    def _scan_data_files(self, workspace_root: str) -> Dict[str, str]:
        root = Path(workspace_root)
        scanned = {}
        gitignore_patterns = _load_gitignore_patterns(workspace_root)
        for pattern in DATA_SCAN_FILES:
            for filepath in root.glob(f"**/{pattern}"):
                rel_path = filepath.relative_to(root)
                if _should_exclude(rel_path, gitignore_patterns): continue
                size = filepath.stat().st_size if filepath.exists() else 0
                scanned[str(rel_path)] = f"[{filepath.suffix} file, {size} bytes]"
        return scanned

    def _read_file_safe(self, filepath: Path) -> Optional[str]:
        try:
            if filepath.stat().st_size > MAX_FILE_SIZE: return None
            content = filepath.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            if len(lines) > MAX_FILE_LINES:
                content = "\n".join(lines[:MAX_FILE_LINES]) + f"\n... [truncated]"
            return content
        except Exception: return None

    def _get_directory_structure(self, root: Path, max_depth: int = 3) -> str:
        lines = []
        def walk(path: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth: return
            try: items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            except PermissionError: return
            items = [i for i in items if not i.name.startswith(".") and i.name not in ["__pycache__", "node_modules", ".git", "venv", ".venv"]]
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                lines.append(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")
                if item.is_dir():
                    walk(item, prefix + ("    " if is_last else "│   "), depth + 1)
        walk(root)
        return "\n".join(lines[:50])

    def save_context_files(
        self,
        project_context: str,
        data_context: str,
        workspace_root: str = "."
    ) -> Dict[str, Path]:
        """
        Atomically save context files using temp + rename pattern.
        
        Returns:
            Dict mapping context type to file path
        """
        base_path = Path(workspace_root)
        resolved_context_dir = base_path / self.context_dir
        resolved_context_dir.mkdir(exist_ok=True, parents=True)
        
        saved_files = {}
        
        # Save project context
        project_path = resolved_context_dir / "_PROJECT_CONTEXT.md"
        self._atomic_write(project_path, project_context)
        saved_files["project"] = project_path
        
        # Save data context
        data_path = resolved_context_dir / "_DATA_CONTEXT.md"
        self._atomic_write(data_path, data_context)
        saved_files["data"] = data_path
        
        logger.info(f"Context files saved to {resolved_context_dir}")
        return saved_files
    
    def save_session_context(
        self,
        session_content: str,
        workspace_root: str = "."
    ) -> Path:
        """Atomically save session supplement."""
        base_path = Path(workspace_root)
        resolved_context_dir = base_path / self.context_dir
        resolved_context_dir.mkdir(exist_ok=True, parents=True)
        
        session_path = resolved_context_dir / "_SESSION_SUPPLEMENT.md"
        self._atomic_write(session_path, session_content)
        
        logger.info(f"Session context saved to {session_path}")
        return session_path
    
    async def update_session_context(
        self,
        summary: str,
        workspace_root: str = "."
    ) -> str:
        """
        Generate session supplement content from session summary.
        
        Uses LLM to format session summary into structured session supplement.
        
        Args:
            summary: Human-readable session summary
            workspace_root: Path to workspace root
            
        Returns:
            Formatted session supplement content as markdown string
        """
        from qwen_mcp.prompts.context import SESSION_SUPPLEMENT_SYSTEM_PROMPT
        
        base_path = Path(workspace_root)
        resolved_context_dir = base_path / self.context_dir
        resolved_context_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if existing session supplement exists
        existing_session_path = resolved_context_dir / "_SESSION_SUPPLEMENT.md"
        existing_content = ""
        if existing_session_path.exists():
            existing_content = existing_session_path.read_text(encoding="utf-8")
            logger.info(f"Found existing _SESSION_SUPPLEMENT.md, will merge")
        
        # Build prompt for LLM to format session summary
        user_prompt = f"""
Format the following session summary into a structured session supplement:

{summary}

The output should be in markdown format with the following sections:
- Session Date
- Objectives
- Accomplishments
- Decisions Made
- Open Questions
- Recommendations for Next Session

If there is existing content, merge the new information while preserving historical sessions.
""".strip()
        
        # Use LLM to format if client is available
        if self.client:
            try:
                response = await self.client.chat.completions.create(
                    model="qwen3.5-plus",
                    messages=[
                        {"role": "system", "content": SESSION_SUPPLEMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=0,  # unlimited - controlled by thinking_budget
                )
                session_content = response.choices[0].message.content
                logger.info(f"Generated session supplement using LLM")
            except Exception as e:
                logger.warning(f"LLM formatting failed, using fallback: {e}")
                session_content = self._format_session_summary_fallback(summary)
        else:
            # Fallback: format summary without LLM
            session_content = self._format_session_summary_fallback(summary)
        
        return session_content
    
    def _format_session_summary_fallback(self, summary: str) -> str:
        """
        Format session summary into markdown without LLM.
        
        Args:
            summary: Human-readable session summary
            
        Returns:
            Formatted markdown string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Parse summary lines into sections
        lines = summary.strip().split("\n")
        
        content = f"""# Session Supplement

**Generated:** {timestamp}

---

## Session Summary

{summary}

---

## Session History

<!-- Previous sessions will be appended here -->
""".strip()
        
        return content
    
    def _atomic_write(self, path: Path, content: str) -> None:
        """
        Write file atomically using temp + rename pattern.
        
        This prevents corruption if the process is interrupted during write.
        Matches the pattern from SessionStore.save().
        """
        parent_dir = path.parent
        # Write to temp file first
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            dir=str(parent_dir)
        )
        try:
            old_content = path.read_text(encoding="utf-8")
            old_sections = self._parse_markdown_sections(old_content)
            new_sections = self._parse_markdown_sections(content)
            merged_content = self._merge_sections(old_sections, new_sections)
            self._full_atomic_write(path, merged_content)
        except Exception as e:
            logger.warning(f"Patch write failed: {e}")
            self._full_atomic_write(path, content)

    def _full_atomic_write(self, path: Path, content: str) -> None:
        fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(content)
            os.replace(temp_path, str(path))
        except Exception:
            if os.path.exists(temp_path): os.unlink(temp_path)
            raise

    def _parse_markdown_sections(self, content: str) -> List[Tuple[str, str]]:
        lines = content.splitlines()
        sections = []
        c_header, c_content = "", []
        for line in lines:
            if line.strip().startswith('#'):
                if c_header or c_content: sections.append((c_header, '\n'.join(c_content)))
                c_header, c_content = line, []
            else: c_content.append(line)
        if c_header or c_content: sections.append((c_header, '\n'.join(c_content)))
        return sections

    def _merge_sections(self, old_sections: List[Tuple[str, str]], new_sections: List[Tuple[str, str]]) -> str:
        old_map = {h: c for h, c in old_sections}
        merged = [f"<!-- version: {datetime.now().strftime('%Y%m%d-%H%M')} -->", ""]
        for header, new_c in new_sections:
            if header in old_map:
                old_c = old_map[header]
                if self._content_changed(old_c, new_c): merged.extend([header, new_c])
                else: merged.extend([header, old_c])
            else: merged.extend([header, new_c])
        return '\n'.join(merged)

    def _content_changed(self, old_c: str, new_c: str) -> bool:
        def norm(c: str) -> str:
            c = re.sub(r'<!--\s*version:.*-->', '', c)
            return re.sub(r'\s+', ' ', c).strip()
        return norm(old_c) != norm(new_c)
