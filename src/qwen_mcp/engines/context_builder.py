"""
Context Builder Engine - Generates project context files using Swarm analysis.

This engine creates and maintains:
- .context/_PROJECT_CONTEXT.md: Tech stack, structure, conventions
- .context/_DATA_CONTEXT.md: Data sources, schemas, pipelines
- .context/_SESSION_SUPPLEMENT.md: Session history and recommendations

Features:
- Scout analysis for complexity-based Swarm routing
- Atomic file writes (temp + rename pattern)
- Session continuity tracking
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

from qwen_mcp.orchestrator import SwarmOrchestrator
from qwen_mcp.prompts.context import (
    PROJECT_CONTEXT_SYSTEM_PROMPT,
    DATA_CONTEXT_SYSTEM_PROMPT,
    SESSION_SUPPLEMENT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

CONTEXT_DIR_NAME = ".context"

# Key files to scan for project context
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

# Key directories to scan
PROJECT_SCAN_DIRS = [
    "src",
    "lib",
    "app",
    "tests",
    "docs",
]

# Key files for data context
DATA_SCAN_FILES = [
    "*.duckdb",
    "*.db",
    "*.sqlite",
    "*.parquet",
    "*.csv",
]

# Max file size to read (in bytes) - avoid huge files
MAX_FILE_SIZE = 50000

# Max lines per file to include in prompt
MAX_FILE_LINES = 100


class ContextBuilderEngine:
    """
    Engine for generating and updating context files using Swarm analysis.
    
    Features:
    - Parallel analysis of tech stack, structure, data, and docs
    - Atomic file writes with temp files
    - Session continuity tracking
    """
    
    def __init__(
        self,
        client: Any = None,
        context_dir: Optional[Path] = None,
    ):
        """
        Initialize the ContextBuilderEngine.
        
        Args:
            client: DashScopeClient for LLM completions
            context_dir: Directory for context files (default: .context)
        """
        self.client = client
        self.context_dir = context_dir or Path(CONTEXT_DIR_NAME)
        self.orchestrator = SwarmOrchestrator(completion_handler=client) if client else None
    
    async def generate_project_context(
        self, 
        workspace_root: str = "."
    ) -> Tuple[str, str]:
        """
        Generate _PROJECT_CONTEXT.md and _DATA_CONTEXT.md.
        
        Uses Scout analysis to determine if Swarm parallel execution is warranted.
        
        Returns:
            Tuple of (project_context_content, data_context_content)
        """
        from qwen_mcp.engines.scout import ScoutEngine
        
        # Ensure context directory exists
        self.context_dir.mkdir(exist_ok=True, parents=True)
        
        # SCOUT ANALYSIS: Determine complexity and Swarm usage
        scout = ScoutEngine(self.client)
        scout_prompt = f"Analyze codebase at {workspace_root} for context documentation scope."
        scout_result = await scout.analyze_task(
            scout_prompt,
            context="Context file generation for project documentation",
            task_hint="context_analysis"
        )
        
        use_swarm = scout_result.get("use_swarm", False)
        complexity = scout_result.get("complexity", "medium")
        
        logger.info(f"Scout result: complexity={complexity}, use_swarm={use_swarm}")
        
        if use_swarm and self.orchestrator:
            # Swarm prompt for parallel context analysis
            swarm_prompt = self._build_context_swarm_prompt(workspace_root)
            await self.orchestrator.run_swarm(swarm_prompt, task_type="context_analysis")
        
        # Generate each context type separately for quality
        project_context = await self._generate_single_context(
            workspace_root, "project", complexity
        )
        data_context = await self._generate_single_context(
            workspace_root, "data", complexity
        )
        
        return project_context, data_context
    
    def _build_context_swarm_prompt(self, workspace_root: str) -> str:
        """Build the swarm decomposition prompt for context analysis."""
        return f"""
Analyze this codebase and generate comprehensive context documentation.

Workspace: {workspace_root}

Required analyses (execute in parallel):
1. **Tech Stack Analysis**: Identify runtime, frameworks, libraries, databases
2. **Structure Mapping**: Map directory structure, entry points, config files
3. **Data Sources**: Find database connections, data files, APIs, schemas
4. **Documentation Review**: Extract key conventions, workflows, scripts

Output should be structured for two files:
- _PROJECT_CONTEXT.md: Tech stack, structure, conventions
- _DATA_CONTEXT.md: Data sources, schemas, pipelines
"""
    
    def _scan_project_files(self, workspace_root: str) -> Dict[str, str]:
        """
        Scan key project files and return their content.
        
        Returns:
            Dict mapping filename to content (truncated if too large)
        """
        root = Path(workspace_root)
        scanned = {}
        
        # Scan specific files
        for filename in PROJECT_SCAN_FILES:
            filepath = root / filename
            if filepath.exists() and filepath.is_file():
                content = self._read_file_safe(filepath)
                if content:
                    scanned[filename] = content
        
        # Scan entry points in source directories
        for dirname in PROJECT_SCAN_DIRS:
            dirpath = root / dirname
            if dirpath.exists() and dirpath.is_dir():
                # Find key files in this directory
                for pattern in ["*.py", "*.ts", "*.js", "*.go", "*.rs"]:
                    for filepath in dirpath.glob(pattern):
                        # Only include entry-point-like files
                        name = filepath.name.lower()
                        if any(kw in name for kw in ["main", "server", "app", "init", "index", "entry", "cli", "api", "tools", "registry", "base", "engine", "orchestrator"]):
                            content = self._read_file_safe(filepath)
                            if content:
                                rel_path = filepath.relative_to(root)
                                scanned[str(rel_path)] = content
        
        # Get directory structure
        structure = self._get_directory_structure(root)
        scanned["_directory_structure"] = structure
        
        return scanned
    
    def _scan_data_files(self, workspace_root: str) -> Dict[str, str]:
        """
        Scan data-related files and return their content.
        
        Returns:
            Dict mapping filename to content or metadata
        """
        root = Path(workspace_root)
        scanned = {}
        
        # Check for database files (don't read content, just note existence)
        for pattern in DATA_SCAN_FILES:
            for filepath in root.glob(f"**/{pattern}"):
                rel_path = filepath.relative_to(root)
                size = filepath.stat().st_size if filepath.exists() else 0
                scanned[str(rel_path)] = f"[{filepath.suffix} file, {size} bytes]"
        
        # Scan data-related Python files
        data_keywords = ["data", "etl", "pipeline", "schema", "model", "db", "database", "storage", "api", "client"]
        for dirname in PROJECT_SCAN_DIRS:
            dirpath = root / dirname
            if dirpath.exists():
                for filepath in dirpath.glob("**/*.py"):
                    name = filepath.name.lower()
                    if any(kw in name for kw in data_keywords):
                        content = self._read_file_safe(filepath)
                        if content:
                            rel_path = filepath.relative_to(root)
                            scanned[str(rel_path)] = content
        
        return scanned
    
    def _read_file_safe(self, filepath: Path) -> Optional[str]:
        """
        Safely read a file with size and line limits.
        
        Returns:
            File content (truncated) or None if unreadable
        """
        try:
            size = filepath.stat().st_size
            if size > MAX_FILE_SIZE:
                logger.warning(f"File too large: {filepath} ({size} bytes)")
                return None
            
            content = filepath.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            
            if len(lines) > MAX_FILE_LINES:
                # Truncate to first N lines
                content = "\n".join(lines[:MAX_FILE_LINES])
                content += f"\n... [truncated, {len(lines)} total lines]"
            
            return content
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            return None
    
    def _get_directory_structure(self, root: Path, max_depth: int = 3) -> str:
        """
        Generate a tree-style directory structure.
        
        Returns:
            String representation of directory tree
        """
        lines = []
        
        def walk(path: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return
            
            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            except PermissionError:
                return
            
            # Skip hidden and common ignore directories
            items = [i for i in items if not i.name.startswith(".") and i.name not in ["__pycache__", "node_modules", ".git", "venv", ".venv"]]
            
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                marker = "└── " if is_last else "├── "
                lines.append(f"{prefix}{marker}{item.name}")
                
                if item.is_dir():
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    walk(item, new_prefix, depth + 1)
        
        lines.append(f"{root.name}/")
        walk(root)
        return "\n".join(lines[:50])  # Limit output
    
    async def _generate_single_context(
        self,
        workspace_root: str,
        context_type: str,
        complexity: str = "medium"
    ) -> str:
        """Generate a single context file content with actual file data."""
        
        # SCAN ACTUAL FILES - this is the fix!
        if context_type == "project":
            system_prompt = PROJECT_CONTEXT_SYSTEM_PROMPT
            file_data = self._scan_project_files(workspace_root)
            
            # Build prompt with actual file content
            file_sections = []
            for filename, content in file_data.items():
                if filename == "_directory_structure":
                    file_sections.append(f"### Directory Structure\n```\n{content}\n```")
                else:
                    file_sections.append(f"### {filename}\n```\n{content}\n```")
            
            files_block = "\n\n".join(file_sections[:15])  # Limit to avoid token overflow
            
            analysis_prompt = f"""
Analyze the ACTUAL project files below and generate _PROJECT_CONTEXT.md.

CRITICAL: Use ONLY the provided file content. DO NOT hallucinate generic defaults like "Flask" or "PostgreSQL" if not present.

## Scanned Project Files:

{files_block}

## Instructions:
1. Extract tech stack from imports and dependencies (look for FastMCP, DashScope, DuckDB, etc.)
2. Identify entry points from actual file names (server.py, tools.py, etc.)
3. Map directory structure from the provided tree
4. Extract conventions from AGENTS.md or README.md if present

Return the complete _PROJECT_CONTEXT.md content based on ACTUAL evidence.
"""
        elif context_type == "data":
            system_prompt = DATA_CONTEXT_SYSTEM_PROMPT
            file_data = self._scan_data_files(workspace_root)
            
            # Build prompt with actual file content
            file_sections = []
            for filename, content in file_data.items():
                file_sections.append(f"### {filename}\n```\n{content}\n```")
            
            files_block = "\n\n".join(file_sections[:10])
            
            analysis_prompt = f"""
Analyze the ACTUAL data infrastructure files below and generate _DATA_CONTEXT.md.

CRITICAL: Use ONLY the provided file content. DO NOT hallucinate generic schemas.

## Scanned Data Files:

{files_block}

## Instructions:
1. Identify database types from actual files (DuckDB, SQLite, etc.)
2. Extract schemas from model files if present
3. Find API clients and data connectors from imports
4. Note any ETL/pipeline scripts

Return the complete _DATA_CONTEXT.md content based on ACTUAL evidence.
"""
        else:
            raise ValueError(f"Unknown context type: {context_type}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": analysis_prompt}
        ]
        
        response = await self.client.generate_completion(
            messages=messages,
            task_type="context_analysis",
            complexity=complexity,
            tags=["context_builder"]
        )
        
        return response
    
    async def update_session_context(
        self,
        session_summary: str,
        workspace_root: str = "."
    ) -> str:
        """
        Update or create _SESSION_SUPPLEMENT.md.
        
        Args:
            session_summary: Summary of current session
            workspace_root: Path to workspace
        
        Returns:
            Updated session supplement content
        """
        self.context_dir.mkdir(exist_ok=True, parents=True)
        session_file = self.context_dir / "_SESSION_SUPPLEMENT.md"
        
        # Read existing content if present
        previous_content = ""
        if session_file.exists():
            previous_content = session_file.read_text(encoding="utf-8")
        
        # Generate update
        messages = [
            {"role": "system", "content": SESSION_SUPPLEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"""
Previous session context (if empty, this is first session):
{previous_content or "No previous sessions."}

Current session summary:
{session_summary}

Workspace: {workspace_root}

Generate updated _SESSION_SUPPLEMENT.md preserving previous sessions as history.
"""}
        ]
        
        response = await self.client.generate_completion(
            messages=messages,
            task_type="context_analysis",
            complexity="low",
            tags=["session_context"]
        )
        
        return response
    
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
        self.context_dir.mkdir(exist_ok=True, parents=True)
        
        saved_files = {}
        
        # Save project context
        project_path = self.context_dir / "_PROJECT_CONTEXT.md"
        self._atomic_write(project_path, project_context)
        saved_files["project"] = project_path
        
        # Save data context
        data_path = self.context_dir / "_DATA_CONTEXT.md"
        self._atomic_write(data_path, data_context)
        saved_files["data"] = data_path
        
        logger.info(f"Context files saved to {self.context_dir}")
        return saved_files
    
    def save_session_context(
        self,
        session_content: str,
        workspace_root: str = "."
    ) -> Path:
        """Atomically save session supplement."""
        self.context_dir.mkdir(exist_ok=True, parents=True)
        
        session_path = self.context_dir / "_SESSION_SUPPLEMENT.md"
        self._atomic_write(session_path, session_content)
        
        logger.info(f"Session context saved to {session_path}")
        return session_path
    
    def _atomic_write(self, path: Path, content: str) -> None:
        """
        Write file atomically using temp + rename pattern.
        
        This prevents corruption if the process is interrupted during write.
        Matches the pattern from SessionStore.save().
        """
        # Write to temp file first
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            dir=self.context_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Atomic rename
            os.replace(temp_path, str(path))
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise