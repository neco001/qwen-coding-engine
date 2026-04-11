"""Functional Snapshot Generator.

Captures and compares functional snapshots of code to detect regressions.
"""

import ast
import asyncio
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from qwen_mcp.anti_degradation_config import get_config


@dataclass
class ContentHash:
    """Semantic content hash for White Cell verification."""
    file_path: str
    hash_value: str
    hash_algorithm: str = "sha256"
    semantic_markers: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "hash_value": self.hash_value,
            "hash_algorithm": self.hash_algorithm,
            "semantic_markers": self.semantic_markers,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentHash":
        created_at = data.get("created_at", datetime.now(timezone.utc).isoformat())
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return cls(
            file_path=data["file_path"],
            hash_value=data["hash_value"],
            hash_algorithm=data.get("hash_algorithm", "sha256"),
            semantic_markers=data.get("semantic_markers", []),
            created_at=created_at,
        )


class FunctionalSnapshotGenerator:
    """Generate and compare functional snapshots of code."""
    
    def __init__(self, shadow_mode: bool = False, storage_dir: Optional[str] = None):
        """Initialize snapshot generator.
        
        Args:
            shadow_mode: If True, warnings only - no blocking (mandatory before production)
            storage_dir: Directory for storing snapshots. Defaults to config.snapshots.storage_dir
        """
        self.shadow_mode = shadow_mode
        self._git_available: Optional[bool] = None
        
        if storage_dir is None:
            config = get_config()
            self.storage_dir = config.snapshots.storage_dir
        else:
            self.storage_dir = storage_dir
    
    def _check_git_available(self) -> bool:
        """Check if git is available in environment."""
        if self._git_available is not None:
            return self._git_available
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True, timeout=5, stdin=subprocess.DEVNULL)
            self._git_available = True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self._git_available = False
        return self._git_available
    
    def _get_changed_files(
        self,
        project_dir: Path,
        commit_range: str = "HEAD~1..HEAD"
    ) -> List[Path]:
        """Get Python files changed in git diff.
        
        Args:
            project_dir: Root directory
            commit_range: Git diff range (e.g., "HEAD~1..HEAD")
            
        Returns:
            List of changed Python file paths
        """
        if not self._check_git_available():
            return list(project_dir.rglob("*.py"))
        
        try:
            result = subprocess.run(
                ["git", "--no-pager", "diff", "--name-only", commit_range],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL
            )
            
            if result.returncode != 0:
                return list(project_dir.rglob("*.py"))
            
            changed_files = [
                project_dir / f
                for f in result.stdout.splitlines()
                if f.endswith(".py") and (project_dir / f).exists()
            ]
            
            return changed_files if changed_files else list(project_dir.rglob("*.py"))
            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return list(project_dir.rglob("*.py"))
    
    def _get_tracked_files(
        self,
        project_dir: Path,
        file_pattern: str = "*.py"
    ) -> List[Path]:
        """Get Python files respecting .gitignore and git exclude rules.
        
        Uses 'git ls-files --exclude-standard' which respects:
        - .gitignore files at all levels
        - .git/info/exclude
        - core.excludesFile (global git exclude)
        
        Args:
            project_dir: Root directory of the project
            file_pattern: File pattern to filter (default: "*.py")
            
        Returns:
            List of Python file paths respecting git ignore rules
        """
        if not self._check_git_available():
            return list(project_dir.rglob(file_pattern))
        
        try:
            result = subprocess.run(
                ["git", "ls-files", "--exclude-standard", "--cached", "--others"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=30,
                stdin=subprocess.DEVNULL
            )
            
            if result.returncode != 0:
                return list(project_dir.rglob(file_pattern))
            
            pattern_suffix = file_pattern.lstrip("*.")
            tracked_files = [
                project_dir / f
                for f in result.stdout.splitlines()
                if f.endswith(f".{pattern_suffix}") and (project_dir / f).exists()
            ]
            
            return tracked_files if tracked_files else list(project_dir.rglob(file_pattern))
            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return list(project_dir.rglob(file_pattern))
    
    async def capture_snapshot(
        self,
        project_dir: Path,
        patterns: Optional[List[str]] = None,
        commit_range: Optional[str] = None,
        changed_files: Optional[List[Path]] = None
    ) -> Dict[str, Any]:
        """Capture a functional snapshot of a project.
        
        Args:
            project_dir: Root directory of the project
            patterns: Optional list of glob patterns to match files
            commit_range: Git diff range (e.g., "HEAD~1..HEAD") for changed file detection
            changed_files: Optional list of pre-computed changed files
            
        Returns:
            Snapshot dictionary with functions, classes, and mappings
        """
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "project_dir": str(project_dir),
            "files": {},
            "functions": [],
            "classes": [],
            "mappings": {},
            "content_hashes": [],
            "shadow_mode": self.shadow_mode,
        }
        
        if not project_dir.exists():
            return snapshot
        
        # Find Python files - use changed files if provided
        if changed_files:
            python_files = changed_files
        elif commit_range:
            python_files = await asyncio.to_thread(self._get_changed_files, project_dir, commit_range)
        elif patterns:
            python_files = []
            for pattern in patterns:
                python_files.extend(project_dir.glob(pattern))
        else:
            python_files = await asyncio.to_thread(self._get_tracked_files, project_dir)
        
        # Parallel file capture with asyncio.gather
        async def _capture_file_snapshot_async(file_path: Path) -> Tuple[Path, Optional[Dict[str, Any]]]:
            """Async wrapper for _capture_file_snapshot."""
            try:
                result = await asyncio.to_thread(self._capture_file_snapshot, file_path)
                return (file_path, result)
            except (SyntaxError, IOError, OSError):
                return (file_path, None)
        
        tasks = [
            _capture_file_snapshot_async(file_path)
            for file_path in python_files
        ]
        results = []
        chunk_size = 20
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
            results.extend(chunk_results)
            await asyncio.sleep(0.01)  # Force yield to let MCP server respond to pings
        
        for file_path, file_snapshot in results:
            if file_snapshot is not None:
                snapshot["files"][str(file_path)] = file_snapshot
                snapshot["functions"].extend(file_snapshot.get("functions", []))
                snapshot["classes"].extend(file_snapshot.get("classes", []))
        
        # Extract mappings (function calls between modules)
        snapshot["mappings"] = await asyncio.to_thread(self._extract_mappings, snapshot["files"])
        
        # Generate content hashes for semantic verification (White Cell requirement)
        snapshot["content_hashes"] = await self._generate_content_hashes(
            project_dir,
            patterns,
            changed_files=python_files if changed_files else None
        )
        
        return snapshot
    
    def _capture_file_snapshot(self, file_path: Path) -> Dict[str, Any]:
        """Capture snapshot of a single file."""
        result = {
            "functions": [],
            "classes": [],
            "imports": [],
        }
        
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
        
        # Extract functions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result["functions"].append({
                    "name": node.name,
                    "signature": self._get_function_signature(node),
                    "lineno": node.lineno,
                    "docstring": ast.get_docstring(node),
                    "file": str(file_path),
                })
            
            elif isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append({
                            "name": item.name,
                            "signature": self._get_function_signature(item),
                            "lineno": item.lineno,
                        })
                
                result["classes"].append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "docstring": ast.get_docstring(node),
                    "methods": methods,
                    "file": str(file_path),
                })
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append({
                        "module": alias.name,
                        "alias": alias.asname,
                    })
            
            elif isinstance(node, ast.ImportFrom):
                result["imports"].append({
                    "module": node.module or "",
                    "names": [alias.name for alias in node.names],
                })
        
        return result
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get a string representation of a function signature."""
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            args.append(arg.arg)
        
        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        return f"{node.name}({', '.join(args)})"
    
    def _extract_mappings(self, files: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract function-to-function mappings from file snapshots."""
        mappings = {}
        
        # Build a map of function names to files
        func_files = {}
        for file_path, file_snapshot in files.items():
            for func in file_snapshot.get("functions", []):
                func_files[func["name"]] = file_path
            for cls in file_snapshot.get("classes", []):
                func_files[cls["name"]] = file_path
        
        # Find function calls
        for file_path, file_snapshot in files.items():
            try:
                source = Path(file_path).read_text(encoding="utf-8")
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            func_name = node.func.id
                            if func_name in func_files and func_files[func_name] != file_path:
                                # Cross-module call
                                caller = self._find_enclosing_function(node, tree)
                                if caller:
                                    key = f"{file_path}:{caller}"
                                    if key not in mappings:
                                        mappings[key] = []
                                    mappings[key].append(func_name)
            except (SyntaxError, IOError, OSError):
                continue
        
        return mappings
    
    def _find_enclosing_function(
        self, node: ast.AST, tree: ast.AST
    ) -> Optional[str]:
        """Find the name of the function containing a node."""
        # Simple approach: walk from root and track function context
        for parent in ast.walk(tree):
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(parent):
                    if child is node:
                        return parent.name
        return None
    
    async def compare_snapshots(
        self, before: Dict[str, Any], after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two snapshots and return differences.
        
        Args:
            before: Snapshot before changes
            after: Snapshot after changes
            
        Returns:
            Diff dictionary with added, removed, and modified items
        """
        diff = {
            "added_functions": [],
            "removed_functions": [],
            "modified_functions": [],
            "added_classes": [],
            "removed_classes": [],
            "modified_classes": [],
            "added_mappings": [],
            "removed_mappings": [],
        }
        
        # Compare functions
        before_funcs = {f["name"]: f for f in before.get("functions", [])}
        after_funcs = {f["name"]: f for f in after.get("functions", [])}
        
        before_names = set(before_funcs.keys())
        after_names = set(after_funcs.keys())
        
        # Added functions
        for name in after_names - before_names:
            diff["added_functions"].append(after_funcs[name])
        
        # Removed functions
        for name in before_names - after_names:
            diff["removed_functions"].append(before_funcs[name])
        
        # Modified functions (signature changed)
        for name in before_names & after_names:
            if before_funcs[name]["signature"] != after_funcs[name]["signature"]:
                diff["modified_functions"].append({
                    "name": name,
                    "before": before_funcs[name]["signature"],
                    "after": after_funcs[name]["signature"],
                })
        
        # Compare classes
        before_classes = {c["name"]: c for c in before.get("classes", [])}
        after_classes = {c["name"]: c for c in after.get("classes", [])}
        
        before_class_names = set(before_classes.keys())
        after_class_names = set(after_classes.keys())
        
        # Added classes
        for name in after_class_names - before_class_names:
            diff["added_classes"].append(after_classes[name])
        
        # Removed classes
        for name in before_class_names - after_class_names:
            diff["removed_classes"].append(before_classes[name])
        
        # Compare mappings
        before_mappings = before.get("mappings", {})
        after_mappings = after.get("mappings", {})
        
        before_map_keys = set(before_mappings.keys())
        after_map_keys = set(after_mappings.keys())
        
        # Added mappings
        for key in after_map_keys - before_map_keys:
            diff["added_mappings"].append({
                "caller": key,
                "callees": after_mappings[key],
            })
        
        # Removed mappings
        for key in before_map_keys - after_map_keys:
            diff["removed_mappings"].append({
                "caller": key,
                "callees": before_mappings[key],
            })
        
        return diff
    
    async def detect_regression(
        self, diff: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect regressions from a snapshot diff.
        
        Args:
            diff: Result from compare_snapshots()
            
        Returns:
            List of regression alerts
        """
        alerts = []
        
        # Removed functions are potential regressions
        for func in diff.get("removed_functions", []):
            alerts.append({
                "type": "function_removed",
                "severity": "warning",
                "name": func["name"],
                "message": f"Function '{func['name']}' was removed",
            })
        
        # Removed classes are potential regressions
        for cls in diff.get("removed_classes", []):
            alerts.append({
                "type": "class_removed",
                "severity": "warning",
                "name": cls["name"],
                "message": f"Class '{cls['name']}' was removed",
            })
        
        # Modified function signatures might break callers
        for func in diff.get("modified_functions", []):
            alerts.append({
                "type": "signature_changed",
                "severity": "info",
                "name": func["name"],
                "message": f"Function '{func['name']}' signature changed: {func['before']} -> {func['after']}",
            })
        
        # Removed mappings might indicate broken dependencies
        for mapping in diff.get("removed_mappings", []):
            alerts.append({
                "type": "dependency_removed",
                "severity": "info",
                "caller": mapping["caller"],
                "message": f"Dependency from {mapping['caller']} to {mapping['callees']} was removed",
            })
        
        return alerts
    
    async def _generate_content_hashes(
        self,
        project_dir: Path,
        patterns: Optional[List[str]] = None,
        changed_files: Optional[List[Path]] = None
    ) -> List[Dict[str, Any]]:
        """Generate content hashes for tracked files (White Cell verification).
        
        Args:
            project_dir: Root directory of the project
            patterns: Optional list of glob patterns to match files
            changed_files: Optional list of files to hash (optimization)
            
        Returns:
            List of content hash dictionaries
        """
        hashes = []
        
        # If changed_files provided, only hash those
        if changed_files:
            for file_path in changed_files:
                if file_path.is_file() and ".snapshots" not in str(file_path) and ".venv" not in str(file_path):
                    try:
                        content_hash = await asyncio.to_thread(self._compute_content_hash, file_path, project_dir)
                        hashes.append(content_hash.to_dict())
                        await asyncio.sleep(0)  # Yield to event loop
                    except (IOError, OSError):
                        continue
            return hashes
        
        # Use git-aware file discovery instead of glob patterns to respect .gitignore
        if patterns is None:
            patterns = []
        # Get git-tracked files to avoid .venv inclusion
        tracked_files = self._get_tracked_files(project_dir)
        for file_path in tracked_files:
            if file_path.is_file() and ".snapshots" not in str(file_path):
                try:
                    content_hash = await asyncio.to_thread(self._compute_content_hash, file_path, project_dir)
                    hashes.append(content_hash.to_dict())
                    await asyncio.sleep(0)  # Yield to event loop
                except (IOError, OSError):
                    continue
        
        return hashes
    
    def _compute_content_hash(self, file_path: Path, project_dir: Path) -> ContentHash:
        """Compute semantic content hash for a file.
        
        Args:
            file_path: Path to the file
            project_dir: Root directory for relative path calculation
            
        Returns:
            ContentHash object with hash value and semantic markers
        """
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            
            # Extract semantic markers (function defs, class defs, imports)
            semantic_markers = self._extract_semantic_markers(content)
            
            # Compute hash of content + semantic markers for semantic verification
            hash_input = content + b"".join([m.encode() for m in sorted(semantic_markers)])
            hash_value = hashlib.sha256(hash_input).hexdigest()
            
            return ContentHash(
                file_path=str(file_path.relative_to(project_dir)),
                hash_value=hash_value,
                semantic_markers=semantic_markers,
            )
        except Exception as e:
            # Return error hash but don't fail snapshot generation
            return ContentHash(
                file_path=str(file_path),
                hash_value="ERROR",
                semantic_markers=[],
            )
    
    def _extract_semantic_markers(self, content: bytes) -> List[str]:
        """Extract semantic markers from file content for White Cell verification.
        
        Args:
            content: Raw file content
            
        Returns:
            List of semantic markers (func:name, class:name, import:module)
        """
        try:
            text = content.decode("utf-8", errors="ignore")
            markers = []
            
            # Extract function definitions
            import re
            func_pattern = r"^\s*def\s+(\w+)\s*\("
            for match in re.finditer(func_pattern, text, re.MULTILINE):
                markers.append(f"func:{match.group(1)}")
            
            # Extract class definitions
            class_pattern = r"^\s*class\s+(\w+)"
            for match in re.finditer(class_pattern, text, re.MULTILINE):
                markers.append(f"class:{match.group(1)}")
            
            # Extract import statements
            import_pattern = r"^(?:from|import)\s+([\w.]+)"
            for match in re.finditer(import_pattern, text, re.MULTILINE):
                markers.append(f"import:{match.group(1)}")
            
            return markers[:50]  # Limit to prevent bloat
        except Exception:
            return []
    
    def compare_content_hashes(
        self, before_hashes: List[Dict[str, Any]], after_hashes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare content hashes between snapshots.
        
        Args:
            before_hashes: Content hashes from before snapshot
            after_hashes: Content hashes from after snapshot
            
        Returns:
            Dictionary with hash changes analysis
        """
        before_map = {h["file_path"]: h for h in before_hashes}
        after_map = {h["file_path"]: h for h in after_hashes}
        
        changes = {
            "added_files": [],
            "removed_files": [],
            "modified_files": [],
            "semantic_changes": [],
        }
        
        before_paths = set(before_map.keys())
        after_paths = set(after_map.keys())
        
        # Added files
        for path in after_paths - before_paths:
            changes["added_files"].append({
                "file": path,
                "hash": after_map[path]["hash_value"][:16],
            })
        
        # Removed files
        for path in before_paths - after_paths:
            changes["removed_files"].append({
                "file": path,
                "hash": before_map[path]["hash_value"][:16],
            })
        
        # Modified files
        for path in before_paths & after_paths:
            before_hash = before_map[path]["hash_value"]
            after_hash = after_map[path]["hash_value"]
            
            if before_hash != after_hash:
                # Check semantic markers change
                before_markers = set(before_map[path].get("semantic_markers", []))
                after_markers = set(after_map[path].get("semantic_markers", []))
                
                semantic_changed = before_markers != after_markers
                
                changes["modified_files"].append({
                    "file": path,
                    "old_hash": before_hash[:16],
                    "new_hash": after_hash[:16],
                    "semantic_markers_changed": semantic_changed,
                })
                
                if semantic_changed:
                    changes["semantic_changes"].append({
                        "file": path,
                        "added_markers": list(after_markers - before_markers),
                        "removed_markers": list(before_markers - after_markers),
                    })
        
        return changes
    
    def save_snapshot(self, snapshot: Dict[str, Any], project_dir: Path, name: str = "latest") -> Path:
        """Save snapshot to disk.
        
        Args:
            snapshot: Snapshot dictionary
            project_dir: Project root directory
            name: Snapshot name
            
        Returns:
            Path to saved snapshot file
        """
        snapshots_dir = project_dir / self.storage_dir
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_path = snapshots_dir / f"{name}.json"
        
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        
        return snapshot_path
    
    def load_snapshot(self, project_dir: Path, name: str = "latest") -> Optional[Dict[str, Any]]:
        """Load snapshot from disk.
        
        Args:
            project_dir: Project root directory
            name: Snapshot name
            
        Returns:
            Snapshot dictionary or None if not found
        """
        snapshot_path = project_dir / self.storage_dir / f"{name}.json"
        
        if not snapshot_path.exists():
            return None
        
        with open(snapshot_path, "r", encoding="utf-8") as f:
            return json.load(f)
