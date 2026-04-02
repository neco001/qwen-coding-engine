"""Dependency Tracker for Python projects.

Analyzes import dependencies between files in a project.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple


class DependencyTracker:
    """Track dependencies between Python files in a project."""
    
    def __init__(self):
        """Initialize the dependency tracker."""
        self._file_modules: Dict[Path, Set[str]] = {}
        self._module_files: Dict[str, Path] = {}
        self._dependencies: List[Dict[str, Any]] = []
    
    async def analyze_project(self, project_dir: Path) -> Dict[str, Any]:
        """Analyze all Python files in a project directory.
        
        Args:
            project_dir: Root directory of the project
            
        Returns:
            Dict with 'files' and 'dependencies' keys
        """
        result = {
            "files": {},
            "dependencies": [],
        }
        
        if not project_dir.exists():
            return result
        
        # Find all Python files
        python_files = list(project_dir.rglob("*.py"))
        
        if not python_files:
            return result
        
        # Parse each file and extract imports
        for file_path in python_files:
            try:
                imports = self._extract_imports(file_path)
                module_name = self._path_to_module(file_path, project_dir)
                
                self._file_modules[file_path] = set(imports)
                self._module_files[module_name] = file_path
                
                result["files"][str(file_path)] = {
                    "module": module_name,
                    "imports": imports,
                }
            except (SyntaxError, IOError, OSError):
                continue
        
        # Build dependency graph
        self._dependencies = self._build_dependencies(project_dir)
        result["dependencies"] = self._dependencies
        
        return result
    
    def _extract_imports(self, file_path: Path) -> List[str]:
        """Extract import statements from a Python file."""
        imports = []
        
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, IOError, OSError):
            return imports
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    
    def _path_to_module(self, file_path: Path, project_dir: Path) -> str:
        """Convert a file path to a module name."""
        try:
            relative_path = file_path.relative_to(project_dir)
        except ValueError:
            relative_path = file_path
        
        # Remove .py extension
        parts = list(relative_path.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        
        # Handle __init__.py
        if parts[-1] == "__init__":
            parts = parts[:-1]
        
        return ".".join(parts) if parts else ""
    
    def _build_dependencies(self, project_dir: Path) -> List[Dict[str, Any]]:
        """Build dependency relationships between files."""
        dependencies = []
        
        for file_path, imports in self._file_modules.items():
            for import_name in imports:
                # Check if this import corresponds to a file in the project
                target_file = self._resolve_import(import_name, file_path, project_dir)
                
                if target_file:
                    dependencies.append({
                        "from_file": str(file_path),
                        "to_file": str(target_file),
                        "import": import_name,
                    })
        
        return dependencies
    
    def _resolve_import(
        self, import_name: str, from_file: Path, project_dir: Path
    ) -> Optional[Path]:
        """Resolve an import name to a file path."""
        # Check if it's a direct module match
        if import_name in self._module_files:
            return self._module_files[import_name]
        
        # Check if it's a submodule
        for module_name, file_path in self._module_files.items():
            if module_name.startswith(import_name + ".") or module_name.endswith("." + import_name):
                return file_path
        
        # Check if it's a relative import
        from_module = self._path_to_module(from_file, project_dir)
        if from_module:
            parent_parts = from_module.split(".")[:-1]
            relative_module = ".".join(parent_parts + [import_name])
            if relative_module in self._module_files:
                return self._module_files[relative_module]
        
        return None
    
    def get_dependents(self, file_path: Path) -> List[Path]:
        """Get all files that depend on the given file.
        
        Args:
            file_path: The file to find dependents for
            
        Returns:
            List of file paths that depend on the given file
        """
        dependents = []
        module_name = None
        
        # Find the module name for this file
        for mod_name, mod_file in self._module_files.items():
            if mod_file == file_path:
                module_name = mod_name
                break
        
        if not module_name:
            return dependents
        
        # Find all files that import this module
        for file_path, imports in self._file_modules.items():
            if module_name in imports:
                dependents.append(file_path)
        
        return dependents
    
    def get_dependencies(self, file_path: Path) -> List[Path]:
        """Get all files that the given file depends on.
        
        Args:
            file_path: The file to find dependencies for
            
        Returns:
            List of file paths that the given file depends on
        """
        dependencies = []
        
        if file_path not in self._file_modules:
            return dependencies
        
        for import_name in self._file_modules[file_path]:
            target_file = self._resolve_import_to_file(import_name)
            if target_file:
                dependencies.append(target_file)
        
        return dependencies
    
    def _resolve_import_to_file(self, import_name: str) -> Optional[Path]:
        """Resolve an import name to a file path."""
        if import_name in self._module_files:
            return self._module_files[import_name]
        
        # Check for submodules
        for module_name, file_path in self._module_files.items():
            if module_name.startswith(import_name + "."):
                return file_path
        
        return None
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the project.
        
        Returns:
            List of cycles, where each cycle is a list of module names
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(module: str) -> bool:
            visited.add(module)
            rec_stack.add(module)
            path.append(module)
            
            # Get dependencies for this module
            file_path = self._module_files.get(module)
            if file_path:
                imports = self._file_modules.get(file_path, set())
                for imp in imports:
                    if imp not in visited:
                        if dfs(imp):
                            return True
                    elif imp in rec_stack:
                        # Found a cycle
                        cycle_start = path.index(imp)
                        cycle = path[cycle_start:] + [imp]
                        cycles.append(cycle)
                        return True
            
            path.pop()
            rec_stack.remove(module)
            return False
        
        for module in self._module_files:
            if module not in visited:
                dfs(module)
        
        return cycles
