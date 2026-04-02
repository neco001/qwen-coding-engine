"""Functional Snapshot Generator.

Captures and compares functional snapshots of code to detect regressions.
"""

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple


class FunctionalSnapshotGenerator:
    """Generate and compare functional snapshots of code."""
    
    async def capture_snapshot(
        self, project_dir: Path, patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Capture a functional snapshot of a project.
        
        Args:
            project_dir: Root directory of the project
            patterns: Optional list of glob patterns to match files
            
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
        }
        
        if not project_dir.exists():
            return snapshot
        
        # Find Python files
        if patterns:
            python_files = []
            for pattern in patterns:
                python_files.extend(project_dir.glob(pattern))
        else:
            python_files = list(project_dir.rglob("*.py"))
        
        for file_path in python_files:
            try:
                file_snapshot = self._capture_file_snapshot(file_path)
                snapshot["files"][str(file_path)] = file_snapshot
                snapshot["functions"].extend(file_snapshot.get("functions", []))
                snapshot["classes"].extend(file_snapshot.get("classes", []))
            except (SyntaxError, IOError, OSError):
                continue
        
        # Extract mappings (function calls between modules)
        snapshot["mappings"] = self._extract_mappings(snapshot["files"])
        
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
