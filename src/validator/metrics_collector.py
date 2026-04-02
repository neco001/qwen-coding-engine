"""Metrics Collector for AI-Driven Testing System."""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ChangeMetrics:
    """Metrics representing code changes."""
    total_files: int = 0
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_imports: int = 0


class MetricsCollector:
    """Collects and compares metrics for code changes."""
    
    async def collect_metrics(self, project_dir: Path) -> ChangeMetrics:
        """Collect metrics from a project directory.
        
        Args:
            project_dir: Path to the project directory.
            
        Returns:
            ChangeMetrics object with collected metrics.
        """
        metrics = ChangeMetrics()
        
        python_files = list(project_dir.rglob("*.py"))
        metrics.total_files = len(python_files)
        
        for file_path in python_files:
            file_metrics = self._collect_file_metrics(file_path)
            metrics.total_lines += file_metrics.total_lines
            metrics.total_functions += file_metrics.total_functions
            metrics.total_classes += file_metrics.total_classes
            metrics.total_imports += file_metrics.total_imports
        
        return metrics
    
    def _collect_file_metrics(self, file_path: Path) -> ChangeMetrics:
        """Collect metrics from a single Python file.
        
        Args:
            file_path: Path to the Python file.
            
        Returns:
            ChangeMetrics object with file-level metrics.
        """
        metrics = ChangeMetrics(total_files=1)
        
        try:
            source = file_path.read_text(encoding="utf-8")
            metrics.total_lines = len(source.splitlines())
            
            tree = ast.parse(source, filename=str(file_path))
            metrics.total_functions = self._count_functions(tree)
            metrics.total_classes = self._count_classes(tree)
            metrics.total_imports = self._count_imports(tree)
        except (SyntaxError, FileNotFoundError):
            pass
        
        return metrics
    
    def _count_functions(self, tree: ast.AST) -> int:
        """Count function definitions in an AST.
        
        Args:
            tree: Parsed AST.
            
        Returns:
            Number of function definitions.
        """
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                count += 1
        return count
    
    def _count_classes(self, tree: ast.AST) -> int:
        """Count class definitions in an AST.
        
        Args:
            tree: Parsed AST.
            
        Returns:
            Number of class definitions.
        """
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                count += 1
        return count
    
    def _count_imports(self, tree: ast.AST) -> int:
        """Count import statements in an AST.
        
        Args:
            tree: Parsed AST.
            
        Returns:
            Number of import statements.
        """
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                count += 1
        return count
    
    def compare_metrics(self, before: ChangeMetrics, after: ChangeMetrics) -> Dict[str, Any]:
        """Compare two sets of metrics and return differences.
        
        Args:
            before: Metrics before changes.
            after: Metrics after changes.
            
        Returns:
            Dictionary with metric differences.
        """
        diff = {
            "files_added": after.total_files - before.total_files,
            "lines_added": after.total_lines - before.total_lines,
            "functions_added": after.total_functions - before.total_functions,
            "classes_added": after.total_classes - before.total_classes,
        }
        
        # Track removals separately
        if diff["files_added"] < 0:
            diff["files_removed"] = abs(diff["files_added"])
            diff["files_added"] = 0
        
        if diff["lines_added"] < 0:
            diff["lines_removed"] = abs(diff["lines_added"])
            diff["lines_added"] = 0
        
        if diff["functions_added"] < 0:
            diff["functions_removed"] = abs(diff["functions_added"])
            diff["functions_added"] = 0
        
        if diff["classes_added"] < 0:
            diff["classes_removed"] = abs(diff["classes_added"])
            diff["classes_added"] = 0
        
        return diff
