"""Static AST Parser for Python files.

Uses Python's ast module to extract functions, classes, and imports.
"""

import ast
from pathlib import Path
from typing import Dict, List, Any, Optional


class StaticASTParser:
    """Parse Python files using AST to extract structural information."""
    
    async def parse_file(self, file_path: Path) -> Dict[str, List[Any]]:
        """Parse a Python file and extract module information.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dict with keys: functions, classes, imports
            Each containing a list of extracted information
        """
        result = {
            "functions": [],
            "classes": [],
            "imports": [],
        }
        
        try:
            source = file_path.read_text(encoding="utf-8")
        except (FileNotFoundError, IOError, OSError):
            return result
        
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return result
        
        # Extract imports
        result["imports"] = self._extract_imports(tree)
        
        # Extract functions (top-level only)
        result["functions"] = self._extract_functions(tree)
        
        # Extract classes
        result["classes"] = self._extract_classes(tree)
        
        return result
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "module": alias.name,
                        "alias": alias.asname,
                        "lineno": node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "lineno": node.lineno,
                    })
        
        return imports
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract top-level function definitions from AST."""
        functions = []
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                functions.append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "args": self._extract_args(node.args),
                    "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                    "docstring": ast.get_docstring(node),
                })
        
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class definitions from AST."""
        classes = []
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append({
                            "name": item.name,
                            "lineno": item.lineno,
                            "args": self._extract_args(item.args),
                            "decorators": [self._get_decorator_name(d) for d in item.decorator_list],
                        })
                
                classes.append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "bases": [self._get_base_name(base) for base in node.bases],
                    "methods": methods,
                    "docstring": ast.get_docstring(node),
                })
        
        return classes
    
    def _extract_args(self, args: ast.arguments) -> List[str]:
        """Extract argument names from function arguments."""
        arg_names = []
        
        # Regular arguments
        for arg in args.args:
            arg_names.append(arg.arg)
        
        # *args
        if args.vararg:
            arg_names.append(f"*{args.vararg.arg}")
        
        # **kwargs
        if args.kwarg:
            arg_names.append(f"**{args.kwarg.arg}")
        
        return arg_names
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get the name of a decorator."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return self._get_attribute_name(decorator)
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        else:
            return "unknown"
    
    def _get_attribute_name(self, attr: ast.Attribute) -> str:
        """Get full name of an attribute access (e.g., module.decorator)."""
        parts = []
        current = attr
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    
    def _get_base_name(self, base: ast.AST) -> str:
        """Get the name of a base class."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return self._get_attribute_name(base)
        else:
            return "unknown"
