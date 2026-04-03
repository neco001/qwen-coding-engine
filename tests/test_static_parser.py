"""Tests for Static AST Parser."""
import pytest
import tempfile
from pathlib import Path
from src.graph.static_parser import StaticASTParser


@pytest.fixture
def parser():
    """Create a StaticASTParser instance."""
    return StaticASTParser()


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
import os
from typing import List, Dict
import pandas as pd

def add_column(df, column_name, default_value=None):
    '''Add a column to DataFrame.'''
    if column_name in df.columns:
        raise ValueError(f"Column '{column_name}' already exists")
    df[column_name] = default_value
    return df

def remove_column(df, column_name):
    '''Remove a column from DataFrame.'''
    return df.drop(columns=[column_name])

class DataProcessor:
    '''Process data frames.'''
    
    def __init__(self, df):
        self.df = df
    
    def add_column(self, name, value):
        '''Add a column.'''
        self.df[name] = value
        return self
    
    def get_summary(self):
        '''Get DataFrame summary.'''
        return self.df.describe()

class InvalidClass:
    def method_with_error(self):
        return undefined_variable  # This will cause NameError
""")
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def syntax_error_file():
    """Create a temporary file with syntax errors."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
def broken_function(
    # Missing closing parenthesis and body
""")
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


@pytest.mark.asyncio
async def test_parse_file_returns_module_info(parser, temp_python_file):
    """parse_file returns dict with functions, classes, imports."""
    result = await parser.parse_file(temp_python_file)
    
    assert "functions" in result
    assert "classes" in result
    assert "imports" in result
    assert isinstance(result["functions"], list)
    assert isinstance(result["classes"], list)
    assert isinstance(result["imports"], list)


@pytest.mark.asyncio
async def test_parse_file_extracts_functions(parser, temp_python_file):
    """extracts function names and their signatures."""
    result = await parser.parse_file(temp_python_file)
    
    functions = result["functions"]
    assert len(functions) == 2
    
    func_names = [f["name"] for f in functions]
    assert "add_column" in func_names
    assert "remove_column" in func_names
    
    # Check function details
    add_column_func = next(f for f in functions if f["name"] == "add_column")
    assert add_column_func["lineno"] > 0
    assert "args" in add_column_func


@pytest.mark.asyncio
async def test_parse_file_extracts_classes(parser, temp_python_file):
    """extracts class names and their methods."""
    result = await parser.parse_file(temp_python_file)
    
    classes = result["classes"]
    assert len(classes) == 2
    
    class_names = [c["name"] for c in classes]
    assert "DataProcessor" in class_names
    assert "InvalidClass" in class_names
    
    # Check class methods
    data_processor = next(c for c in classes if c["name"] == "DataProcessor")
    method_names = [m["name"] for m in data_processor["methods"]]
    assert "__init__" in method_names
    assert "add_column" in method_names
    assert "get_summary" in method_names


@pytest.mark.asyncio
async def test_parse_file_extracts_imports(parser, temp_python_file):
    """extracts import statements."""
    result = await parser.parse_file(temp_python_file)
    
    imports = result["imports"]
    assert len(imports) >= 3
    
    import_modules = [i["module"] for i in imports]
    assert "os" in import_modules
    assert "typing" in import_modules
    assert "pandas" in import_modules


@pytest.mark.asyncio
async def test_parse_file_handles_errors(parser):
    """returns empty dict for non-existent files."""
    result = await parser.parse_file(Path("/nonexistent/file.py"))
    
    assert result == {"functions": [], "classes": [], "imports": []}


@pytest.mark.asyncio
async def test_parse_file_handles_syntax_errors(parser, syntax_error_file):
    """returns empty dict for invalid Python syntax."""
    result = await parser.parse_file(syntax_error_file)
    
    assert result == {"functions": [], "classes": [], "imports": []}
