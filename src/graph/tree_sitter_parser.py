"""
Tree-sitter AST Parser Module

Provides language-agnostic code analysis using tree-sitter parsers.
Supports Python, TypeScript, and YAML with graceful fallback when
tree-sitter is not installed.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Attempt to import tree-sitter with graceful fallback
try:
    from tree_sitter import Language, Parser
    import tree_sitter_python
    import tree_sitter_typescript
    import tree_sitter_yaml
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Language = None
    Parser = None


class TreeSitterAnalyzer:
    """
    Analyzes source code files using tree-sitter parsers.
    
    Extracts symbols, detects architectural patterns, and identifies
    decision points in the code structure.
    """

    SUPPORTED_LANGUAGES = {"python", "typescript", "yaml"}
    
    ARCHITECTURAL_PATTERNS = [
        "Repository",
        "Factory", 
        "Service",
        "Controller",
        "Decorator",
        "Singleton"
    ]

    def __init__(self) -> None:
        """
        Initialize the analyzer with tree-sitter parsers for supported languages.
        
        Sets up parser instances for python, typescript, and yaml.
        If tree-sitter is not available, initializes with None parsers
        and will use fallback parsing methods.
        """
        self._parsers: dict[str, Optional[Parser]] = {}
        self._languages: dict[str, Optional[Language]] = {}
        
        if TREE_SITTER_AVAILABLE:
            self._initialize_parsers()
        else:
            logger.warning(
                "tree-sitter not installed. Using fallback parsing methods. "
                "Install with: pip install tree-sitter tree-sitter-python "
                "tree-sitter-typescript tree-sitter-yaml"
            )

    def _initialize_parsers(self) -> None:
        """Initialize tree-sitter parsers for each supported language."""
        language_configs = {
            "python": tree_sitter_python.language(),
            "typescript": tree_sitter_typescript.language_typescript(),
            "yaml": tree_sitter_yaml.language(),
        }
        
        for lang_name, lang_obj in language_configs.items():
            try:
                parser = Parser()
                parser.set_language(lang_obj)
                self._parsers[lang_name] = parser
                self._languages[lang_name] = lang_obj
                logger.debug(f"Initialized tree-sitter parser for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize parser for {lang_name}: {e}")
                self._parsers[lang_name] = None
                self._languages[lang_name] = None

    async def parse(self, file_path: Path, language: str) -> dict[str, Any]:
        """
        Parse a source file and extract structural information.
        
        Args:
            file_path: Path to the file to parse
            language: Language identifier (python, typescript, yaml)
            
        Returns:
            Dictionary with keys: functions, classes, imports
            
        Raises:
            ValueError: If language is not supported
            FileNotFoundError: If file does not exist
        """
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {', '.join(self.SUPPORTED_LANGUAGES)}"
            )
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return {"functions": [], "classes": [], "imports": []}
        
        if TREE_SITTER_AVAILABLE and self._parsers.get(language):
            return await self._parse_with_tree_sitter(content, language)
        else:
            return await self._parse_with_fallback(content, language, file_path)

    async def _parse_with_tree_sitter(
        self, 
        content: str, 
        language: str
    ) -> dict[str, Any]:
        """Parse content using tree-sitter parser."""
        parser = self._parsers.get(language)
        if not parser:
            return {"functions": [], "classes": [], "imports": []}
        
        try:
            tree = parser.parse(bytes(content, "utf-8"))
            root_node = tree.root_node
            
            functions = await self._extract_functions(root_node, content, language)
            classes = await self._extract_classes(root_node, content, language)
            imports = await self._extract_imports(root_node, content, language)
            
            return {
                "functions": functions,
                "classes": classes,
                "imports": imports,
            }
        except Exception as e:
            logger.error(f"Tree-sitter parsing error for {language}: {e}")
            return {"functions": [], "classes": [], "imports": []}

    async def _parse_with_fallback(
        self, 
        content: str, 
        language: str,
        file_path: Path
    ) -> dict[str, Any]:
        """Fallback parsing using regex-based approach when tree-sitter unavailable."""
        functions = []
        classes = []
        imports = []
        
        lines = content.split("\n")
        
        if language == "python":
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                
                if stripped.startswith("def "):
                    func_name = stripped.split("def ")[1].split("(")[0].strip()
                    functions.append({
                        "name": func_name,
                        "line": line_num,
                        "file": str(file_path)
                    })
                
                elif stripped.startswith("class "):
                    class_name = stripped.split("class ")[1].split("(")[0].split(":")[0].strip()
                    classes.append({
                        "name": class_name,
                        "line": line_num,
                        "file": str(file_path)
                    })
                
                elif stripped.startswith("import ") or stripped.startswith("from "):
                    imports.append({
                        "statement": stripped,
                        "line": line_num,
                        "file": str(file_path)
                    })
        
        elif language == "typescript":
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                
                if stripped.startswith("function ") or stripped.startswith("async function "):
                    func_name = stripped.replace("async function ", "").replace("function ", "").split("(")[0].strip()
                    functions.append({
                        "name": func_name,
                        "line": line_num,
                        "file": str(file_path)
                    })
                
                elif stripped.startswith("class "):
                    class_name = stripped.split("class ")[1].split("{")[0].split(" ")[0].strip()
                    classes.append({
                        "name": class_name,
                        "line": line_num,
                        "file": str(file_path)
                    })
                
                elif stripped.startswith("import "):
                    imports.append({
                        "statement": stripped,
                        "line": line_num,
                        "file": str(file_path)
                    })
        
        elif language == "yaml":
            # YAML doesn't have functions/classes in traditional sense
            # Extract top-level keys as structural elements
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and ":" in stripped:
                    key = stripped.split(":")[0].strip()
                    if not key.startswith("-"):
                        imports.append({
                            "key": key,
                            "line": line_num,
                            "file": str(file_path)
                        })
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
        }

    async def _extract_functions(
        self, 
        root_node: Any, 
        content: str, 
        language: str
    ) -> list[dict[str, Any]]:
        """Extract function definitions from AST."""
        functions = []
        
        query_patterns = {
            "python": "(function_definition name: (identifier) @name) @func",
            "typescript": "(function_declaration name: (identifier) @name) @func",
        }
        
        pattern = query_patterns.get(language)
        if not pattern or not self._languages.get(language):
            return functions
        
        try:
            lang = self._languages[language]
            query = lang.query(pattern)
            captures = query.captures(root_node)
            
            for capture_name, node in captures:
                if capture_name == "name":
                    func_name = content[node.start_byte:node.end_byte]
                    functions.append({
                        "name": func_name,
                        "line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                    })
        except Exception as e:
            logger.debug(f"Error extracting functions: {e}")
        
        return functions

    async def _extract_classes(
        self, 
        root_node: Any, 
        content: str, 
        language: str
    ) -> list[dict[str, Any]]:
        """Extract class definitions from AST."""
        classes = []
        
        query_patterns = {
            "python": "(class_definition name: (identifier) @name) @class",
            "typescript": "(class_declaration name: (identifier) @name) @class",
        }
        
        pattern = query_patterns.get(language)
        if not pattern or not self._languages.get(language):
            return classes
        
        try:
            lang = self._languages[language]
            query = lang.query(pattern)
            captures = query.captures(root_node)
            
            for capture_name, node in captures:
                if capture_name == "name":
                    class_name = content[node.start_byte:node.end_byte]
                    classes.append({
                        "name": class_name,
                        "line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                    })
        except Exception as e:
            logger.debug(f"Error extracting classes: {e}")
        
        return classes

    async def _extract_imports(
        self, 
        root_node: Any, 
        content: str, 
        language: str
    ) -> list[dict[str, Any]]:
        """Extract import statements from AST."""
        imports = []
        
        query_patterns = {
            "python": """
                [
                    (import_statement) @import
                    (import_from_statement) @import
                ]
            """,
            "typescript": "(import_statement) @import",
        }
        
        pattern = query_patterns.get(language)
        if not pattern or not self._languages.get(language):
            return imports
        
        try:
            lang = self._languages[language]
            query = lang.query(pattern)
            captures = query.captures(root_node)
            
            seen_lines = set()
            for capture_name, node in captures:
                if capture_name == "import":
                    line_num = node.start_point[0] + 1
                    if line_num not in seen_lines:
                        import_stmt = content[node.start_byte:node.end_byte]
                        imports.append({
                            "statement": import_stmt,
                            "line": line_num,
                        })
                        seen_lines.add(line_num)
        except Exception as e:
            logger.debug(f"Error extracting imports: {e}")
        
        return imports

    async def extract_symbols(self, parse_result: dict[str, Any]) -> dict[str, Any]:
        """
        Extract and organize symbols from parse result.
        
        Args:
            parse_result: Dictionary from parse() method
            
        Returns:
            Dictionary with organized symbols by type
        """
        symbols = {
            "functions": [],
            "classes": [],
            "imports": [],
            "total_count": 0,
        }
        
        try:
            functions = parse_result.get("functions", [])
            classes = parse_result.get("classes", [])
            imports = parse_result.get("imports", [])
            
            symbols["functions"] = [
                {
                    "name": f.get("name", "unknown"),
                    "line": f.get("line", 0),
                    "type": "function",
                }
                for f in functions
            ]
            
            symbols["classes"] = [
                {
                    "name": c.get("name", "unknown"),
                    "line": c.get("line", 0),
                    "type": "class",
                }
                for c in classes
            ]
            
            symbols["imports"] = [
                {
                    "statement": i.get("statement", i.get("key", "")),
                    "line": i.get("line", 0),
                    "type": "import",
                }
                for i in imports
            ]
            
            symbols["total_count"] = (
                len(symbols["functions"]) + 
                len(symbols["classes"]) + 
                len(symbols["imports"])
            )
            
        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")
        
        return symbols

    async def detect_patterns(
        self, 
        parse_result: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Detect architectural patterns in the parsed code.
        
        Identifies patterns: Repository, Factory, Service, Controller, 
        Decorator, Singleton
        
        Args:
            parse_result: Dictionary from parse() method
            
        Returns:
            List of detected patterns with metadata
        """
        detected_patterns = []
        
        try:
            classes = parse_result.get("classes", [])
            functions = parse_result.get("functions", [])
            imports = parse_result.get("imports", [])
            
            # Pattern detection rules based on naming conventions
            pattern_indicators = {
                "Repository": ["repository", "repo", "dao"],
                "Factory": ["factory", "builder", "creator"],
                "Service": ["service", "manager", "handler"],
                "Controller": ["controller", "api", "endpoint"],
                "Decorator": ["decorator", "wrapper", "middleware"],
                "Singleton": ["singleton", "instance", "registry"],
            }
            
            # Check class names for pattern indicators
            for cls in classes:
                class_name = cls.get("name", "").lower()
                line = cls.get("line", 0)
                
                for pattern, indicators in pattern_indicators.items():
                    if any(indicator in class_name for indicator in indicators):
                        detected_patterns.append({
                            "pattern": pattern,
                            "location": cls.get("name", "unknown"),
                            "line": line,
                            "type": "class",
                            "confidence": "high",
                        })
            
            # Check function names for pattern indicators
            for func in functions:
                func_name = func.get("name", "").lower()
                line = func.get("line", 0)
                
                for pattern, indicators in pattern_indicators.items():
                    if any(indicator in func_name for indicator in indicators):
                        # Avoid duplicate if already detected in class
                        existing = [
                            p for p in detected_patterns 
                            if p["pattern"] == pattern and p["line"] == line
                        ]
                        if not existing:
                            detected_patterns.append({
                                "pattern": pattern,
                                "location": func.get("name", "unknown"),
                                "line": line,
                                "type": "function",
                                "confidence": "medium",
                            })
            
            # Check imports for pattern hints
            for imp in imports:
                stmt = imp.get("statement", "").lower()
                line = imp.get("line", 0)
                
                for pattern, indicators in pattern_indicators.items():
                    if any(indicator in stmt for indicator in indicators):
                        detected_patterns.append({
                            "pattern": pattern,
                            "location": stmt[:50],
                            "line": line,
                            "type": "import",
                            "confidence": "low",
                        })
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return detected_patterns

    async def find_decision_points(
        self, 
        parse_result: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Find decision points in the parsed code.
        
        Decision points include: class definitions, function definitions 
        with conditionals, import statements
        
        Args:
            parse_result: Dictionary from parse() method
            
        Returns:
            List of decision points with line numbers and types
        """
        decision_points = []
        
        try:
            classes = parse_result.get("classes", [])
            functions = parse_result.get("functions", [])
            imports = parse_result.get("imports", [])
            
            # Class definitions are decision points (inheritance, composition)
            for cls in classes:
                decision_points.append({
                    "type": "class_definition",
                    "name": cls.get("name", "unknown"),
                    "line": cls.get("line", 0),
                    "description": "Class definition - inheritance/composition decision",
                })
            
            # Function definitions are decision points
            for func in functions:
                decision_points.append({
                    "type": "function_definition",
                    "name": func.get("name", "unknown"),
                    "line": func.get("line", 0),
                    "description": "Function definition - logic flow decision",
                })
            
            # Import statements are decision points (dependencies)
            for imp in imports:
                statement = imp.get("statement", imp.get("key", ""))
                decision_points.append({
                    "type": "import_statement",
                    "name": statement[:40] if statement else "unknown",
                    "line": imp.get("line", 0),
                    "description": "Import statement - dependency decision",
                })
            
            # Sort by line number for easier navigation
            decision_points.sort(key=lambda x: x.get("line", 0))
            
        except Exception as e:
            logger.error(f"Error finding decision points: {e}")
        
        return decision_points

    def is_tree_sitter_available(self) -> bool:
        """Check if tree-sitter is available for parsing."""
        return TREE_SITTER_AVAILABLE

    def get_supported_languages(self) -> set[str]:
        """Return set of supported language identifiers."""
        return self.SUPPORTED_LANGUAGES.copy()
