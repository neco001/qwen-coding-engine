"""
Smart Code Linking Engine for ADR-to-Code mapping.

This module provides functionality to automatically link Architecture Decision
Records (ADRs) to relevant code sections in a codebase using keyword extraction
and intelligent search/ranking.
"""

import asyncio
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CodeLink:
    """Represents a link between an ADR and a code location."""
    
    file_path: str
    line: int
    snippet: str
    relevance_score: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate and normalize the CodeLink data."""
        if not self.file_path:
            raise ValueError("file_path cannot be empty")
        if self.line < 1:
            raise ValueError("line must be positive")
        if not 0.0 <= self.relevance_score <= 1.0:
            self.relevance_score = max(0.0, min(1.0, self.relevance_score))


class SmartCodeLinker:
    """
    Smart Code Linking Engine that connects ADRs to relevant code sections.
    
    Uses keyword extraction and intelligent search to find code locations
    that implement or relate to architecture decisions documented in ADRs.
    """
    
    # Common programming stopwords to ignore during keyword extraction
    STOPWORDS = frozenset({
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
        'then', 'once', 'if', 'else', 'while', 'until', 'because', 'since',
        'use', 'using', 'used', 'uses', 'make', 'makes', 'made', 'making',
        'get', 'gets', 'got', 'getting', 'set', 'sets', 'setting', 'let',
        'new', 'return', 'true', 'false', 'null', 'none', 'class', 'def',
        'import', 'from', 'export', 'module', 'function', 'method', 'variable'
    })
    
    # File type patterns for relevance scoring
    FILE_RELEVANCE_PATTERNS = {
        'auth': ['auth', 'login', 'session', 'token', 'jwt', 'oauth', 'permission'],
        'database': ['db', 'database', 'model', 'repository', 'dao', 'schema', 'migration'],
        'api': ['api', 'endpoint', 'route', 'controller', 'handler', 'request', 'response'],
        'config': ['config', 'setting', 'env', 'environment', 'constant'],
        'service': ['service', 'manager', 'processor', 'worker', 'task'],
        'middleware': ['middleware', 'interceptor', 'filter', 'decorator'],
        'test': ['test', 'spec', 'fixture', 'mock', 'stub'],
    }
    
    def __init__(self, codebase_path: Path, search_tool: str = "ripgrep") -> None:
        """
        Initialize the SmartCodeLinker.
        
        Args:
            codebase_path: Path to the root of the codebase to search.
            search_tool: Search tool to use ('ripgrep' or 'python').
        
        Raises:
            ValueError: If codebase_path is not a valid directory.
        """
        self.codebase_path = codebase_path.resolve()
        self.search_tool = search_tool
        
        if not self.codebase_path.is_dir():
            raise ValueError(f"codebase_path must be a valid directory: {codebase_path}")
        
        self._ripgrep_available: Optional[bool] = None
    
    def _check_ripgrep_available(self) -> bool:
        """Check if ripgrep is available on the system."""
        if self._ripgrep_available is not None:
            return self._ripgrep_available
        
        try:
            result = subprocess.run(
                ['rg', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            self._ripgrep_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._ripgrep_available = False
        
        return self._ripgrep_available
    
    def extract_keywords(self, adr_content: str) -> list[str]:
        """
        Extract relevant keywords from ADR text.
        
        Uses simple NLP techniques to identify nouns and technical terms,
        filtering out common stopwords.
        
        Args:
            adr_content: The text content of the ADR.
        
        Returns:
            List of extracted keywords (lowercase, unique).
        """
        if not adr_content or not isinstance(adr_content, str):
            return []
        
        # Extract potential keywords: words with 3+ characters
        # Focus on camelCase, snake_case, and technical terms
        words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{2,}', adr_content)
        
        # Filter stopwords and normalize
        keywords = []
        seen = set()
        
        for word in words:
            word_lower = word.lower()
            
            # Skip stopwords
            if word_lower in self.STOPWORDS:
                continue
            
            # Skip very common programming words
            if len(word_lower) < 3:
                continue
            
            # Avoid duplicates
            if word_lower in seen:
                continue
            
            seen.add(word_lower)
            keywords.append(word_lower)
        
        # Prioritize technical-looking terms (camelCase, snake_case, acronyms)
        technical_pattern = re.compile(r'^[a-z]+[A-Z]|[a-z]+_[a-z]+|[A-Z]{2,}$')
        technical_keywords = [
            kw for kw in keywords if technical_pattern.search(kw)
        ]
        
        # Return technical keywords first, then others
        other_keywords = [kw for kw in keywords if kw not in technical_keywords]
        
        return technical_keywords + other_keywords
    
    async def search_code_files(self, keywords: list[str]) -> list[dict]:
        """
        Search codebase for files containing the given keywords.
        
        Args:
            keywords: List of keywords to search for.
        
        Returns:
            List of match dictionaries with file_path, line, snippet, and matches.
        """
        if not keywords:
            return []
        
        # Use ripgrep if available and requested, otherwise fallback to Python
        if self.search_tool == "ripgrep" and self._check_ripgrep_available():
            return await self._search_with_ripgrep(keywords)
        else:
            return await self._search_with_python(keywords)
    
    async def _search_with_ripgrep(self, keywords: list[str]) -> list[dict]:
        """Search using ripgrep for better performance."""
        results = []
        
        # Build search pattern (OR combination of keywords)
        # Escape special regex characters in keywords
        escaped_keywords = [re.escape(kw) for kw in keywords[:10]]  # Limit to 10 keywords
        if not escaped_keywords:
            return []
        
        pattern = '|'.join(escaped_keywords)
        
        try:
            # Run ripgrep with line numbers and context
            cmd = [
                'rg',
                '--line-number',
                '--no-heading',
                '--with-filename',
                '--max-count', '50',  # Limit matches per keyword
                '--ignore-case',
                pattern,
                str(self.codebase_path)
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30.0
            )
            
            if proc.returncode == 0 and stdout:
                lines = stdout.decode('utf-8', errors='ignore').strip().split('\n')
                
                for line in lines:
                    if not line:
                        continue
                    
                    # Parse ripgrep output: file:line:content
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        try:
                            line_num = int(parts[1])
                        except ValueError:
                            continue
                        snippet = parts[2].strip()
                        
                        # Make file path relative to codebase
                        try:
                            rel_path = str(Path(file_path).relative_to(self.codebase_path))
                        except ValueError:
                            rel_path = file_path
                        
                        # Count keyword matches in snippet
                        snippet_lower = snippet.lower()
                        match_count = sum(
                            1 for kw in keywords if kw in snippet_lower
                        )
                        
                        results.append({
                            'file_path': rel_path,
                            'line': line_num,
                            'snippet': snippet,
                            'matches': match_count,
                            'matched_keywords': [
                                kw for kw in keywords if kw in snippet_lower
                            ]
                        })
            
        except (asyncio.TimeoutError, subprocess.SubprocessError) as e:
            # Log error but continue with empty results
            # In production, you'd use proper logging
            pass
        
        return results
    
    async def _search_with_python(self, keywords: list[str]) -> list[dict]:
        """Fallback search using Python glob and grep."""
        results = []
        seen_locations = set()
        
        # Define file patterns to search
        file_patterns = [
            '**/*.py', '**/*.js', '**/*.ts', '**/*.java', '**/*.go',
            '**/*.rb', '**/*.php', '**/*.cpp', '**/*.c', '**/*.h',
            '**/*.rs', '**/*.swift', '**/*.kt', '**/*.scala',
            '**/*.sql', '**/*.yaml', '**/*.yml', '**/*.json',
            '**/*.xml', '**/*.md', '**/*.txt', '**/*.sh'
        ]
        
        for pattern in file_patterns:
            try:
                files = list(self.codebase_path.glob(pattern))
            except (OSError, PermissionError):
                continue
            
            for file_path in files[:500]:  # Limit files to search
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except (OSError, PermissionError, UnicodeDecodeError):
                    continue
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, start=1):
                    line_lower = line.lower()
                    matched_keywords = [
                        kw for kw in keywords if kw in line_lower
                    ]
                    
                    if matched_keywords:
                        location_key = f"{file_path}:{line_num}"
                        if location_key in seen_locations:
                            continue
                        
                        seen_locations.add(location_key)
                        
                        try:
                            rel_path = str(file_path.relative_to(self.codebase_path))
                        except ValueError:
                            rel_path = str(file_path)
                        
                        results.append({
                            'file_path': rel_path,
                            'line': line_num,
                            'snippet': line.strip()[:200],  # Limit snippet length
                            'matches': len(matched_keywords),
                            'matched_keywords': matched_keywords
                        })
                    
                    # Limit results per file
                    if len([r for r in results if r['file_path'] == rel_path]) >= 10:
                        break
                
                # Limit total results
                if len(results) >= 200:
                    break
            
            if len(results) >= 200:
                break
        
        return results
    
    def rank_results(self, results: list[dict], keywords: list[str]) -> list[dict]:
        """
        Rank search results by relevance.
        
        Scoring considers:
        - Number of keyword matches
        - Keyword density in snippet
        - File type relevance (e.g., auth files for auth keywords)
        - Snippet length (prefer concise matches)
        
        Args:
            results: List of search result dictionaries.
            keywords: Original keywords used for search.
        
        Returns:
            List of results sorted by relevance score (descending).
        """
        if not results:
            return []
        
        ranked = []
        
        for result in results:
            score = self._calculate_relevance_score(result, keywords)
            ranked_result = result.copy()
            ranked_result['relevance_score'] = score
            ranked.append(ranked_result)
        
        # Sort by relevance score (descending)
        ranked.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return ranked
    
    def _calculate_relevance_score(self, result: dict, keywords: list[str]) -> float:
        """
        Calculate relevance score for a single result.
        
        Args:
            result: Search result dictionary.
            keywords: List of search keywords.
        
        Returns:
            Relevance score between 0.0 and 1.0.
        """
        score = 0.0
        
        # Base score from match count (max 0.4)
        match_count = result.get('matches', 0)
        max_possible_matches = len(keywords)
        if max_possible_matches > 0:
            match_ratio = min(match_count / max_possible_matches, 1.0)
            score += match_ratio * 0.4
        
        # File relevance bonus (max 0.3)
        file_path = result.get('file_path', '').lower()
        file_relevance = self._get_file_relevance_bonus(file_path, keywords)
        score += file_relevance * 0.3
        
        # Keyword density in snippet (max 0.2)
        snippet = result.get('snippet', '')
        if snippet:
            snippet_words = len(snippet.split())
            if snippet_words > 0:
                density = match_count / snippet_words
                score += min(density * 2, 0.2)
        
        # Prefer shorter, more focused snippets (max 0.1)
        snippet_length = len(snippet)
        if 20 <= snippet_length <= 150:
            score += 0.1
        elif snippet_length > 150:
            score += 0.05
        
        return min(score, 1.0)
    
    def _get_file_relevance_bonus(self, file_path: str, keywords: list[str]) -> float:
        """
        Calculate file relevance bonus based on file path and keywords.
        
        Args:
            file_path: Relative file path.
            keywords: Search keywords.
        
        Returns:
            Bonus score between 0.0 and 1.0.
        """
        bonus = 0.0
        
        for category, patterns in self.FILE_RELEVANCE_PATTERNS.items():
            # Check if any category pattern matches keywords
            keyword_match = any(
                any(pattern in kw for pattern in patterns)
                for kw in keywords
            )
            
            # Check if file path suggests this category
            file_match = any(
                pattern in file_path for pattern in patterns
            )
            
            if keyword_match and file_match:
                bonus = max(bonus, 1.0)
            elif file_match:
                bonus = max(bonus, 0.5)
        
        return bonus
    
    async def link_adr_to_code(
        self,
        adr_content: str,
        max_results: int = 10
    ) -> list[CodeLink]:
        """
        Main API: Link an ADR to relevant code locations.
        
        This is the primary method for finding code sections that implement
        or relate to architecture decisions documented in an ADR.
        
        Args:
            adr_content: The text content of the ADR.
            max_results: Maximum number of results to return.
        
        Returns:
            List of CodeLink objects sorted by relevance.
        
        Raises:
            ValueError: If adr_content is empty or invalid.
        """
        if not adr_content or not isinstance(adr_content, str):
            raise ValueError("adr_content must be a non-empty string")
        
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        
        # Step 1: Extract keywords from ADR
        keywords = self.extract_keywords(adr_content)
        
        if not keywords:
            return []
        
        # Step 2: Search codebase for matches
        search_results = await self.search_code_files(keywords)
        
        if not search_results:
            return []
        
        # Step 3: Rank results by relevance
        ranked_results = self.rank_results(search_results, keywords)
        
        # Step 4: Convert to CodeLink objects
        code_links = []
        for result in ranked_results[:max_results]:
            try:
                code_link = CodeLink(
                    file_path=result['file_path'],
                    line=result['line'],
                    snippet=result['snippet'],
                    relevance_score=result.get('relevance_score', 0.0)
                )
                code_links.append(code_link)
            except (KeyError, ValueError) as e:
                # Skip invalid results
                continue
        
        return code_links
