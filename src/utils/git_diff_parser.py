"""
Git Diff Parser Utility for Anti-Degradation System.

Parses git diffs and extracts change metadata for audit.
"""

import subprocess
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone


@dataclass
class DiffHunk:
    """Represents a single hunk in a git diff."""
    file_path: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    added_lines: List[str] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)
    context_lines: List[str] = field(default_factory=list)


@dataclass
class FileDiff:
    """Represents changes to a single file."""
    file_path: str
    status: str  # added, modified, deleted, renamed
    hunks: List[DiffHunk] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    old_path: Optional[str] = None  # For renamed files


@dataclass
class GitDiffResult:
    """Complete git diff result."""
    commit_from: str
    commit_to: str
    files: List[FileDiff] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        return {
            "commit_from": self.commit_from,
            "commit_to": self.commit_to,
            "files": [
                {
                    "file_path": f.file_path,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "hunks_count": len(f.hunks),
                }
                for f in self.files
            ],
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
        }


class GitDiffParser:
    """Parser for git diffs with semantic analysis capabilities."""
    
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
    
    def _run_git_command(self, args: List[str]) -> Tuple[str, str, int]:
        """Run git command and return stdout, stderr, returncode."""
        try:
            result = subprocess.run(
                ["git", "--no-pager"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", 1
        except Exception as e:
            return "", str(e), 1
    
    def get_diff(
        self, from_ref: str = "HEAD~1", to_ref: str = "HEAD", staged: bool = False
    ) -> GitDiffResult:
        """
        Get git diff between two refs.
        
        Args:
            from_ref: Source ref (commit, branch, or HEAD~N)
            to_ref: Target ref
            staged: If True, compare staged changes vs HEAD
            
        Returns:
            GitDiffResult with parsed diff information
        """
        if staged:
            stdout, stderr, code = self._run_git_command(["diff", "--cached", "--unified=0"])
            from_ref = "INDEX"
            to_ref = "HEAD"
        else:
            stdout, stderr, code = self._run_git_command([
                "diff", f"{from_ref}...{to_ref}", "--unified=0"
            ])
        
        if code != 0:
            raise RuntimeError(f"Git diff failed: {stderr}")
        
        return self._parse_diff(stdout, from_ref, to_ref)
    
    def get_staged_diff(self) -> GitDiffResult:
        """Get diff of staged changes only (for pre-commit)."""
        return self.get_diff(staged=True)
    
    def _parse_diff(self, diff_output: str, from_ref: str, to_ref: str) -> GitDiffResult:
        """Parse raw git diff output into structured data."""
        result = GitDiffResult(
            commit_from=from_ref,
            commit_to=to_ref,
        )
        
        if not diff_output.strip():
            return result
        
        current_file: Optional[FileDiff] = None
        current_hunk: Optional[DiffHunk] = None
        
        for line in diff_output.split("\n"):
            # File header: diff --git a/path b/path
            if line.startswith("diff --git"):
                if current_file:
                    result.files.append(current_file)
                
                match = re.search(r"b/(.+)$", line)
                file_path = match.group(1) if match else "unknown"
                
                current_file = FileDiff(
                    file_path=file_path,
                    status="modified",
                )
                current_hunk = None
                continue
            
            # File status markers
            if line.startswith("new file mode"):
                if current_file:
                    current_file.status = "added"
                continue
            
            if line.startswith("deleted file mode"):
                if current_file:
                    current_file.status = "deleted"
                continue
            
            # Rename detection
            if line.startswith("rename from"):
                if current_file:
                    current_file.old_path = line.replace("rename from ", "").strip()
                continue
            
            if line.startswith("rename to"):
                if current_file:
                    current_file.status = "renamed"
                continue
            
            # Hunk header: @@ -old_start,old_lines +new_start,new_lines @@
            hunk_match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if hunk_match and current_file:
                if current_hunk and current_file:
                    current_file.hunks.append(current_hunk)
                
                current_hunk = DiffHunk(
                    file_path=current_file.file_path,
                    old_start=int(hunk_match.group(1)),
                    old_lines=int(hunk_match.group(2) or 1),
                    new_start=int(hunk_match.group(3)),
                    new_lines=int(hunk_match.group(4) or 1),
                )
                continue
            
            # Content lines
            if current_hunk:
                if line.startswith("+") and not line.startswith("+++"):
                    current_hunk.added_lines.append(line[1:])
                    if current_file:
                        current_file.additions += 1
                        result.total_additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    current_hunk.removed_lines.append(line[1:])
                    if current_file:
                        current_file.deletions += 1
                        result.total_deletions += 1
                elif line.startswith(" ") or (not line.startswith(("+", "-")) and line.strip()):
                    current_hunk.context_lines.append(line[1:] if line.startswith(" ") else line)
        
        # Append last file and hunk
        if current_hunk and current_file:
            current_file.hunks.append(current_hunk)
        if current_file:
            result.files.append(current_file)
        
        return result
    
    def get_changed_files(self, from_ref: str = "HEAD~1", to_ref: str = "HEAD") -> List[str]:
        """Get list of changed file paths."""
        stdout, stderr, code = self._run_git_command([
            "diff", "--name-only", f"{from_ref}...{to_ref}"
        ])
        
        if code != 0:
            return []
        
        return [f.strip() for f in stdout.split("\n") if f.strip()]
    
    def get_file_content_at_ref(self, file_path: str, ref: str = "HEAD") -> Optional[str]:
        """Get file content at a specific ref."""
        stdout, stderr, code = self._run_git_command([
            "show", f"{ref}:{file_path}"
        ])
        
        if code != 0:
            return None
        
        return stdout
    
    def analyze_change_impact(self, diff_result: GitDiffResult) -> Dict:
        """
        Analyze the impact of changes for risk assessment.
        
        Returns dict with:
        - high_risk_files: Files with critical changes
        - change_categories: Categorized changes (api, logic, config, etc.)
        - affected_functions: Functions that may be affected
        """
        analysis = {
            "high_risk_files": [],
            "change_categories": {
                "api_changes": [],
                "logic_changes": [],
                "config_changes": [],
                "test_changes": [],
                "documentation": [],
            },
            "affected_functions": [],
            "risk_score": 0.0,
        }
        
        for file_diff in diff_result.files:
            file_path = file_diff.file_path.lower()
            
            # Categorize by file type
            if file_path.endswith(("test.py", "_spec.py", "test_")):
                analysis["change_categories"]["test_changes"].append(file_diff.file_path)
            elif file_path.endswith(("yaml", ".yml", ".json", ".toml")):
                analysis["change_categories"]["config_changes"].append(file_diff.file_path)
            elif file_path.endswith(("md", ".rst")):
                analysis["change_categories"]["documentation"].append(file_diff.file_path)
            else:
                # Check for API changes (function signature changes)
                for hunk in file_diff.hunks:
                    for line in hunk.removed_lines + hunk.added_lines:
                        if re.match(r"^\s*def\s+\w+\s*\(", line):
                            analysis["change_categories"]["api_changes"].append({
                                "file": file_diff.file_path,
                                "change": line.strip(),
                            })
                            analysis["affected_functions"].append(line.strip())
                            break
                
                analysis["change_categories"]["logic_changes"].append(file_diff.file_path)
            
            # High risk: large changes or critical files
            if file_diff.additions + file_diff.deletions > 100:
                analysis["high_risk_files"].append({
                    "file": file_diff.file_path,
                    "reason": "large_change",
                    "lines_changed": file_diff.additions + file_diff.deletions,
                })
            
            if any(critical in file_path for critical in ["auth", "payment", "security", "db"]):
                analysis["high_risk_files"].append({
                    "file": file_diff.file_path,
                    "reason": "critical_file",
                    "lines_changed": file_diff.additions + file_diff.deletions,
                })
        
        # Calculate risk score
        analysis["risk_score"] = (
            len(analysis["change_categories"]["api_changes"]) * 0.3
            + len(analysis["high_risk_files"]) * 0.2
            + diff_result.total_deletions * 0.01
            + diff_result.total_additions * 0.005
        )
        analysis["risk_score"] = min(1.0, analysis["risk_score"])
        
        return analysis
    
    def get_commit_info(self, ref: str = "HEAD") -> Dict:
        """Get commit information for a ref."""
        stdout, stderr, code = self._run_git_command([
            "log", "-1", "--format=%H|%an|%ae|%ai|%s", ref
        ])
        
        if code != 0 or not stdout.strip():
            return {}
        
        parts = stdout.strip().split("|")
        if len(parts) >= 5:
            return {
                "hash": parts[0],
                "author_name": parts[1],
                "author_email": parts[2],
                "date": parts[3],
                "message": parts[4],
            }
        
        return {}