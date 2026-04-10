"""
qwen_diff_audit MCP Tool.

Provides diff auditing capabilities for Anti-Degradation System.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timezone

from utils.git_diff_parser import GitDiffParser, GitDiffResult
from graph.snapshot import FunctionalSnapshotGenerator


class QwenDiffAuditTool:
    """MCP Tool for diff auditing and regression detection."""
    
    def __init__(self, repo_path: Optional[str] = None, shadow_mode: bool = False):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.git_parser = GitDiffParser(str(self.repo_path))
        self.snapshot_generator = FunctionalSnapshotGenerator(shadow_mode=shadow_mode)
        self.shadow_mode = shadow_mode
        self.audit_log_path = self.repo_path / ".anti_degradation" / "audit_log.jsonl"
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def audit_diff(
        self,
        from_ref: str = "HEAD~1",
        to_ref: str = "HEAD",
        baseline_snapshot: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Audit git diff for potential regressions.
        
        Args:
            from_ref: Source ref
            to_ref: Target ref
            baseline_snapshot: Name of baseline snapshot (default: "latest")
            
        Returns:
            Audit result with regression detection and risk assessment
        """
        start_time = datetime.now(timezone.utc)
        
        # Get git diff
        diff_result = self.git_parser.get_diff(from_ref, to_ref)
        
        # Analyze change impact
        impact_analysis = self.git_parser.analyze_change_impact(diff_result)
        
        # Load baseline snapshot
        baseline_name = baseline_snapshot or "latest"
        baseline = self.snapshot_generator.load_snapshot(self.repo_path, baseline_name)
        
        # Generate current snapshot
        current = await self.snapshot_generator.capture_snapshot(self.repo_path)
        
        # Detect regression
        regression_detected = False
        regression_details = {}
        
        if baseline:
            diff = await self.snapshot_generator.compare_snapshots(baseline, current)
            alerts = await self.snapshot_generator.detect_regression(diff)
            regression_detected = len(alerts) > 0
            regression_details = {
                "alerts": alerts,
                "diff": diff,
            }
            
            # Compare content hashes
            hash_changes = self.snapshot_generator.compare_content_hashes(
                baseline.get("content_hashes", []),
                current.get("content_hashes", []),
            )
            regression_details["hash_changes"] = hash_changes
        
        # Build audit result
        result = {
            "audit_id": f"audit_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "timestamp": start_time.isoformat().replace("+00:00", "Z"),
            "from_ref": from_ref,
            "to_ref": to_ref,
            "shadow_mode": self.shadow_mode,
            "diff_summary": {
                "files_changed": len(diff_result.files),
                "total_additions": diff_result.total_additions,
                "total_deletions": diff_result.total_deletions,
            },
            "impact_analysis": impact_analysis,
            "regression_detected": regression_detected,
            "regression_details": regression_details,
            "recommendation": self._generate_recommendation(
                regression_detected,
                impact_analysis,
                diff_result,
            ),
            "blocking": self._should_block(regression_detected, impact_analysis),
        }
        
        # Log audit result
        self._log_audit(result)
        
        return result
    
    async def audit_staged(self, baseline_snapshot: Optional[str] = None) -> Dict[str, Any]:
        """Audit staged changes (for pre-commit hook)."""
        return await self.audit_diff(
            from_ref="HEAD",
            to_ref="INDEX",
            baseline_snapshot=baseline_snapshot,
        )
    
    def _generate_recommendation(
        self,
        regression_detected: bool,
        impact_analysis: Dict,
        diff_result: GitDiffResult,
    ) -> str:
        """Generate human-readable recommendation."""
        if not regression_detected:
            if impact_analysis["risk_score"] > 0.7:
                return "HIGH_RISK: No regression detected but changes are high-risk. Recommend thorough review."
            return "PASS: No regression detected. Safe to proceed."
        
        issues = []
        if impact_analysis["change_categories"]["api_changes"]:
            issues.append(f"{len(impact_analysis['change_categories']['api_changes'])} API changes")
        if impact_analysis["high_risk_files"]:
            issues.append(f"{len(impact_analysis['high_risk_files'])} high-risk files")
        if diff_result.total_deletions > 50:
            issues.append(f"{diff_result.total_deletions} lines deleted")
        
        issues_str = ", ".join(issues) if issues else "regression detected"
        
        if self.shadow_mode:
            return f"SHADOW_MODE: Regression detected ({issues_str}). Warning only - not blocking."
        else:
            return f"BLOCK: Regression detected ({issues_str}). Review required before merge."
    
    def _should_block(self, regression_detected: bool, impact_analysis: Dict) -> bool:
        """Determine if changes should block commit/merge."""
        if self.shadow_mode:
            return False  # Never block in shadow mode
        
        if not regression_detected:
            return False
        
        # Block on high risk score
        if impact_analysis["risk_score"] > 0.8:
            return True
        
        # Block on API changes with regression
        if impact_analysis["change_categories"]["api_changes"]:
            return True
        
        # Block on critical file changes with regression
        for high_risk in impact_analysis["high_risk_files"]:
            if high_risk.get("reason") == "critical_file":
                return True
        
        return False
    
    def _log_audit(self, result: Dict[str, Any]) -> None:
        """Log audit result to audit log."""
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"Warning: Could not write audit log: {e}")
    
    def get_audit_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit history."""
        audits = []
        try:
            with open(self.audit_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        audits.append(json.loads(line))
        except FileNotFoundError:
            return []
        
        return audits[-limit:]
    
    async def create_baseline_snapshot(self, name: str = "baseline") -> str:
        """Create a new baseline snapshot."""
        snapshot = await self.snapshot_generator.capture_snapshot(self.repo_path)
        path = self.snapshot_generator.save_snapshot(snapshot, self.repo_path, name)
        return str(path)
    
    async def compare_snapshots(
        self, snapshot1_name: str, snapshot2_name: str
    ) -> Dict[str, Any]:
        """Compare two snapshots."""
        snap1 = self.snapshot_generator.load_snapshot(self.repo_path, snapshot1_name)
        snap2 = self.snapshot_generator.load_snapshot(self.repo_path, snapshot2_name)
        
        if not snap1 or not snap2:
            return {"error": "One or both snapshots not found"}
        
        diff = await self.snapshot_generator.compare_snapshots(snap1, snap2)
        alerts = await self.snapshot_generator.detect_regression(diff)
        
        return {
            "snapshot1": snapshot1_name,
            "snapshot2": snapshot2_name,
            "regression_detected": len(alerts) > 0,
            "alerts": alerts,
            "diff": diff,
        }


# MCP Tool functions for registration
async def qwen_diff_audit(
    from_ref: str = "HEAD~1",
    to_ref: str = "HEAD",
    baseline_snapshot: Optional[str] = None,
    shadow_mode: bool = False,
    workspace_root: str = ".",
) -> Dict[str, Any]:
    """
    Audit git diff for potential regressions using Anti-Degradation System.
    
    Args:
        from_ref: Source ref (commit, branch, or HEAD~N)
        to_ref: Target ref
        baseline_snapshot: Name of baseline snapshot (default: "latest")
        shadow_mode: If True, warnings only - no blocking
        workspace_root: Path to workspace root
        
    Returns:
        Audit result with regression detection and risk assessment
    """
    tool = QwenDiffAuditTool(repo_path=workspace_root, shadow_mode=shadow_mode)
    return await tool.audit_diff(from_ref, to_ref, baseline_snapshot)


async def qwen_diff_audit_staged(
    baseline_snapshot: Optional[str] = None,
    shadow_mode: bool = False,
    workspace_root: str = ".",
) -> Dict[str, Any]:
    """
    Audit staged changes for potential regressions (for pre-commit hook).
    
    Args:
        baseline_snapshot: Name of baseline snapshot (default: "latest")
        shadow_mode: If True, warnings only - no blocking
        workspace_root: Path to workspace root
        
    Returns:
        Audit result with regression detection and blocking decision
    """
    tool = QwenDiffAuditTool(repo_path=workspace_root, shadow_mode=shadow_mode)
    return await tool.audit_staged(baseline_snapshot)


async def qwen_create_baseline(
    name: str = "baseline",
    workspace_root: str = ".",
) -> str:
    """
    Create a new baseline snapshot for Anti-Degradation System.
    
    Args:
        name: Snapshot name (default: "baseline")
        workspace_root: Path to workspace root
        
    Returns:
        Path to saved snapshot file
    """
    tool = QwenDiffAuditTool(repo_path=workspace_root)
    return await tool.create_baseline_snapshot(name)


async def qwen_compare_snapshots(
    snapshot1_name: str,
    snapshot2_name: str,
    workspace_root: str = ".",
) -> Dict[str, Any]:
    """
    Compare two snapshots for regression detection.
    
    Args:
        snapshot1_name: First snapshot name
        snapshot2_name: Second snapshot name
        workspace_root: Path to workspace root
        
    Returns:
        Comparison result with regression alerts
    """
    tool = QwenDiffAuditTool(repo_path=workspace_root)
    return await tool.compare_snapshots(snapshot1_name, snapshot2_name)


async def qwen_audit_history(
    limit: int = 100,
    workspace_root: str = ".",
) -> List[Dict[str, Any]]:
    """
    Get recent audit history from Anti-Degradation System.
    
    Args:
        limit: Maximum number of audits to return
        workspace_root: Path to workspace root
        
    Returns:
        List of recent audit results
    """
    tool = QwenDiffAuditTool(repo_path=workspace_root)
    return tool.get_audit_history(limit)