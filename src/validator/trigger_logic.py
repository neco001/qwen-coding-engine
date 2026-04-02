"""Trigger Logic for AI-Driven Testing System."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TriggerThresholds:
    """Thresholds for triggering validator sessions."""
    lines_changed: int = 500
    files_modified: int = 5
    dependencies_affected: int = 10
    risk_score: float = 0.7


@dataclass
class TriggerResult:
    """Result of trigger evaluation."""
    should_trigger: bool
    reason: str
    metrics: Dict[str, Any]


class TriggerLogic:
    """Evaluates whether a validator session should be triggered."""
    
    def __init__(self, thresholds: Optional[TriggerThresholds] = None):
        """Initialize trigger logic with thresholds.
        
        Args:
            thresholds: Custom thresholds. Uses defaults if not provided.
        """
        self.thresholds = thresholds or TriggerThresholds()
    
    def evaluate_triggers(self, change_info: Dict[str, Any]) -> TriggerResult:
        """Evaluate whether validator should be triggered.
        
        Args:
            change_info: Dictionary containing change metrics:
                - lines_changed: Number of lines changed
                - files_modified: Number of files modified
                - dependencies_affected: Number of dependencies affected
                - risk_score: Risk score (0.0-1.0)
                - structural_change_detected: Boolean flag for structural changes
                
        Returns:
            TriggerResult with decision and reason.
        """
        lines_changed = change_info.get("lines_changed", 0)
        files_modified = change_info.get("files_modified", 0)
        dependencies_affected = change_info.get("dependencies_affected", 0)
        risk_score = change_info.get("risk_score", 0.0)
        structural_change = change_info.get("structural_change_detected", False)
        
        triggered_thresholds: List[str] = []
        
        # Check lines threshold
        if lines_changed >= self.thresholds.lines_changed:
            triggered_thresholds.append(
                f"lines_changed ({lines_changed} >= {self.thresholds.lines_changed})"
            )
        
        # Check files threshold
        if files_modified >= self.thresholds.files_modified:
            triggered_thresholds.append(
                f"files_modified ({files_modified} >= {self.thresholds.files_modified})"
            )
        
        # Check dependencies threshold
        if dependencies_affected >= self.thresholds.dependencies_affected:
            triggered_thresholds.append(
                f"dependencies_affected ({dependencies_affected} >= {self.thresholds.dependencies_affected})"
            )
        
        # Check risk score threshold
        if risk_score >= self.thresholds.risk_score:
            triggered_thresholds.append(
                f"risk_score ({risk_score:.2f} >= {self.thresholds.risk_score:.2f})"
            )
        
        # Check structural change flag
        if structural_change:
            triggered_thresholds.append("structural_change_detected")
        
        # Determine if validator should be triggered
        should_trigger = len(triggered_thresholds) > 0
        
        if should_trigger:
            reason = f"Validator triggered: {', '.join(triggered_thresholds)}"
        else:
            reason = "All metrics below thresholds"
        
        return TriggerResult(
            should_trigger=should_trigger,
            reason=reason,
            metrics=change_info,
        )
