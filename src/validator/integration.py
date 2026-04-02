"""Validator Integration with Sparring Engine.

This module provides integration between the validator trigger system
and the Sparring Engine, enabling automatic validator session triggers
based on code change metrics.
"""

import logging
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

from src.validator.metrics_collector import MetricsCollector, ChangeMetrics
from src.validator.trigger_logic import TriggerLogic, TriggerResult, TriggerThresholds
from src.session.isolation_manager import SessionIsolationManager, SessionConfig

logger = logging.getLogger(__name__)


class ValidatorIntegration:
    """Integrates validator triggers with Sparring Engine.
    
    This class provides the bridge between code changes and validator
    session creation, automatically triggering validator sessions when
    change thresholds are exceeded.
    """
    
    def __init__(
        self,
        trigger_logic: Optional[TriggerLogic] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        isolation_manager: Optional[SessionIsolationManager] = None,
    ):
        """Initialize validator integration.
        
        Args:
            trigger_logic: Trigger logic instance. Uses default if not provided.
            metrics_collector: Metrics collector instance. Uses default if not provided.
            isolation_manager: Session isolation manager. Uses default if not provided.
        """
        self.trigger_logic = trigger_logic or TriggerLogic()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.isolation_manager = isolation_manager or SessionIsolationManager()
    
    async def evaluate_and_trigger_validator(
        self,
        project_dir: Path,
        change_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str], TriggerResult]:
        """Evaluate changes and trigger validator session if needed.
        
        Args:
            project_dir: Path to the project directory.
            change_info: Optional pre-computed change metrics. If not provided,
                        will collect metrics from project_dir.
        
        Returns:
            Tuple of (triggered, session_id, trigger_result):
            - triggered: True if validator session was created
            - session_id: ID of created session, or None if not triggered
            - trigger_result: Result of trigger evaluation
        """
        # Collect metrics if not provided
        if change_info is None:
            metrics = await self.metrics_collector.collect_metrics(project_dir)
            change_info = {
                "lines_changed": metrics.total_lines,
                "files_modified": metrics.total_files,
                "dependencies_affected": metrics.total_imports,
                "risk_score": 0.0,  # Will be computed by AIEnhancer if needed
            }
        
        # Evaluate triggers
        trigger_result = self.trigger_logic.evaluate_triggers(change_info)
        
        if not trigger_result.should_trigger:
            logger.info(f"Validator not triggered: {trigger_result.reason}")
            return False, None, trigger_result
        
        # Create validator session
        try:
            session_id = self._create_validator_session(change_info)
            logger.info(f"Validator session created: {session_id}")
            return True, session_id, trigger_result
        except Exception as e:
            logger.error(f"Failed to create validator session: {e}")
            return False, None, trigger_result
    
    def _create_validator_session(self, change_info: Dict[str, Any]) -> str:
        """Create a validator session.
        
        Args:
            change_info: Dictionary with change metrics.
        
        Returns:
            Session ID for the created validator session.
        """
        topic = f"Validator: Code changes review"
        context = self._build_validator_context(change_info)
        
        checkpoint = self.isolation_manager.create_session(
            role="validator",
            topic=topic,
            context=context,
        )
        
        return checkpoint.session_id
    
    def _build_validator_context(self, change_info: Dict[str, Any]) -> str:
        """Build context string for validator session.
        
        Args:
            change_info: Dictionary with change metrics.
        
        Returns:
            Formatted context string for validator.
        """
        lines = change_info.get("lines_changed", 0)
        files = change_info.get("files_modified", 0)
        deps = change_info.get("dependencies_affected", 0)
        risk = change_info.get("risk_score", 0.0)
        structural = change_info.get("structural_change_detected", False)
        
        context_parts = [
            "## Change Metrics",
            f"- Lines changed: {lines}",
            f"- Files modified: {files}",
            f"- Dependencies affected: {deps}",
            f"- Risk score: {risk:.2f}",
            f"- Structural change detected: {structural}",
            "",
            "## Validation Tasks",
            "1. Review code changes for regressions",
            "2. Verify dependency impacts",
            "3. Check for structural integrity",
            "4. Validate against original requirements",
        ]
        
        return "\n".join(context_parts)
    
    def get_trigger_thresholds(self) -> TriggerThresholds:
        """Get current trigger thresholds.
        
        Returns:
            Current trigger thresholds configuration.
        """
        return self.trigger_logic.thresholds
    
    def update_thresholds(self, thresholds: TriggerThresholds) -> None:
        """Update trigger thresholds.
        
        Args:
            thresholds: New thresholds configuration.
        """
        self.trigger_logic.thresholds = thresholds
        logger.info(f"Updated trigger thresholds: {thresholds}")
