"""Approval Modal Integration with MCP Server.

This module provides the backend API for handling approval requests
from the SPECTER QWEN HUD UI.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ChangeRequest:
    """Represents a change request requiring approval."""
    id: str
    type: str  # 'code_change' or 'validator_trigger'
    lines_changed: int
    files_modified: int
    dependencies_affected: int
    risk_score: float
    reason: str


@dataclass
class ApprovalResult:
    """Result of processing an approval request."""
    changeId: str
    status: str  # 'approved', 'rejected', 'failed'
    message: Optional[str] = None


@dataclass
class ApprovalResponse:
    """Response sent back to UI after processing approval."""
    success: bool
    results: List[ApprovalResult]


class ApprovalModalHandler:
    """Handles approval requests from the UI.
    
    This class processes batch approval requests and integrates with
    the validator system to apply user decisions.
    """
    
    def __init__(self, validator_integration=None):
        """Initialize approval handler.
        
        Args:
            validator_integration: Optional ValidatorIntegration instance
                                   for applying approvals to validator system.
        """
        self.validator_integration = validator_integration
        self._approval_history: List[Dict[str, Any]] = []
    
    async def handle_approval_request(
        self,
        changes: List[Dict[str, Any]],
        action: str,
        reason: Optional[str] = None,
    ) -> ApprovalResponse:
        """Handle a batch approval request from UI.
        
        Args:
            changes: List of change request dictionaries
            action: 'approve' or 'reject'
            reason: Optional reason provided by user
            
        Returns:
            ApprovalResponse with results for each change
        """
        logger.info(f"Handling approval request: action={action}, changes={len(changes)}")
        
        results: List[ApprovalResult] = []
        
        for change_data in changes:
            try:
                change = self._parse_change_request(change_data)
                result = await self._process_change(change, action, reason)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process change {change_data.get('id', 'unknown')}: {e}")
                results.append(ApprovalResult(
                    changeId=change_data.get('id', 'unknown'),
                    status='failed',
                    message=str(e)
                ))
        
        # Record in history
        self._record_approval(action, changes, results, reason)
        
        success = all(r.status in ('approved', 'rejected') for r in results)
        return ApprovalResponse(success=success, results=results)
    
    def _parse_change_request(self, data: Dict[str, Any]) -> ChangeRequest:
        """Parse change request from dictionary.
        
        Args:
            data: Dictionary with change request data
            
        Returns:
            ChangeRequest object
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ['id', 'type', 'lines_changed', 'files_modified', 
                          'dependencies_affected', 'risk_score', 'reason']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return ChangeRequest(
            id=data['id'],
            type=data['type'],
            lines_changed=int(data['lines_changed']),
            files_modified=int(data['files_modified']),
            dependencies_affected=int(data['dependencies_affected']),
            risk_score=float(data['risk_score']),
            reason=data['reason']
        )
    
    async def _process_change(
        self,
        change: ChangeRequest,
        action: str,
        reason: Optional[str] = None
    ) -> ApprovalResult:
        """Process a single change request.
        
        Args:
            change: ChangeRequest object
            action: 'approve' or 'reject'
            reason: Optional reason
            
        Returns:
            ApprovalResult for this change
        """
        if action == 'approve':
            return await self._approve_change(change, reason)
        elif action == 'reject':
            return await self._reject_change(change, reason)
        else:
            return ApprovalResult(
                changeId=change.id,
                status='failed',
                message=f"Invalid action: {action}"
            )
    
    async def _approve_change(
        self,
        change: ChangeRequest,
        reason: Optional[str] = None
    ) -> ApprovalResult:
        """Approve a change request.
        
        Args:
            change: ChangeRequest to approve
            reason: Optional approval reason
            
        Returns:
            ApprovalResult with status
        """
        # If validator integration is available, notify it
        if self.validator_integration:
            try:
                # Mark change as approved in validator system
                await self._notify_validator_approved(change)
            except Exception as e:
                logger.warning(f"Validator notification failed: {e}")
        
        return ApprovalResult(
            changeId=change.id,
            status='approved',
            message=f"Change approved: {change.reason[:50]}..." if len(change.reason) > 50 else f"Change approved: {change.reason}"
        )
    
    async def _reject_change(
        self,
        change: ChangeRequest,
        reason: Optional[str] = None
    ) -> ApprovalResult:
        """Reject a change request.
        
        Args:
            change: ChangeRequest to reject
            reason: Optional rejection reason
            
        Returns:
            ApprovalResult with status
        """
        # If validator integration is available, notify it
        if self.validator_integration:
            try:
                # Mark change as rejected in validator system
                await self._notify_validator_rejected(change)
            except Exception as e:
                logger.warning(f"Validator notification failed: {e}")
        
        return ApprovalResult(
            changeId=change.id,
            status='rejected',
            message=f"Change rejected: {reason or 'User declined'}"
        )
    
    async def _notify_validator_approved(self, change: ChangeRequest) -> None:
        """Notify validator system of approved change.
        
        Args:
            change: Approved ChangeRequest
        """
        # Placeholder for validator integration
        # In full implementation, this would update the validator state
        logger.debug(f"Validator notified: change {change.id} approved")
    
    async def _notify_validator_rejected(self, change: ChangeRequest) -> None:
        """Notify validator system of rejected change.
        
        Args:
            change: Rejected ChangeRequest
        """
        # Placeholder for validator integration
        # In full implementation, this would rollback or skip the change
        logger.debug(f"Validator notified: change {change.id} rejected")
    
    def _record_approval(
        self,
        action: str,
        changes: List[Dict[str, Any]],
        results: List[ApprovalResult],
        reason: Optional[str] = None
    ) -> None:
        """Record approval decision in history.
        
        Args:
            action: 'approve' or 'reject'
            changes: Original change requests
            results: Processing results
            reason: User-provided reason
        """
        record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'action': action,
            'changes_count': len(changes),
            'results': [asdict(r) for r in results],
            'reason': reason,
        }
        self._approval_history.append(record)
        
        # Keep only last 100 records
        if len(self._approval_history) > 100:
            self._approval_history = self._approval_history[-100:]
    
    def get_approval_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent approval history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of approval history records
        """
        return self._approval_history[-limit:]
