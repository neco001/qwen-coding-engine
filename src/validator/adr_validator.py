"""TDD Validation Gate - ADR Compliance Checker."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ValidationResult:
    """Result of validating a single ADR record."""
    
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)


@dataclass
class FileValidationReport:
    """Report for validating an entire parquet file."""
    
    file_path: str
    total_records: int
    compliant_records: int
    records_with_warnings: int
    records_with_errors: int
    compliance_percentage: float = 0.0
    
    def __post_init__(self):
        if self.total_records > 0:
            self.compliance_percentage = (self.compliant_records / self.total_records) * 100


class DependencyGraph:
    """Simple dependency graph for ADR dependency validation."""
    
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str], str] = {}
    
    def add_node(self, node_id: str, data: dict[str, Any]) -> None:
        self.nodes[node_id] = data
    
    def add_edge(self, source: str, target: str, relationship: str = "depends_on") -> None:
        self.edges[(source, target)] = relationship
    
    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes


class ADRValidator:
    """
    ADR Compliance Validator.
    
    Validates ADR records against schema requirements and best practices:
    - Required field presence
    - Valid adr_status values
    - Linked code node existence
    - Dependency chain integrity
    """
    
    VALID_ADR_STATUSES = {"accepted", "rejected", "proposed", "deprecated", "superseded"}
    
    REQUIRED_BASE_FIELDS = {
        "decision_id",
        "timestamp",
        "session_id",
        "decision_type",
        "title",
        "advice",
        "complexity",
        "risk_score",
    }
    
    REQUIRED_ADR_FIELDS = {
        "adr_status",
        "adr_context",
        "adr_consequences",
        "adr_alternatives",
    }
    
    def __init__(self, codebase_path: Optional[Path] = None) -> None:
        """
        Initialize the ADR validator.
        
        Args:
            codebase_path: Optional path to codebase for validating linked code nodes
        """
        self.codebase_path = codebase_path
    
    def validate_record(self, record: dict[str, Any]) -> ValidationResult:
        """
        Validate a single ADR record.
        
        Args:
            record: Dictionary containing ADR record data
        
        Returns:
            ValidationResult with errors, warnings, and info messages
        """
        errors: list[str] = []
        warnings: list[str] = []
        info: list[str] = []
        
        # Check required base fields
        missing_base = self.REQUIRED_BASE_FIELDS - set(record.keys())
        if missing_base:
            errors.append(f"Missing required base fields: {missing_base}")
        
        # Check ADR-specific fields (warnings, not errors for backward compatibility)
        missing_adr = self.REQUIRED_ADR_FIELDS - set(record.keys())
        if missing_adr:
            warnings.append(f"Missing ADR-specific fields: {missing_adr}")
        
        # Check for None values in present ADR fields
        for field_name in self.REQUIRED_ADR_FIELDS:
            if field_name in record and record[field_name] is None:
                warnings.append(f"Field '{field_name}' is null")
        
        # Validate adr_status if present
        if "adr_status" in record and record["adr_status"] is not None:
            status = record["adr_status"]
            if status not in self.VALID_ADR_STATUSES:
                errors.append(
                    f"Invalid adr_status '{status}'. "
                    f"Valid values: {self.VALID_ADR_STATUSES}"
                )
        
        # Validate linked_code_nodes exist (if codebase_path is set)
        if "linked_code_nodes" in record and record["linked_code_nodes"]:
            linked_nodes = record["linked_code_nodes"]
            if isinstance(linked_nodes, list):
                for node_path in linked_nodes:
                    if not self._code_node_exists(node_path):
                        warnings.append(f"Linked code node not found: {node_path}")
        
        # Validate depends_on_adr format
        if "depends_on_adr" in record and record["depends_on_adr"]:
            depends_on = record["depends_on_adr"]
            if not isinstance(depends_on, list):
                errors.append("depends_on_adr must be a list")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info
        )
    
    def validate_dependency_chain(
        self,
        adr_id: str,
        depends_on: list[str],
        graph: DependencyGraph
    ) -> ValidationResult:
        """
        Validate that an ADR's dependency chain is valid.
        
        Args:
            adr_id: The ADR being validated
            depends_on: List of ADR IDs this ADR depends on
            graph: DependencyGraph containing known ADRs
        
        Returns:
            ValidationResult with any dependency errors
        """
        errors: list[str] = []
        warnings: list[str] = []
        
        # Check that all dependencies exist in the graph
        for dep_id in depends_on:
            if not graph.has_node(dep_id):
                errors.append(f"Dependency '{dep_id}' does not exist in graph")
        
        # Check for circular dependencies
        if depends_on:
            cycle_detected = self._check_cycle(adr_id, depends_on, graph)
            if cycle_detected:
                errors.append(f"Circular dependency detected involving {adr_id}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _check_cycle(
        self,
        start_node: str,
        dependencies: list[str],
        graph: DependencyGraph,
        visited: Optional[set[str]] = None
    ) -> bool:
        """Check for circular dependencies using DFS."""
        if visited is None:
            visited = set()
        
        if start_node in visited:
            return True
        
        visited.add(start_node)
        
        for dep_id in dependencies:
            if dep_id in graph.nodes:
                # Get dependencies of this dependency
                dep_deps = [
                    target for (source, target), _ in graph.edges.items()
                    if source == dep_id
                ]
                if self._check_cycle(dep_id, dep_deps, graph, visited.copy()):
                    return True
        
        return False
    
    def _code_node_exists(self, node_path: str) -> bool:
        """Check if a linked code node exists in the codebase."""
        if not self.codebase_path:
            return True  # Can't validate without codebase path
        
        full_path = Path(node_path)
        if not full_path.is_absolute():
            full_path = self.codebase_path / node_path
        
        return full_path.exists()
    
    def validate_parquet_file(self, file_path: Path) -> FileValidationReport:
        """
        Validate an entire parquet file for ADR compliance.
        
        Args:
            file_path: Path to the parquet file
        
        Returns:
            FileValidationReport with summary statistics
        """
        table = pq.read_table(file_path)
        total_records = table.num_rows
        
        compliant = 0
        with_warnings = 0
        with_errors = 0
        
        # Convert to list of dicts for iteration
        records = table.to_pylist()
        
        for record in records:
            result = self.validate_record(record)
            
            if result.is_valid and len(result.warnings) == 0:
                compliant += 1
            elif result.is_valid:
                with_warnings += 1
            else:
                with_errors += 1
        
        return FileValidationReport(
            file_path=str(file_path),
            total_records=total_records,
            compliant_records=compliant,
            records_with_warnings=with_warnings,
            records_with_errors=with_errors
        )
    
    def get_compliance_summary(self, file_path: Path) -> dict[str, Any]:
        """
        Get a compliance summary for a parquet file.
        
        Args:
            file_path: Path to the parquet file
        
        Returns:
            Dictionary with compliance summary statistics
        """
        report = self.validate_parquet_file(file_path)
        
        return {
            "file": report.file_path,
            "total_records": report.total_records,
            "compliant": report.compliant_records,
            "warnings": report.records_with_warnings,
            "errors": report.records_with_errors,
            "compliance_percentage": report.compliance_percentage,
        }
