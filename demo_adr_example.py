#!/usr/bin/env python3
"""
ADR Integration Demo - Practical Examples

This script demonstrates what the ADR integration actually DOES
in real-world terms.

Run: .\\.venv\\Scripts\\python.exe demo_adr_example.py
"""

import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def demo_1_schema_extension():
    """Phase 1: See the new ADR fields in the schema."""
    print("\n" + "="*60)
    print("DEMO 1: Schema Extension - What changed?")
    print("="*60)
    
    from decision_log.decision_schema import DECISION_LOG_SCHEMA
    
    print(f"\nTotal fields in schema: {len(DECISION_LOG_SCHEMA)}")
    print("\nNEW ADR Fields (added in Phase 1):")
    
    adr_fields = [
        "adr_status", "adr_context", "adr_consequences",
        "adr_alternatives", "linked_code_nodes", "depends_on_adr"
    ]
    
    for field in DECISION_LOG_SCHEMA:
        if field.name in adr_fields:
            nullable = "NULLABLE" if field.nullable else "REQUIRED"
            print(f"  • {field.name:25} → {field.type} ({nullable})")
    
    print("\nWhy this matters:")
    print("  Before: Decision log could only store basic decision metadata")
    print("  After:  You can store FULL architectural decision records")
    print("          with context, consequences, alternatives, and code links")


def demo_2_tree_sitter_parser():
    """Phase 2: Parse Python code and detect patterns."""
    print("\n" + "="*60)
    print("DEMO 2: Tree-sitter Parser - Reading code structure")
    print("="*60)
    
    from graph.tree_sitter_parser import TreeSitterAnalyzer
    
    analyzer = TreeSitterAnalyzer()
    
    print(f"\nSupported languages: {analyzer.SUPPORTED_LANGUAGES}")
    print("\nWhat it can detect:")
    print("  • Functions and classes")
    print("  • Architectural patterns (Repository, Factory, Service, etc.)")
    print("  • Decision points in code (if/else, config choices)")
    
    # Example: parse this demo file
    print(f"\nTrying to parse: {Path(__file__).absolute()}")
    
    result = asyncio.run(analyzer.parse(Path(__file__), "python"))
    
    print(f"  Functions found: {len(result.get('functions', []))}")
    print(f"  Classes found: {len(result.get('classes', []))}")
    
    patterns = asyncio.run(analyzer.detect_patterns(result))
    print(f"  Architectural patterns detected: {len(patterns)}")
    
    if patterns:
        for p in patterns[:3]:
            print(f"    - {p.get('pattern', 'Unknown')}")


def demo_3_smart_code_linker():
    """Phase 3: Link an ADR to relevant code."""
    print("\n" + "="*60)
    print("DEMO 3: Smart Code Linker - ADR → Code mapping")
    print("="*60)
    
    from graph.adr_linker import SmartCodeLinker, CodeLink
    
    linker = SmartCodeLinker(
        codebase_path=Path(__file__).parent,
        search_tool="python"  # Use Python fallback (no ripgrep needed)
    )
    
    # Example ADR content
    adr_content = """
    # ADR-001: Use PostgreSQL for Data Storage
    
    ## Context
    We need a reliable database for user authentication and session management.
    The authentication module requires persistent storage with ACID guarantees.
    
    ## Decision
    Use PostgreSQL with asyncpg driver for all database operations.
    We will use Repository pattern for database access.
    
    ## Consequences
    - Better data integrity
    - Slight performance overhead
    - Need connection pooling
    """
    
    print("\nSample ADR: 'Use PostgreSQL for Data Storage'")
    print("\nExtracting keywords from ADR...")
    
    keywords = linker.extract_keywords(adr_content)
    print(f"  Keywords: {keywords[:8]}...")  # First 8 keywords
    
    print("\nSearching codebase for relevant files...")
    
    links = asyncio.run(linker.link_adr_to_code(adr_content, max_results=5))
    
    print(f"\nFound {len(links)} relevant code locations:")
    for link in links[:5]:
        print(f"  • {link.file_path}:{link.line}")
        print(f"    Relevance: {link.relevance_score:.2f}")
        snippet = link.snippet[:60].replace('\n', ' ')
        print(f"    Snippet: {snippet}...")


def demo_4_knowledge_graph():
    """Phase 4: Track dependencies between ADRs."""
    print("\n" + "="*60)
    print("DEMO 4: Knowledge Graph - ADR dependency tracking")
    print("="*60)
    
    from graph.knowledge_graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    # Create sample ADR network
    print("\nCreating sample ADR network:")
    print("  ADR-001: Database Choice (PostgreSQL)")
    print("  ADR-002: Authentication System")
    print("  ADR-003: Session Management")
    print("  ADR-004: Caching Strategy")
    
    graph.add_node("ADR-001", {"title": "Database Choice", "status": "accepted"})
    graph.add_node("ADR-002", {"title": "Authentication System", "status": "accepted"})
    graph.add_node("ADR-003", {"title": "Session Management", "status": "proposed"})
    graph.add_node("ADR-004", {"title": "Caching Strategy", "status": "proposed"})
    
    # Add dependencies
    print("\nAdding dependencies:")
    print("  ADR-002 depends_on ADR-001 (auth needs DB)")
    print("  ADR-003 depends_on ADR-001 (sessions need DB)")
    print("  ADR-003 depends_on ADR-002 (sessions need auth)")
    
    graph.add_edge("ADR-002", "ADR-001", "depends_on")
    graph.add_edge("ADR-003", "ADR-001", "depends_on")
    graph.add_edge("ADR-003", "ADR-002", "depends_on")
    
    print(f"\nGraph stats: {len(graph)} nodes, {len(graph.edges)} edges")
    
    # Query dependencies
    print("\nQuery: What does ADR-003 depend on?")
    deps = graph.get_dependencies("ADR-003")
    print(f"  Answer: {deps}")
    
    print("\nQuery: What ADRs depend on ADR-001 (Database)?")
    dependents = graph.get_dependents("ADR-001")
    print(f"  Answer: {dependents}")
    print("  → If you change the database, these 2 ADRs are affected!")
    
    # Check for cycles
    print("\nChecking for circular dependencies...")
    cycles = graph.detect_cycles()
    print(f"  Cycles found: {len(cycles)}")
    
    # Export
    print("\nExport to JSON:")
    export_data = graph.to_dict()
    print(f"  Nodes: {len(export_data['nodes'])}")
    print(f"  Edges: {len(export_data['edges'])}")


def demo_5_validator():
    """Phase 5: Validate ADR compliance."""
    print("\n" + "="*60)
    print("DEMO 5: ADR Validator - Compliance checking")
    print("="*60)
    
    from validator.adr_validator import ADRValidator, ValidationResult
    
    validator = ADRValidator()
    
    # Example 1: Complete ADR record
    print("\n1. Validating COMPLETE ADR record:")
    complete_record = {
        "decision_id": "ADR-001",
        "timestamp": "2024-01-15T10:00:00Z",
        "session_id": "session-123",
        "decision_type": "architecture",
        "title": "Use PostgreSQL for Data Storage",
        "advice": "PostgreSQL provides ACID compliance",
        "complexity": "medium",
        "risk_score": 0.3,
        "tags": ["database", "storage"],
        "adr_status": "accepted",
        "adr_context": "Need reliable database for user auth",
        "adr_consequences": "Better data integrity, slight performance cost",
        "adr_alternatives": "MongoDB, Redis",
        "linked_code_nodes": [],
        "depends_on_adr": [],
    }
    
    result = validator.validate_record(complete_record)
    print(f"  Valid: {result.is_valid}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    
    # Example 2: Incomplete ADR record
    print("\n2. Validating INCOMPLETE ADR record (missing ADR fields):")
    incomplete_record = {
        "decision_id": "ADR-002",
        "timestamp": "2024-01-15T11:00:00Z",
        "session_id": "session-123",
        "decision_type": "architecture",
        "title": "Some Decision",
        "advice": "Some advice",
        "complexity": "low",
        "risk_score": 0.1,
        # Missing all ADR fields!
    }
    
    result = validator.validate_record(incomplete_record)
    print(f"  Valid: {result.is_valid}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    if result.warnings:
        print(f"  Warning example: {result.warnings[0]}")
    
    # Example 3: Invalid status
    print("\n3. Validating INVALID adr_status:")
    invalid_status = {
        "decision_id": "ADR-003",
        "timestamp": "2024-01-15T12:00:00Z",
        "session_id": "session-123",
        "decision_type": "architecture",
        "title": "Bad Status",
        "advice": "Advice",
        "complexity": "low",
        "risk_score": 0.1,
        "adr_status": "maybe_someday",  # Invalid!
    }
    
    result = validator.validate_record(invalid_status)
    print(f"  Valid: {result.is_valid}")
    print(f"  Errors: {len(result.errors)}")
    if result.errors:
        print(f"  Error: {result.errors[0]}")
    
    print("\nValid ADR statuses:")
    print(f"  {validator.VALID_ADR_STATUSES}")


def main():
    print("\n" + "="*60)
    print("  ADR INTEGRATION DEMO - The Lachman Protocol")
    print("="*60)
    print("\nThis demo shows what each phase ACTUALLY does.")
    print("No bullshit, just working code.\n")
    
    demo_1_schema_extension()
    demo_2_tree_sitter_parser()
    demo_3_smart_code_linker()
    demo_4_knowledge_graph()
    demo_5_validator()
    
    print("\n" + "="*60)
    print("  END OF DEMO")
    print("="*60)
    print("\nTL;DR - What you now have:")
    print("""
    1. Schema: Can store full ADR records in decision_log.parquet
    2. Parser:  Reads your code and finds architectural patterns
    3. Linker:  Connects decisions to actual code files
    4. Graph:   Tracks which decisions depend on which
    5. Validator: Checks if your ADRs are complete and valid
    
    Next steps (if you want):
    - Phase 6: Async pipeline to auto-enrich decisions
    - Phase 7: Integration tests + docs
    
    Or... we can stop here and see if this is even useful for you.
    """)


if __name__ == "__main__":
    main()
