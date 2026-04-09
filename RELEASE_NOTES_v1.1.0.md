# Release Notes - v1.1.0

## Major Features

### ADR Management System
- Schema-based Decision Record Parsing - Automatic validation and parsing of Architecture Decision Records
- ADR Linking & Dependencies - Track relationships between decisions with `depends_on_adr` field
- ADR Validation Engine - Comprehensive validation with metrics collection and trigger logic
- Knowledge Graph Integration - ADRs integrated with code knowledge graph for dependency tracking

### Sparring Engine v2
- Modular Cell Architecture - Separate Red/Blue/White cell implementations
- BudgetManager - Token budget management with circuit breaker protection
- Stage Metadata Tracking - Enhanced session state tracking with timeout protection
- Reduced Timeouts - Optimized for MCP 300s limit (discovery=45s, red/blue/white=60s)

### DecisionLogSync Engine
- Parquet-based Decision Tracking - 23-field schema for comprehensive decision logging
- BACKLOG.md Synchronization - Auto-sync decisions to project backlog
- CHANGELOG.md Generation - Automatic changelog updates from decision records
- Path Standardization - Unified `.decision_log/` directory for all decision logs

## Bug Fixes
- Fixed decision_log.parquet path inconsistency across components
- Removed decision log from git tracking (now properly ignored via .gitignore)
- Fixed SOS Sync to read/write from single canonical location

## Documentation
- Updated README.md with ADR tools and Sparring Engine v2 architecture
- Added SPARRING_V2.md documentation
- Updated CHANGELOG.md with v1.1.0 changes

## Technical Changes
- Version bump: 1.0.1 -> 1.1.0
- Migrated decision log storage to `.decision_log/` directory
- Updated `.gitignore` and `.git/info/exclude` for decision log

---

Full Changelog: v1.0.1...v1.1.0

Contributors: @neco001
