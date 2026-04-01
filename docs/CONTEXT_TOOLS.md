# Context Tools User Guide

## Overview

Context Tools automate the creation and maintenance of project documentation,
ensuring session continuity and reducing onboarding time.

## Tools

### qwen_init_context

**Purpose**: Generate initial project context files at session start.

**Usage**:
```python
qwen_init_context_tool(workspace_root=".")
```

**Output Files**:
- `.context/_PROJECT_CONTEXT.md`: Tech stack, structure, conventions
- `.context/_DATA_CONTEXT.md`: Data sources, schemas, pipelines

**When to Use**:
- First session with a new project
- After major architectural changes
- When onboarding new team members

### qwen_update_session_context

**Purpose**: Capture session insights for continuity.

**Usage**:
```python
qwen_update_session_context_tool(
    session_summary="""
    Session 2026-03-31:
    - Implemented user authentication
    - Added JWT token validation
    - Updated database schema
    """,
    workspace_root="."
)
```

**Output File**:
- `.context/_SESSION_SUPPLEMENT.md`: Session history and recommendations

**When to Use**:
- End of every session
- Before switching to different task
- When making significant decisions

## Best Practices

1. **Run init_context at session start** - Ensures fresh context
2. **Always update at session end** - Maintains continuity
3. **Be specific in summaries** - Include decisions, not just actions
4. **Review before next session** - Read _SESSION_SUPPLEMENT.md first

## Example Workflow

```
Session Start:
1. Call qwen_init_context_tool()
2. Review generated _PROJECT_CONTEXT.md
3. Review _SESSION_SUPPLEMENT.md from previous session

During Session:
1. Work on tasks
2. Take notes separately

Session End:
1. Call qwen_update_session_context_tool(session_summary="...")
2. Verify _SESSION_SUPPLEMENT.md updated
```

## Generated File Structure

```
project/
├── .context/
│   ├── _PROJECT_CONTEXT.md    # Static project knowledge
│   ├── _DATA_CONTEXT.md       # Data sources, schemas, pipelines
│   └── _SESSION_SUPPLEMENT.md # Dynamic session notes
```

## _PROJECT_CONTEXT.md Sections

1. **Project Overview** - Name, purpose, architecture
2. **Tech Stack Summary** - Runtime, frameworks, libraries
3. **Directory Structure** - Key directories and entry points
4. **Development Workflow** - Build/test commands, scripts
5. **Key Conventions** - Code style, testing patterns

## _DATA_CONTEXT.md Sections

1. **Data Sources** - Databases, files, APIs
2. **Schema Overview** - Tables, entities, relationships
3. **Data Pipelines** - ETL processes, transformations
4. **Data Access Patterns** - Query utilities, caching

## _SESSION_SUPPLEMENT.md Sections

1. **Current Session Summary** - Date, objectives, accomplishments
2. **Decisions Made** - Architectural, tool choices, config changes
3. **Open Questions** - Unresolved issues, technical debt
4. **Next Session Recommendations** - Priority tasks, files to review

## Technical Details

### Scout Analysis

The `qwen_init_context` tool uses ScoutEngine to analyze codebase complexity
before deciding whether to use Swarm parallel analysis:

- **Low complexity**: Single LLM call per context file
- **High complexity**: Swarm parallel decomposition for faster analysis

### Atomic Writes

All context files are written atomically using temp + rename pattern
to prevent corruption during write operations.

### Error Handling

Both tools include:
- Input validation (workspace path existence, non-empty summaries)
- Try/except blocks with detailed error logging
- Graceful error messages returned to users

## Troubleshooting

### "Workspace path does not exist"

Ensure the `workspace_root` parameter points to a valid directory:
```python
qwen_init_context_tool(workspace_root="/path/to/project")
```

### "Session summary cannot be empty"

Provide a non-empty summary:
```python
qwen_update_session_context_tool(session_summary="Fixed bug in auth module")
```

### Context files not generated

Check logs for errors. The tools require:
1. Valid DashScope API key configured
2. Write permissions to workspace directory
3. Network connectivity for LLM calls