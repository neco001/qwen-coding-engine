# Release Notes - v1.1.1

**Date:** 2026-04-09

## Critical Bug Fixes

### 1. Sparring Session File Location

**Problem:** Agent couldn't find sparring session results, showing error:
```
Error reading file .sparring_sessions/sp_80476010b689.md: ENOENT: no such file or directory
```

**Root Causes:**
- Session files saved as `.json` but agent looked for `.md`
- `SparringResponse.to_markdown()` showed session_id but NOT full file path
- SessionStore uses 4-tier directory resolution opaque to agent

**Solution:**
- Added `storage_dir` parameter to `SparringResponse.to_markdown()`
- Agent now sees full path: `{storage_dir}/{session_id}.json`
- Works with all storage tiers (env → user-level → fallback)

**Files Changed:**
- `src/qwen_mcp/engines/sparring_v2/models.py`: Added `storage_dir` parameter
- `src/qwen_mcp/tools.py`: Pass session store directory to response formatter

**Tests:**
- `tests/test_sparring_session_path.py`: 5 unit tests verifying path display

---

### 2. max_tokens=0 Truncation Fix (Sparring3 Output Cut Off)

**Problem:** All sparring3 (pro) responses were truncated mid-sentence.

**Root Cause:** Python falsy evaluation - `if max_tokens:` treated 0 as falsy, causing `max_tokens=0` (unlimited) setting to be ignored.

**Impact:** All tools using `max_tokens=0` configuration were getting default token limits instead of unlimited output.

**Solution:**
- Changed `if max_tokens:` to `if max_tokens is not None:` in `completions.py:69`
- This properly handles `max_tokens=0` as explicit "unlimited" setting

**Configuration Applied:**
- `MAX_TOKENS_CONFIG` set to 0 for all sparring modes (flash/full/pro)
- `max_tokens=0` for session supplement generation
- `max_tokens=0` for meta-analysis endpoint

**Tools Fixed:**
- ✅ qwen_architect
- ✅ qwen_coder
- ✅ qwen_audit
- ✅ qwen_sparring (all modes: sparring1, sparring2, sparring3)
- ✅ qwen_update_session_context

**Files Changed:**
- `src/qwen_mcp/completions.py`: Fixed max_tokens zero check
- `src/qwen_mcp/engines/sparring_v2/config.py`: Set MAX_TOKENS_CONFIG to 0
- `src/qwen_mcp/api.py`: Set max_tokens=0 for meta-analysis
- `src/qwen_mcp/engines/context_builder.py`: Set max_tokens=0 for session supplement

**Tests:**
- `tests/test_max_tokens_zero.py`: Verifies no truncation with max_tokens=0

---

## Summary

**Version:** 1.1.1  
**Commits:** 4 (fix + docs)  
**Files Changed:** 8  
**Tests Added:** 6

Both fixes are critical for production use:
1. Session file location fix enables proper sparring session continuity
2. max_tokens=0 fix prevents response truncation in all coding/architectural tools

---

Full Changelog: v1.1.0...v1.1.1

Contributors: @neco001
