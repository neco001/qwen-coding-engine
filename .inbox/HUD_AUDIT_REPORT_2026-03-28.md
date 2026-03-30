# HUD (Specter Lens) Audit Report

**Date:** 2026-03-28  
**Auditor:** qwen_audit (Lachman Protocol)  
**Scope:** Session/Instance identification, Tab differentiation, Heartbeat flickering

---

## Executive Summary

All three issues stem from **identity and message protocol mismatches** between the VSCode extension, React UI, and Python telemetry server. The system has no unified concept of "session identity" across the stack.

| Issue | Root Cause | Priority | Fix Complexity |
|-------|------------|----------|----------------|
| Heartbeat flickering | Message format mismatch | **P1** | Simple |
| Wrong VSCode instance | Broadcast-to-all + no instance ID | **P2** | Medium |
| Tab differentiation | projectId namespace disconnect | **P3** | Complex |

---

## Issue 1: Wrong VSCode Instance Receiving Data

### Problem Description
User executes `qwen_sparring` in VSCode instance 2, but HUD shows data in instance 1.

### Root Cause Analysis

**1. Broadcast-to-all anti-pattern in [`telemetry.py`](src/qwen_mcp/specter/telemetry.py:108-150):**

```python
async def broadcast_state(self, payload: dict, project_id: str = "default") -> None:
    async with self._lock:
        # BUG: Updates ALL project states regardless of project_id
        for proj_id in self._project_states:
            state = self._project_states[proj_id]
            state.update(payload)
        
        # BUG: Sends to ALL clients across ALL projects
        all_clients = []
        for proj_clients in self._clients.values():
            all_clients.extend(list(proj_clients))
```

The `project_id` parameter is **ignored** - all states are updated and all clients receive the message.

**2. Insufficient instance identity in [`identity.py`](src/qwen_mcp/specter/identity.py:4-21):**

```python
def get_current_project_id() -> str:
    cwd = os.getcwd()
    normalized_cwd = os.path.realpath(cwd).lower()
    project_hash = hashlib.sha256(normalized_cwd.encode()).hexdigest()[:8]
    return f"project_{project_hash}"
```

All VSCode instances opening the **same workspace folder** generate the **SAME project_id**. The identity system cannot distinguish between different VSCode instances - it only distinguishes between different workspace folders.

### Fix Strategy

**[SIMPLE - Immediate]** Fix broadcast logic to respect project_id:

```python
async def broadcast_state(self, payload: dict, project_id: str = "default") -> None:
    async with self._lock:
        # Only update the specific project state
        if project_id in self._project_states:
            self._project_states[project_id].update(payload)
        
        # Only send to clients of that project
        if project_id in self._clients:
            clients = list(self._clients[project_id])
    # Send outside lock
    for client in clients:
        await client.send_text(json.dumps(self._project_states[project_id]))
```

**[COMPLEX - Complete]** Add instance-level identity:
- VSCode extension generates unique `instance_id` (UUID) at startup
- Pass `instance_id` through MCP tool calls as metadata
- Include `instance_id` in WebSocket connection query params
- Server maintains `instance_id -> client` mapping

---

## Issue 2: Gemini vs Roo Code Tab Differentiation

### Problem Description
HUD has two tabs (`gemini` and `roo code`) but cannot distinguish which source sent the telemetry data.

### Root Cause Analysis

**Complete projectId namespace disconnect:**

**Client-side (extension.js + App.tsx):**
```javascript
// extension.js line 67
projectId: this._hashProjectId('gemini-' + workspace.name)  // e.g., "a3f2b1c4"

// App.tsx line 86
const ws = new WebSocket(`ws://127.0.0.1:8878/ws/telemetry?project_id=${session.projectId}`);
```

**Server-side (identity.py):**
```python
# Returns "project_{cwd_hash}" - completely different namespace!
project_id = get_current_project_id()  # e.g., "project_7d8e9f0a"
```

When `qwen_sparring` is invoked, the server uses `get_current_project_id()` which returns `project_{cwd_hash}` - **completely different** from the client's `hash('gemini-workspace')` or `hash('roocode-workspace')`. These will **NEVER match**!

### Fix Strategy

**[SIMPLE]** Pass client source through MCP tool calls:

```python
# tools.py - add client_source parameter
async def qwen_sparring(ctx, prompt: str, client_source: str = "default"):
    project_id = get_client_project_id(client_source, os.getcwd())
```

**[COMPLEX]** Full session negotiation protocol:
- MCP tools receive `client_source` metadata from VSCode extension context
- Server maintains separate state buckets per `(workspace, client_source)` tuple
- WebSocket connections specify both `workspace_id` and `source_id`

---

## Issue 3: Heartbeat Causing UI Flickering

### Problem Description
HUD flickers every 5 seconds due to heartbeat messages triggering full React re-renders.

### Root Cause Analysis

**Message format mismatch:**

**Server sends (telemetry.py line 222-230):**
```python
await self.broadcast_state({"heartbeat": self._heartbeat_counter}, project_id=project_id)
# Message: {"heartbeat": 123, ...other_state_fields}
```

**UI expects (App.tsx line 97):**
```typescript
if (data.type === 'heartbeat') return;  // <-- FILTER ATTEMPT
```

The heartbeat message has **NO `type` field**, so the filter `if (data.type === 'heartbeat') return;` **NEVER triggers**. Every heartbeat causes:
1. `setSessions(prev => prev.map(...))` - full state update
2. React re-render of entire HUD
3. Visual flickering

### Fix Strategy

**[SIMPLE - Immediate]** Fix heartbeat message format:

```python
async def _send_heartbeat(self):
    self._heartbeat_counter += 1
    heartbeat_msg = {"type": "heartbeat", "count": self._heartbeat_counter}
    # Send ONLY to clients, do NOT call broadcast_state()
    for project_id in list(self._clients.keys()):
        for client in list(self._clients[project_id]):
            await client.send_text(json.dumps(heartbeat_msg))
```

**Alternative [SIMPLE]** Fix UI filter:

```typescript
// App.tsx - fix filter condition
if (data.heartbeat !== undefined && Object.keys(data).length === 1) return;
```

**Recommended:** Fix server-side message format for protocol consistency.

---

## Implementation Priority

### Phase 1: Immediate Relief (P1)
1. Fix heartbeat message format in [`telemetry.py`](src/qwen_mcp/specter/telemetry.py:222-230)
   - Add `type: "heartbeat"` field
   - Send directly to clients, don't call `broadcast_state()`

### Phase 2: Stop Cross-Instance Bleeding (P2)
2. Fix [`broadcast_state()`](src/qwen_mcp/specter/telemetry.py:108-150) to respect `project_id`
   - Only update specified project state
   - Only send to clients of that project

### Phase 3: Complete Solution (P3)
3. Implement unified identity system:
   - Add `instance_id` generation in [`extension.js`](vscode-extension/extension.js)
   - Add `client_source` parameter to MCP tools in [`tools.py`](src/qwen_mcp/tools.py)
   - Update [`identity.py`](src/qwen_mcp/specter/identity.py) to support `(workspace, instance, source)` tuple
   - Update WebSocket handshake in [`App.tsx`](specter-lens-ui/src/App.tsx:86)

---

## Files to Modify

| File | Changes |
|------|---------|
| [`src/qwen_mcp/specter/telemetry.py`](src/qwen_mcp/specter/telemetry.py:1) | Fix `_send_heartbeat()` message format, fix `broadcast_state()` to respect project_id |
| [`src/qwen_mcp/specter/identity.py`](src/qwen_mcp/specter/identity.py:1) | Add `get_client_project_id(client_source, cwd)` function |
| [`src/qwen_mcp/tools.py`](src/qwen_mcp/tools.py:1) | Add `client_source` parameter extraction from MCP context |
| [`vscode-extension/extension.js`](vscode-extension/extension.js:1) | Generate unique `instance_id`, pass to webview |
| [`specter-lens-ui/src/App.tsx`](specter-lens-ui/src/App.tsx:1) | Include `instance_id` in WebSocket connection |

---

## Test Verification

After fixes, verify:
1. Heartbeat messages logged in browser console show `type: "heartbeat"`
2. Two VSCode instances with same workspace show different data in their HUDs
3. `qwen_sparring` called from Gemini shows in Gemini tab, Roo Code shows in Roo Code tab