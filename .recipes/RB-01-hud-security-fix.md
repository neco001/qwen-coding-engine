# Recipe: RB-01 - HUD Security Fix

**Date:** 2026-03-30  
**Based on:** HUD Audit Report 2026-03-30  
**Architect Blueprint:** qwen_architect response 2026-03-30

---

## Traceability Matrix

| Finding ID | Severity | Spec Quote | Planned Function | File |
|------------|----------|------------|------------------|------|
| #1 | CRITICAL | "Query parameter is declared but never parsed" | Fix WebSocket endpoint to extract `project_id` from `websocket.query_params` | `server.py:311` |
| #2 | CRITICAL | "get_session_id() exists but is NEVER called" | Update all MCP tools to use `get_session_id()` with client_source | `server.py:43-196` |
| #3 | HIGH | "Client references may become stale" | Re-verify client membership under lock before sending | `telemetry.py:173-182` |
| #4 | HIGH | "Heartbeat counter unbounded" | Reset counter in `start_request()` | `telemetry.py:190-211` |
| #5 | HIGH | "Hardcoded config paths" | Use `process.env.USERPROFILE` for dynamic paths | `extension.js:59-61` |
| #6 | MEDIUM | "UI re-renders on every message" | Add JSON.stringify diff check before setState | `App.tsx:103-111` |
| #7 | MEDIUM | "Fade timer not cleared on session remove" | Clear timer when session is filtered out | `App.tsx:153-156` |
| #8 | MEDIUM | "Untracked fire-and-forget task" | Add done_callback with exception logging | `telemetry.py:227-232` |
| #9 | LOW | "threading.Lock in async code" | Replace with `asyncio.Lock()` | `telemetry.py:21` |
| #10 | LOW | "Display name not recomputed" | Compute display name from session_number on render | `App.tsx:195` |

---

## Implementation Phases

### Phase P0: Critical Security Fixes (IMMEDIATE)

#### P0-1: Fix WebSocket project_id Extraction
**File:** `src/qwen_mcp/server.py`  
**Line:** 311-313

```python
@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # FIX: Extract project_id from query string manually
    project_id = websocket.query_params.get("project_id", "default")
    await broadcaster.add_client(websocket, project_id=project_id)
```

#### P0-2: Integrate client_source into Session ID
**File:** `src/qwen_mcp/server.py`  
**Lines:** 43-196 (all MCP tools)

```python
# Add helper function
def _get_tool_session_id(ctx: Context, default_source: str = "default") -> str:
    """Extract client_source from context and generate session ID."""
    from qwen_mcp.specter.identity import get_session_id, get_or_create_instance_id
    
    # Try to extract client_source from MCP context headers
    client_source = default_source
    if hasattr(ctx, 'request_context') and ctx.request_context:
        # Check for X-Client-Source header (set by MCP host)
        client_source = ctx.request_context.headers.get('X-Client-Source', default_source)
    
    instance_id = get_or_create_instance_id()
    cwd = os.getcwd()
    return get_session_id(instance_id, client_source, cwd)

# Update all tool calls
project_id = _get_tool_session_id(ctx, default_source="mcp")
```

### Phase P1: High Severity Fixes

#### P1-1: Fix Race Condition in broadcast_state()
**File:** `src/qwen_mcp/specter/telemetry.py`  
**Lines:** 173-182

```python
# Step 2: Send messages - re-verify client under lock
disconnected = set()
async with self._lock:
    # Re-check clients are still valid before sending
    if project_id not in self._clients:
        return
    current_clients = set(self._clients[project_id])

tasks = [client.send_text(message) for client in current_clients]
results = await asyncio.gather(*tasks, return_exceptions=True)

for client, result in zip(current_clients, results):
    if isinstance(result, (ConnectionClosed, Exception)):
        disconnected.add(client)
```

#### P1-2: Reset Heartbeat Counter
**File:** `src/qwen_mcp/specter/telemetry.py`  
**Lines:** 190-211

```python
async def start_request(self, project_id: str = "default"):
    """Resets counters and buffers for a new high-level user request."""
    state = self.get_state(project_id)
    state["request_prompt_total"] = 0
    state["request_completion_total"] = 0
    state["thinking_buffer"] = ""
    state["content_buffer"] = ""
    state["request_images_total"] = 0
    state["status"] = "live"
    state["operation"] = "Processing request..."
    state["progress_percent"] = 0
    
    # FIX: Reset heartbeat counter on new request
    async with self._lock:
        self._heartbeat_counter = 0
    
    await self.broadcast_state({...}, project_id=project_id)
```

#### P1-3: Fix Hardcoded Config Paths
**File:** `vscode-extension/extension.js`  
**Lines:** 59-61

```javascript
_detectMcpSessions() {
    const sessions = [];
    const workspaceName = vscode.workspace.workspaceFolders?.[0]?.name || 'default';
    const workspaceHash = this._hashWorkspace(workspaceName);
    
    // FIX: Use environment variables for portable paths
    const userHome = process.env.USERPROFILE || process.env.HOME || '';
    
    // Path to Gemini/Antigravity MCP config
    const geminiConfigPath = path.join(userHome, '.gemini/antigravity/mcp_config.json');
    // Path to Roo Code MCP settings
    const rooConfigPath = path.join(userHome, 'AppData/Roaming/Antigravity/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json');
    
    // ... rest of detection logic
}
```

### Phase P2: Medium Severity Fixes

#### P2-1: Add State Diff Check
**File:** `specter-lens-ui/src/App.tsx`  
**Lines:** 103-111

```typescript
ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        if (data.type === 'heartbeat') return;
        
        setSessions(prev => prev.map(s => {
            if (s.id === session.id) {
                const newTelemetry = { ...s.telemetry, ...data };
                newTelemetry.is_live = data.status === 'live' || data.status === 'processing';
                
                // FIX: Skip update if nothing meaningful changed
                if (JSON.stringify(newTelemetry) === JSON.stringify(s.telemetry)) {
                    return s;
                }
                return { ...s, telemetry: newTelemetry };
            }
            return s;
        }));
        // ... rest of handler
    }
};
```

#### P2-2: Fix Fade Timer Memory Leak
**File:** `specter-lens-ui/src/App.tsx`  
**Lines:** 153-156

```typescript
return () => {
    sessions.forEach(s => s.ws?.close());
    fadeTimers.current.forEach((timer) => clearTimeout(timer));
    fadeTimers.current.clear(); // FIX: Clear the map
};
```

#### P2-3: Track Fire-and-Forget Tasks
**File:** `src/qwen_mcp/specter/telemetry.py`  
**Lines:** 227-232

```python
async def update_stream(self, thinking: str = "", content: str = "", project_id: str = "default"):
    """Appends to the current stream buffers and broadcasts."""
    state = self.get_state(project_id)
    
    if thinking:
        state["thinking_buffer"] += thinking
    if content:
        state["content_buffer"] += content
        state["status"] = "live"
        state["operation"] = "Generating response..."
    
    # FIX: Track task and add exception handler
    async def broadcast_with_logging():
        try:
            await self.broadcast_state({...}, project_id=project_id)
        except Exception as e:
            logger.warning(f"Stream broadcast failed: {e}")
    
    asyncio.create_task(broadcast_with_logging())
```

### Phase P3: Low Severity Fixes

#### P3-1: Replace threading.Lock with asyncio.Lock
**File:** `src/qwen_mcp/specter/telemetry.py`  
**Line:** 21

```python
class SessionMapper:
    def __init__(self):
        self._project_to_session: Dict[str, int] = {}
        self._next_session_number: int = 1
        self._lock = asyncio.Lock()  # FIX: Use asyncio.Lock for async code
    
    async def get_or_create_session_number(self, project_id: str) -> int:
        """Get existing session number or assign a new one."""
        async with self._lock:  # FIX: async with
            if project_id not in self._project_to_session:
                self._project_to_session[project_id] = self._next_session_number
                self._next_session_number += 1
            return self._project_to_session[project_id]
```

#### P3-2: Dynamic Display Name
**File:** `specter-lens-ui/src/App.tsx`  
**Line:** 195

```typescript
// Compute display name from session_display_id if available
const getDisplayName = (session: Session) => {
    if (session.telemetry.session_display_name) {
        return session.telemetry.session_display_name;
    }
    if (session.telemetry.session_display_id) {
        return `Sesja ${session.telemetry.session_display_id}`;
    }
    return session.name;
};

// In render:
{getDisplayName(session)}
```

---

## Test Verification Checklist

- [ ] Two VSCode instances with same workspace show DIFFERENT telemetry data
- [ ] Gemini tab shows ONLY Gemini tool calls
- [ ] Roo Code tab shows ONLY Roo Code tool calls
- [ ] No UI flickering during 5-minute observation period
- [ ] Heartbeat counter resets to 0 on `qwen_init_request()`
- [ ] Extension works on different Windows usernames (not hardcoded path)
- [ ] No memory leak after 100 session connect/disconnect cycles
- [ ] No unhandled promise rejections in browser console
- [ ] WebSocket reconnection after server restart preserves session isolation

---

## Files to Modify Summary

| File | Priority | Changes |
|------|----------|---------|
| `src/qwen_mcp/server.py` | P0 | Fix WebSocket endpoint, add `_get_tool_session_id()` helper |
| `src/qwen_mcp/specter/telemetry.py` | P0, P1, P2, P3 | Fix broadcast_state, heartbeat reset, task tracking, asyncio.Lock |
| `src/qwen_mcp/specter/identity.py` | P0 | Export `get_session_id` for use in server.py |
| `vscode-extension/extension.js` | P1 | Fix hardcoded paths with environment variables |
| `specter-lens-ui/src/App.tsx` | P2, P3 | Add state diff, fix timer cleanup, dynamic display name |

---

## Archive Instructions

After implementation:
1. Run all tests: `pytest tests/ -v`
2. Run build check: `cd specter-lens-ui && npm run build`
3. Create commit: `git commit -m "fix(HUD): Critical session isolation and security fixes"`
4. Move this recipe: `.recipes/RB-01-hud-security-fix.md` → `.implemented/[YYYYMMDD_HHMMSS]_impl_RB-01.md`
