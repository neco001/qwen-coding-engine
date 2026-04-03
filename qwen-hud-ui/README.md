# ⚠️ QWEN-HUD-UI (MCP Telemetry HUD)

## Status: UNDER REPAIR

The HUD streaming functionality is currently broken. The MCP server works fully without it.

### What Works:
- ✅ MCP server (`qwen_architect`, `qwen_coder`, `qwen_audit`, `qwen_sparring`)
- ✅ Model registry with billing modes
- ✅ DuckDB billing reports (`qwen_usage_report`)
- ✅ Telemetry WebSocket server (port 8878)

### What's Broken:
- ❌ VSCode extension not displaying token streaming
- ❌ Real-time telemetry not visible in HUD

### Workaround:
Use `qwen_usage_report()` tool to get billing/token usage data from DuckDB.

---

## 🛠️ For Contributors

**Looking for contributors!** If you can fix WebSocket streaming in VSCode extensions, please open a PR.

### Known Issues:
1. WebSocket connection on `ws://127.0.0.1:8878/ws/telemetry` not being established properly
2. VSCode extension may not be receiving heartbeat messages from `TelemetryBroadcaster`

### Architecture:
```
┌─────────────────────────────────────────────────────────────┐
│ qwen-hud-ui/ (React/Vite)                                   │
│   → Built to ../vscode-extension/dist/assets/               │
│   → Loaded as webview in VSCode extension                   │
│   → Connects to ws://127.0.0.1:8878/ws/telemetry           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ MCP Server (src/qwen_mcp/specter/telemetry.py)              │
│   → TelemetryBroadcaster on port 8878                       │
│   → Broadcasts: token usage, streaming content, model info  │
└─────────────────────────────────────────────────────────────┘
```

### Build Instructions:
```bash
cd qwen-hud-ui
npm install
npm run build
# Output: ../vscode-extension/dist/assets/
```

---

## 📦 Package Info

| Attribute | Value |
|-----------|-------|
| **Name** | `qwen-hud-ui` |
| **Type** | React + Vite + TypeScript |
| **Dependencies** | react, react-dom, framer-motion, lucide-react, react-markdown |

---

*Part of The Lachman Protocol: Qwen Engineering Engine*
