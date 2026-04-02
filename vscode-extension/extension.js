const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const os = require('os');

/**
 * Generate unique instance ID for this VSCode window.
 * Uses crypto random bytes to create 8-character hex string.
 */
function generateInstanceId() {
    return crypto.randomBytes(4).toString('hex');
}

function activate(context) {
    // P3-1 FIX: Generate unique instance ID for this VSCode window
    // Stored in globalState to persist across extension reloads
    let instanceId = context.globalState.get('instanceId');
    if (!instanceId) {
        // Try vscode.env.sessionId first (VSCode-provided), fallback to random
        instanceId = vscode.env.sessionId || generateInstanceId();
        context.globalState.update('instanceId', instanceId);
        console.log(`[SPECTER] Generated new instance ID: ${instanceId}`);
    } else {
        console.log(`[SPECTER] Using existing instance ID: ${instanceId}`);
    }

    const provider = new SpecterViewProvider(context.extensionUri, instanceId);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('specter-qwen-cockpit-v1', provider)
    );
}

async function fetchServerProjectId() {
    // Query the telemetry server's HTTP endpoint to get its project_id
    const http = require('http');

    return new Promise((resolve) => {
        const req = http.get('http://127.0.0.1:8878/', (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    if (json.project_id) {
                        resolve(json.project_id);
                    } else {
                        resolve(null);
                    }
                } catch (e) {
                    resolve(null);
                }
            });
        });

        req.on('error', () => resolve(null));
        req.setTimeout(2000, () => {
            req.destroy();
            resolve(null);
        });
    });
}

class SpecterViewProvider {
    constructor(extensionUri, instanceId) {
        this._extensionUri = extensionUri;
        this._instanceId = instanceId;  // P3-2 FIX: Store instance ID for session generation
        this._sessions = null;  // P3-9 FIX: Cache sessions to avoid async issues
        console.log(`[SPECTER] Provider initialized with instanceId: ${instanceId}`);
        
        // P3-10 FIX: Pre-detect sessions synchronously at construction time
        this._sessions = this._detectMcpSessionsSync();
    }

    _detectMcpSessionsSync() {
        // P3-11 FIX: Synchronous version for use in HTML generation
        const sessions = [];
        const workspaceName = vscode.workspace.workspaceFolders?.[0]?.name || 'default';
        const workspaceHash = this._hashWorkspace(workspaceName);
        const userHome = process.env.USERPROFILE || process.env.HOME || os.homedir();

        console.log(`[SPECTER] Detecting MCP sessions (sync) for instance: ${this._instanceId}`);

        // Path to Roo Code MCP settings
        const rooConfigPath = path.join(userHome, 'AppData/Roaming/Antigravity/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json');
        // Path to Gemini/Antigravity MCP config
        const geminiConfigPath = path.join(userHome, '.gemini/antigravity/mcp_config.json');

        // Check Roo Code config first
        try {
            if (fs.existsSync(rooConfigPath)) {
                const rooConfig = JSON.parse(fs.readFileSync(rooConfigPath, 'utf8'));
                const servers = rooConfig.mcpServers || {};
                const hasRooQwenServer = Object.keys(servers).some(key =>
                    key.toLowerCase().includes('qwen') && key.toLowerCase().includes('_roo')
                );

                if (hasRooQwenServer) {
                    const projectId = `${this._instanceId}_roocode_${workspaceHash}`;
                    console.log(`[SPECTER] Found Roo Code MCP session: ${projectId}`);
                    sessions.push({
                        id: 'roocode',
                        name: 'Roo Code',
                        projectId: projectId,
                        clientSource: 'roocode'
                    });
                }
            }
        } catch (e) {
            console.log('[SPECTER] Failed to read Roo config:', e.message);
        }

        // Check Gemini config
        try {
            if (fs.existsSync(geminiConfigPath)) {
                const geminiConfig = JSON.parse(fs.readFileSync(geminiConfigPath, 'utf8'));
                const servers = geminiConfig.mcpServers || {};
                const hasQwenServer = Object.keys(servers).some(key =>
                    key.toLowerCase().includes('qwen') && !key.toLowerCase().includes('_roo')
                );

                if (hasQwenServer) {
                    const projectId = `${this._instanceId}_gemini_${workspaceHash}`;
                    console.log(`[SPECTER] Found Gemini MCP session: ${projectId}`);
                    sessions.push({
                        id: 'gemini',
                        name: 'Gemini',
                        projectId: projectId,
                        clientSource: 'gemini'
                    });
                }
            }
        } catch (e) {
            console.log('[SPECTER] Failed to read Gemini config:', e.message);
        }

        // Fallback
        if (sessions.length === 0) {
            const projectId = `${this._instanceId}_default_${workspaceHash}`;
            console.log(`[SPECTER] No MCP config found, using default session: ${projectId}`);
            sessions.push({
                id: 'default',
                name: 'Default',
                projectId: projectId,
                clientSource: 'default'
            });
        }

        console.log(`[SPECTER] Detected ${sessions.length} session(s):`, sessions.map(s => s.id).join(', '));
        return sessions;
    }

    resolveWebviewView(webviewView) {
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        try {
            webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
            
            // Set up message handling for session updates
            webviewView.webview.onDidReceiveMessage(async (message) => {
                if (message.type === 'getSessionConfig') {
                    const sessions = await this._detectMcpSessions();
                    webviewView.webview.postMessage({
                        type: 'sessionConfig',
                        sessions: sessions
                    });
                }
            });
        } catch (e) {
            webviewView.webview.html = `<html><body><pre>FAILED TO LOAD HUD: ${e.message}</pre></body></html>`;
        }
    }

    _hashWorkspace(input) {
        return crypto.createHash('sha256').update(input.toLowerCase()).digest('hex').substring(0, 8);
    }

    _getHtmlForWebview(webview) {
        // Use joinPath for bulletproof URI generation
        const assetsUri = vscode.Uri.joinPath(this._extensionUri, 'dist', 'assets');
        const assetsPath = assetsUri.fsPath;

        if (!fs.existsSync(assetsPath)) {
            return `<html><body style="background-color: #0a0a0a; color: #facc15; font-family: monospace; padding: 20px;">
                <h1>[SPECTER] Critical Error</h1>
                <p>Assets folder not found: ${assetsPath}</p>
                <p>Please check your build/installation.</p>
            </body></html>`;
        }

        const files = fs.readdirSync(assetsPath);
        const jsFile = files.find(f => f.endsWith('.js'));
        const cssFile = files.find(f => f.endsWith('.css'));

        if (!jsFile || !cssFile) {
            return `<html><body style="background-color: #0a0a0a; color: #facc15; font-family: monospace; padding: 20px;">
                <h1>[SPECTER] Entry Point Error</h1>
                <p>JS or CSS not found in: ${assetsPath}</p>
            </body></html>`;
        }

        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(assetsUri, jsFile));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(assetsUri, cssFile));

        // P3-12 FIX: Use cached sessions (computed synchronously in constructor)
        const sessions = this._sessions || [];
        const sessionsJson = JSON.stringify(sessions);

        console.log(`[SPECTER] Injecting sessions into HTML: ${sessionsJson}`);

        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src ${webview.cspSource} 'unsafe-inline' 'unsafe-eval' blob:; connect-src ws://127.0.0.1:8878 http://127.0.0.1:8878 ws://localhost:8878 http://localhost:8878;">
                <link href="${styleUri}" rel="stylesheet">
                <title>Specter Cockpit</title>
                <script>
                    console.log('[HUD] Setting INITIAL_SESSIONS:', ${sessionsJson});
                    window.INITIAL_SESSIONS = ${sessionsJson};
                </script>
                <style>
                    body { padding: 0; margin: 0; overflow: hidden; background-color: #0a0a0a; color: #333; font-family: monospace; }
                    #root { height: 100vh; position: relative; z-index: 1; }
                    #debug-fallback { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #facc15; font-size: 10px; letter-spacing: 2px; opacity: 0.5; z-index: 0; text-align: center; width: 100%; transition: opacity 0.5s; }
                    .error-log { color: #ef4444; font-size: 8px; margin-top: 10px; opacity: 1 !important; }
                </style>
                <script>
                    window.onerror = function(msg, url, line, col, error) {
                        const fb = document.getElementById('debug-fallback');
                        if (fb) {
                            fb.innerHTML += '<div class="error-log">ERR: ' + msg + '<br>Line: ' + line + '</div>';
                            fb.style.opacity = '1';
                        }
                        return false;
                    };
                </script>
            </head>
            <body>
                <div id="debug-fallback">
                    [ SPECTER_LENS_INITIALIZING... ]
                </div>
                <div id="root"></div>
                <script type="module" src="${scriptUri}"></script>
            </body>
            </html>`;
    }
}

function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}

function deactivate() {
    // P3-4 FIX: Clean up resources on extension deactivate
    console.log('[SPECTER] Extension deactivated');
}

module.exports = {
    activate,
    deactivate
};
