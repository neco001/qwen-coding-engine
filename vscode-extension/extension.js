const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const os = require('os');

function activate(context) {
    // Query server's HTTP endpoint to get the server's project_id (ensures instanceId sync)
    const provider = new SpecterViewProvider(context.extensionUri);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('specter-qwen-cockpit-v1', provider)
    );
}

async function fetchServerProjectId() {
    // Query the telemetry server's HTTP endpoint to get its project_id
    // This ensures the extension uses the same instanceId as the server
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
    constructor(extensionUri) {
        this._extensionUri = extensionUri;
        this._serverProjectId = null;  // Will be fetched from server HTTP endpoint
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

    async _detectMcpSessions() {
        // CRITICAL FIX: Fetch server's project_id from HTTP endpoint to ensure instanceId sync
        // The server generates instanceId from PID, extension cannot know it otherwise
        if (!this._serverProjectId) {
            this._serverProjectId = await fetchServerProjectId();
        }
        
        // If we got server's project_id, use it directly (server knows best)
        if (this._serverProjectId) {
            // Parse server's project_id to extract clientSource
            // Format: {instanceId}_{clientSource}_{workspaceHash}
            const parts = this._serverProjectId.split('_');
            const clientSource = parts.length >= 2 ? parts[1] : 'default';
            
            return [{
                id: clientSource,
                name: clientSource === 'roocode' ? 'Roo Code' :
                      clientSource === 'gemini' ? 'Gemini' : 'Default',
                projectId: this._serverProjectId,
                clientSource: clientSource
            }];
        }
        
        // Fallback: Detect MCP server configurations from Antigravity/Gemini config files
        // This is used when server is not running yet
        const sessions = [];
        const workspaceName = vscode.workspace.workspaceFolders?.[0]?.name || 'default';
        const workspaceHash = this._hashWorkspace(workspaceName);
        
        // P1-3 FIX: Use environment variables for portable paths (not hardcoded username)
        const userHome = process.env.USERPROFILE || process.env.HOME || os.homedir();
        
        // Path to Gemini/Antigravity MCP config
        const geminiConfigPath = path.join(userHome, '.gemini/antigravity/mcp_config.json');
        // Path to Roo Code MCP settings
        const rooConfigPath = path.join(userHome, 'AppData/Roaming/Antigravity/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json');
        
        // Check Roo Code config first (more likely to be active)
        try {
            if (fs.existsSync(rooConfigPath)) {
                const rooConfig = JSON.parse(fs.readFileSync(rooConfigPath, 'utf8'));
                const servers = rooConfig.mcpServers || {};
                
                // Check for any qwen-coding_roo or roo-specific server
                const hasRooQwenServer = Object.keys(servers).some(key =>
                    key.toLowerCase().includes('qwen') && key.toLowerCase().includes('_roo')
                );
                
                if (hasRooQwenServer) {
                    sessions.push({
                        id: 'roocode',
                        name: 'Roo Code',
                        projectId: `roocode_${workspaceHash}`,  // Will be updated when server starts
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
                
                // Check for any qwen-coding server in Gemini config
                const hasQwenServer = Object.keys(servers).some(key =>
                    key.toLowerCase().includes('qwen') && !key.toLowerCase().includes('_roo')
                );
                
                if (hasQwenServer) {
                    sessions.push({
                        id: 'gemini',
                        name: 'Gemini',
                        projectId: `gemini_${workspaceHash}`,  // Will be updated when server starts
                        clientSource: 'gemini'
                    });
                }
            }
        } catch (e) {
            console.log('[SPECTER] Failed to read Gemini config:', e.message);
        }
        
        // Fallback: always provide at least one default session
        if (sessions.length === 0) {
            sessions.push({
                id: 'default',
                name: 'Default',
                projectId: `default_${workspaceHash}`,  // Will be updated when server starts
                clientSource: 'default'
            });
        }
        
        return sessions;
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

        // Get detected sessions
        const sessions = this._detectMcpSessions();
        const sessionsJson = JSON.stringify(sessions).replace(/"/g, '"');

        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src ${webview.cspSource} 'unsafe-inline' 'unsafe-eval' blob:; connect-src ws://127.0.0.1:8878 http://127.0.0.1:8878 ws://localhost:8878 http://localhost:8878;">
                <link href="${styleUri}" rel="stylesheet">
                <title>Specter Cockpit</title>
                <script>
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

function deactivate() { }

module.exports = {
    activate,
    deactivate
};
