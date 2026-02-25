const vscode = require('vscode');
const path = require('path');
const fs = require('fs');

function activate(context) {
    const provider = new SpecterViewProvider(context.extensionUri);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('specter-qwen-hud-view', provider)
    );
}

class SpecterViewProvider {
    constructor(extensionUri) {
        this._extensionUri = extensionUri;
    }

    resolveWebviewView(webviewView) {
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
    }

    _getHtmlForWebview(webview) {
        // Path to built files in dist
        const distPath = path.join(this._extensionUri.fsPath, 'dist');
        // Find the JS and CSS files (they have hashes in names)
        const assetsPath = path.join(distPath, 'assets');
        const files = fs.readdirSync(assetsPath);

        const jsFile = files.find(f => f.endsWith('.js'));
        const cssFile = files.find(f => f.endsWith('.css'));

        const scriptUri = webview.asWebviewUri(vscode.Uri.file(path.join(assetsPath, jsFile)));
        const styleUri = webview.asWebviewUri(vscode.Uri.file(path.join(assetsPath, cssFile)));

        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link href="${styleUri}" rel="stylesheet">
                <title>Specter Cockpit</title>
                <style>
                    body { padding: 0; margin: 0; overflow: hidden; background-color: #0a0a0a; }
                    #root { height: 100vh; }
                </style>
            </head>
            <body>
                <div id="root"></div>
                <script type="module" src="${scriptUri}"></script>
            </body>
            </html>`;
    }
}

function deactivate() { }

module.exports = {
    activate,
    deactivate
};
