# AI Installation Prompt: Qwen Engineering Engine

Copy and paste the following prompt to your AI assistant (e.g., Antigravity) to automatically install and configure this MCP server.

---

**Prompt:**

> "I want to install the Qwen Engineering Engine MCP server. Please follow these steps:
> 
> 1. Find my `mcp_config.json` file (usually in `%APPDATA%\Google\Antigravity` or similar for your environment).
> 2. Add a new server named `qwen-coding` with the following configuration:
>    - `command`: "uv"
>    - `args`: [
>        "--directory", 
>        "{{FULL_PATH_TO_THIS_REPOSITORY}}", 
>        "run", 
>        "qwen-coding-engine"
>      ]
>    - `env`: {
>        "DASHSCOPE_API_KEY": "{{YOUR_API_KEY}}",
>        "LP_MAX_RETRIES": "3"
>      }
> 3. Replace the placeholders with the absolute path to this folder and my actual DashScope API key.
> 4. Verify the configuration and restart the MCP connection."

---

**Note to User:** Before running this, ensure you have `uv` installed and your `DASHSCOPE_API_KEY` ready.
