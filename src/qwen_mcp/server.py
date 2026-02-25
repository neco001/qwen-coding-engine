from typing import Optional
from mcp.server.fastmcp import FastMCP, Context
from qwen_mcp.tools import (
    generate_audit,
    generate_code,
    generate_code_25,
    generate_lp_blueprint,
    read_repo_file,
    list_repo_files,
    generate_usage_report,
    list_available_models,
    set_model_in_registry,
    generate_sparring,
)

# Initialize FastMCP Server
mcp = FastMCP("Qwen MCP Server (DashScope)")


@mcp.tool()
async def qwen_audit(
    content: str, context: Optional[str] = None, ctx: Context = None
) -> str:
    """
    Audits the provided code or terminal logs using Qwen models.

    Args:
        content: The code snippet or terminal log content to analyze.
        context: Optional background context (e.g., project goals, related files).
    """
    return await generate_audit(content, context, ctx)


@mcp.tool()
async def qwen_coder(
    prompt: str, context: Optional[str] = None, ctx: Context = None
) -> str:
    """
    Generates or completes code using Qwen 3.5 Plus.
    """
    return await generate_code(prompt, context, ctx)


@mcp.tool()
async def qwen_coder_25(
    prompt: str, context: Optional[str] = None, ctx: Context = None
) -> str:
    """
    Generates or completes code using specialized Qwen-2.5-Coder-32B.
    """
    return await generate_code_25(prompt, context, ctx)


@mcp.tool()
async def qwen_architect(
    goal: str, context: Optional[str] = None, ctx: Context = None
) -> str:
    """
    Initiates 'The Lachman Protocol' (LP).
    The server hires a dynamic expert squad to audit your goal and generate a high-precision Blueprint.

    Args:
        goal: What do you want to achieve?
        context: Optional legacy code or background information.
    """
    if ctx:
        await ctx.report_progress(
            progress=0, total=None, message="Initiating Lachman Protocol..."
        )
    return await generate_lp_blueprint(goal, context, ctx)


@mcp.tool()
async def qwen_sparring(
    topic: str, context: str = "", mode: str = "flash", ctx: Context = None
) -> str:
    """
    Initiates the 5D Sparring Engine (Multi-Agent Strategic Debate).
    Moves the MCP from a coding assistant to a 'Cognitive Board of Directors'.

    Args:
        topic: The strategic dilemma or move to evaluate.
        context: Optional situational background (e.g., power structure, history).
        mode: 'flash' (Reasoning-only deep dive) or 'pro' (Adversarial Multi-Agent Debate).
    """
    return await generate_sparring(topic, context, mode, ctx)


@mcp.tool()
async def qwen_refresh_models() -> str:
    """
    Checks for the latest SOTA models from Alibaba DashScope and identifies candidates.
    Also synchronizes metadata from Hugging Face.
    """
    from qwen_mcp.api import DashScopeClient
    from qwen_mcp.registry import registry

    client = DashScopeClient()
    ds_res = await client.refresh_registry()
    hf_res = await registry.sync_with_hf()
    return f"DashScope: {ds_res}\nHugging Face: {hf_res}"


@mcp.tool()
async def qwen_list_available_models() -> str:
    """
    Fetches and displays the full list of models available via your DashScope API key.
    Use this to find IDs for qwen_set_model.
    """
    return await list_available_models()


@mcp.tool()
async def qwen_set_model(role: str, model_id: str) -> str:
    """
    Manually sets a specific model ID for a role.
    Roles: 'strategist' (audit/LP), 'coder' (qwen_coder_25), 'scout' (internal/discovery).
    """
    return await set_model_in_registry(role, model_id)


@mcp.tool()
async def qwen_read_file(path: str) -> str:
    """
    Reads a file from the local repository to use as context for your request.
    """
    return await read_repo_file(path)


@mcp.tool()
async def qwen_list_files(directory: str = ".", pattern: str = "**/*") -> str:
    """
    Lists files in the repository to discover context.
    """
    return await list_repo_files(directory, pattern)


@mcp.tool()
async def qwen_usage_report() -> str:
    """
    Retrieves the DuckDB token billing usage report and formats it as an aggregated table
    (Grouped by Date, Project, and Model).
    """
    return await generate_usage_report()


def main():
    """Main entrypoint for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
