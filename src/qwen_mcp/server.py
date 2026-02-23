import asyncio
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
async def qwen_refresh_models() -> str:
    """
    Checks for the latest SOTA models from Alibaba DashScope and identifies candidates.
    Note: This is now a passive check. Use qwen_list_available_models to see everything.
    """
    from qwen_mcp.api import DashScopeClient

    client = DashScopeClient()
    return await client.refresh_registry()


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
