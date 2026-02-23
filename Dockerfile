# Use a slim Python image
FROM python:3.11-slim-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies using uv
# Note: We use --system to install to the system Python environment inside the container
RUN uv pip install --system -e .

# Set environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1

# Expose no ports as MCP typically works via stdio

# Set the entrypoint
ENTRYPOINT ["qwen-mcp"]
