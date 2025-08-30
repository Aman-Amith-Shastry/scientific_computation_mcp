FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_TIMEOUT=60

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv add numpy \
    && uv add "mcp[cli]" \
    && uv add sympy \
    && uv add matplotlib \
    && uv add pydantic \
    && uv add uvicorn \
    && uv add starlette

# Expose the port (Smithery will set PORT env var)
EXPOSE 8081

# Start the MCP server - IMPORTANT: must use PORT env var from Smithery
CMD ["uv", "run", "src/main.py"]