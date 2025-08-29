FROM python:3.12-slim

RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install project dependencies
RUN uv sync --locked --no-install-project --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Set transport mode to HTTP
ENV TRANSPORT=http
ENV PORT=8081
EXPOSE 8081

# Run the application directly using the venv Python
CMD ["python", "src/server.py"]