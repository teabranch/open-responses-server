FROM python:3.12-slim AS builder

# Install uv
RUN pip install uv

# Set work directory
WORKDIR /app

# Copy only what's needed for installation
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies using uv with --system flag
RUN uv pip install --system --no-cache-dir .

# Create a lightweight runtime image
FROM python:3.12-slim AS runtime

# Set work directory
WORKDIR /app

# Copy only the installed packages and application code
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src

# Create log directory
RUN mkdir -p /app/log

# Define build arguments with default values
ARG OPENAI_BASE_URL_INTERNAL=http://localhost:8000
ARG OPENAI_BASE_URL=http://localhost:8080
ARG OPENAI_API_KEY=sk-fakekey
ARG API_ADAPTER_HOST=0.0.0.0
ARG API_ADAPTER_PORT=8080
ARG LOG_LEVEL=INFO
ARG LOG_FILE_PATH=/app/log/api_adapter.log

# Set environment variables from build arguments
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # OpenAI API Configuration
    OPENAI_BASE_URL_INTERNAL=${OPENAI_BASE_URL_INTERNAL} \
    OPENAI_BASE_URL=${OPENAI_BASE_URL} \
    OPENAI_API_KEY=${OPENAI_API_KEY} \
    # Server Configuration
    API_ADAPTER_HOST=${API_ADAPTER_HOST} \
    API_ADAPTER_PORT=${API_ADAPTER_PORT} \
    # Logging Configuration
    LOG_LEVEL=${LOG_LEVEL} \
    LOG_FILE_PATH=${LOG_FILE_PATH}

# Create entrypoint script for proper environment variable handling
RUN echo '#!/bin/sh\nexec uvicorn openai_responses_server.server:app --host $API_ADAPTER_HOST --port $API_ADAPTER_PORT "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Create non-root user for security and set permissions
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app/log && \
    chown appuser:appuser /app/entrypoint.sh
USER appuser

# Use JSON array format for CMD with entrypoint script
CMD ["sh", "/app/entrypoint.sh"] 