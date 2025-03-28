# Build stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PATH="/opt/venv/bin:$PATH"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create a non-root user
RUN useradd -m -s /bin/bash llmops

# Create app directories
RUN mkdir -p /app/config /app/logs /app/model_cache /app/prompt_cache \
    && chown -R llmops:llmops /app

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R llmops:llmops /app

# Switch to non-root user
USER llmops

# Create default config if it doesn't exist
RUN if [ ! -f /app/config/llmops.yaml ]; then \
    mkdir -p /app/config && \
    echo "aws:\n  region: us-west-2\n  role_arn: arn:aws:iam::123456789012:role/LLMOpsManagerRole\n\
gcp:\n  project_id: llmops-project\n  region: us-central1\n\
azure:\n  subscription_id: subscription-id\n  resource_group: llmops-resource-group\n  workspace_name: llmops-workspace\n\
deployment:\n  default_instance_type: ml.g4dn.xlarge\n  default_instance_count: 1\n\
monitoring:\n  collectors:\n    collection_interval_seconds: 60\n\
scaling:\n  check_interval_seconds: 60\n  default_min_replicas: 1\n  default_max_replicas: 10\n\
versioning:\n  model_registry:\n    storage_prefix: models\n    local_cache_dir: /app/model_cache\n\
  prompt_registry:\n    storage_prefix: prompts\n    local_cache_dir: /app/prompt_cache" > /app/config/llmops.yaml; \
fi

# Expose port for API
EXPOSE 8000

# Set environment variables for the application
ENV LLMOPS_CONFIG_PATH=/app/config/llmops.yaml \
    LLMOPS_PROVIDER=aws \
    LLMOPS_API_KEY=test-api-key \
    PYTHONPATH=/app

# Add version label
LABEL version="0.1.0"

# Command to run the application
CMD ["uvicorn", "src.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
